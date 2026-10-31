"""
A YouTube video's words, for video sources.

Captions come from `youtube-transcript-api`. It is unofficial, and YouTube blocks most cloud IPs,
so a request from the deployed engine can be refused where the same request from home works:
`YOUTUBE_PROXY_URL` routes it through a proxy, and a refusal is reported as `blocked`, never as
"this video has no captions". The Gemini fallback asks Google's model to transcribe the public
video instead; it costs quota, so ingest only calls it when switched on (`YOUTUBE_GEMINI_FALLBACK`).

A transcript is a list of parts, {start: seconds, text}. The video id is checked against YT_ID
before it gets here, and only YouTube and Google are contacted, so `safe_fetch` has no role.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Optional

import httpx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import (
    NoTranscriptFound, RequestBlocked, TranscriptsDisabled, YouTubeRequestFailed, YouTubeTranscriptApi,
)
from youtube_transcript_api.proxies import GenericProxyConfig

from app.core.config import settings
from app.core.logger import logger

YT_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
CAPTIONS_TIMEOUT_SECONDS = 25
GEMINI_TIMEOUT_SECONDS = 200  # long videos take minutes; the ingest route allows 300 s in all
CHUNK_CHARS = 1000

_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_GEMINI_PROMPT = (
    "Transcribe what is said in this video, word for word, in the language it is spoken in. "
    "Start each line with its time as [MM:SS] or [H:MM:SS]. Output only the transcript. "
    "If no words are spoken, output nothing."
)
_SOUND_ONLY = re.compile(r"^(\[[^\]]*\]|\([^)]*\)|[♪♫\s]+)$")
_TIMED_LINE = re.compile(r"^\s*\[?(?:(\d{1,2}):)?(\d{1,2}):(\d{2})\]?(?:\s+(.*))?$")

# A hung request is abandoned, not waited for (the same pattern as web_query).
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="youtube")


class TranscriptError(Exception):
    """Why there is no transcript: `no_captions`, `blocked` or `failed`."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _api() -> YouTubeTranscriptApi:
    proxy = settings.YOUTUBE_PROXY_URL
    return YouTubeTranscriptApi(
        proxy_config=GenericProxyConfig(http_url=proxy, https_url=proxy) if proxy else None)


def _fetch_rows(video_id: str) -> list:
    available = _api().list(video_id)
    try:
        chosen = available.find_transcript(["en"])
    except NoTranscriptFound:
        chosen = next(iter(available), None)
    if chosen is None:
        raise TranscriptError("no_captions")
    return chosen.fetch().to_raw_data()


def fetch_captions(video_id: str) -> list[dict]:
    """The video's captions as parts, in English when there are English ones, else any language."""
    try:
        rows = _executor.submit(_fetch_rows, video_id).result(timeout=CAPTIONS_TIMEOUT_SECONDS)
    except TranscriptError:
        raise
    except (TranscriptsDisabled, NoTranscriptFound):
        raise TranscriptError("no_captions") from None
    except (RequestBlocked, YouTubeRequestFailed):  # IpBlocked is a RequestBlocked
        raise TranscriptError("blocked") from None
    except FutureTimeout:
        raise TranscriptError("failed") from None
    except Exception as exc:  # noqa: BLE001 - the library raises many kinds; none reaches the user
        logger.info("Caption fetch failed (%s)", type(exc).__name__)
        raise TranscriptError("failed") from None
    parts = []
    for row in rows:
        text = " ".join(str(row.get("text") or "").split())
        if text and not _SOUND_ONLY.match(text):
            parts.append({"start": float(row.get("start") or 0), "text": text})
    if not parts:
        raise TranscriptError("no_captions")
    return parts


def parse_timed_lines(text: str) -> list[dict]:
    """Parts from lines that start with a time; other lines join the line before them."""
    parts: list[dict] = []
    for line in str(text or "").splitlines():
        match = _TIMED_LINE.match(line)
        if match:
            hours, minutes, seconds, rest = match.groups()
            start = int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds)
            parts.append({"start": float(start), "text": " ".join((rest or "").split())})
        elif parts and line.strip():
            parts[-1]["text"] = f"{parts[-1]['text']} {' '.join(line.split())}".strip()
    return [p for p in parts if p["text"]]


def fetch_ai_transcript(video_id: str) -> list[dict]:
    """Gemini's transcript of a public video as parts; [] on any failure. The key stays in a header."""
    key = settings.GOOGLE_API_KEY
    if not key:
        return []
    body = {"contents": [{"parts": [
        {"fileData": {"fileUri": f"https://www.youtube.com/watch?v={video_id}"}},
        {"text": _GEMINI_PROMPT},
    ]}]}
    try:
        reply = httpx.post(_GEMINI_URL.format(model=settings.YOUTUBE_GEMINI_MODEL), json=body,
                           headers={"x-goog-api-key": key}, timeout=GEMINI_TIMEOUT_SECONDS)
        reply.raise_for_status()
        parts = reply.json()["candidates"][0]["content"]["parts"]
        return parse_timed_lines("\n".join(str(p.get("text") or "") for p in parts))
    except Exception as exc:  # noqa: BLE001 - never logs the reply, which can echo the request
        logger.warning("Gemini transcript failed (%s)", type(exc).__name__)
        return []


def timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def duration_label(parts: list[dict]) -> str:
    """Roughly how long the video is, from when its last line starts."""
    minutes = max(1, round((parts[-1]["start"] if parts else 0) / 60))
    hours, rest = divmod(minutes, 60)
    return f"{hours} h {rest} min" if hours else f"{minutes} min"


def transcript_chunks(parts: list[dict]) -> tuple[list[str], list[dict], str]:
    """(chunks of up to CHUNK_CHARS, metadatas with a unique time `loc` and `start_s`, the full text)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_CHARS, chunk_overlap=0)
    chunks: list[str] = []
    metas: list[dict] = []

    def flush(text: str, start: Optional[float]) -> None:
        if text:
            for i, piece in enumerate(splitter.split_text(text)):
                loc = timestamp(start or 0)
                chunks.append(piece)
                metas.append({"loc": f"{loc} ({i + 1})" if i else loc, "start_s": int(start or 0)})

    current, began = "", None
    for part in parts:
        if current and len(current) + 1 + len(part["text"]) > CHUNK_CHARS:
            flush(current, began)
            current, began = "", None
        began = part["start"] if began is None else began
        current = f"{current} {part['text']}" if current else part["text"]
    flush(current, began)
    return chunks, metas, "\n".join(f"[{timestamp(p['start'])}] {p['text']}" for p in parts)
