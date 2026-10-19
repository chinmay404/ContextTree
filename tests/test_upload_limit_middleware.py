import asyncio

from app.api.upload_limit import LIMIT, UploadSizeLimitMiddleware

PATH = "/api/v1/boards/b1/sources/file"


def _scope(path=PATH, method="POST", headers=()):
    return {"type": "http", "method": method, "path": path, "headers": list(headers)}


class FakeApp:
    def __init__(self, read=True, stream=False):
        self.entered = False
        self.read = read
        self.stream = stream

    async def __call__(self, scope, receive, send):
        self.entered = True
        if self.read:
            while True:
                m = await receive()
                if not m.get("more_body"):
                    break
        await send({"type": "http.response.start", "status": 200, "headers": [(b"x-app", b"1")]})
        if self.stream:
            for i in range(3):
                await send({"type": "http.response.body", "body": b"part%d" % i, "more_body": True})
            await send({"type": "http.response.body", "body": b"", "more_body": False})
        else:
            await send({"type": "http.response.body", "body": b"app"})


def run(mw, scope, receive):
    sent = []

    async def send(m):
        sent.append(m)

    asyncio.run(mw(scope, receive, send))
    return sent


def test_declared_content_length_over_cap_rejected_app_not_entered():
    app = FakeApp()

    async def receive():
        raise AssertionError("body must not be read")

    sent = run(UploadSizeLimitMiddleware(app), _scope(headers=[(b"content-length", str(LIMIT + 1).encode())]), receive)
    assert sent[0]["status"] == 413 and b"too_large" in sent[1]["body"]
    assert app.entered is False


def test_chunked_body_past_cap_413_stops_reading_and_app_response_dropped():
    app = FakeApp()
    chunk = 256 * 1024
    consumed = []

    async def receive():
        consumed.append(1)
        assert len(consumed) <= LIMIT // chunk + 2, "receive() kept being consumed past the cap"
        return {"type": "http.request", "body": b"x" * chunk, "more_body": True}

    sent = run(UploadSizeLimitMiddleware(app), _scope(), receive)
    assert app.entered is True
    assert len(sent) == 2 and sent[0]["status"] == 413
    assert not any(m.get("body") == b"app" for m in sent)
    # receive() is not consumed beyond the chunk that crossed the cap
    assert len(consumed) == LIMIT // chunk + 1


def test_body_under_cap_passes_through():
    app = FakeApp()
    msgs = [{"type": "http.request", "body": b"x" * 1000, "more_body": True},
            {"type": "http.request", "body": b"", "more_body": False}]

    async def receive():
        return msgs.pop(0)

    sent = run(UploadSizeLimitMiddleware(app), _scope(), receive)
    assert sent[0]["status"] == 200 and sent[1]["body"] == b"app"


def test_non_upload_path_and_non_post_pass_through_untouched():
    big = [(b"content-length", str(LIMIT * 5).encode())]
    for scope in (_scope(path="/api/v1/boards", headers=big), _scope(method="GET", headers=big)):
        app = FakeApp(read=False, stream=True)

        async def receive():
            raise AssertionError("middleware must not touch receive")

        sent = run(UploadSizeLimitMiddleware(app), scope, receive)
        assert app.entered is True
        assert sent[0]["status"] == 200
        assert [m.get("body") for m in sent[1:]] == [b"part0", b"part1", b"part2", b""]
