from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


def get_deploy_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "sha": os.getenv("APP_BUILD_SHA") or "",
        "deployed_at": os.getenv("APP_BUILD_AT") or "",
        "workflow": os.getenv("APP_BUILD_WORKFLOW") or "",
        "run_id": os.getenv("APP_BUILD_RUN_ID") or "",
        "source": "env",
    }

    deploy_file = Path("DEPLOY_INFO.json")
    if deploy_file.exists():
        try:
            file_info = json.loads(deploy_file.read_text())
            if isinstance(file_info, dict):
                info.update({k: v for k, v in file_info.items() if v not in (None, "")})
                info["source"] = "file"
        except Exception:
            pass

    return info
