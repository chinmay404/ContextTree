from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


limiter = Limiter(key_func=get_remote_address, default_limits=["100/hour"])

# Board routes are limited per verified user (all traffic arrives from the proxy's IPs).
BOARD_RATE = "600/minute"


def user_key(request: Request) -> str:
    """
    SlowAPI key function. Per-user limits, not per-IP, so users behind the
    same NAT don't block each other. request.state.user_id is set by the
    auth dependency from the verified JWT — it cannot be spoofed via the body.
    """
    uid = getattr(request.state, "user_id", None)
    if uid:
        return uid
    # Unauthenticated requests (will be rejected by the dependency anyway)
    return request.client.host if request.client else "unknown"
