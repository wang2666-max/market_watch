import os
from pathlib import Path

def load_env():
    """
    Minimal .env loader so we don't depend on python-dotenv yet.
    If you already installed python-dotenv, we can swap to that later.
    """
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        # don't overwrite real environment vars
        os.environ.setdefault(k, v)

def getenv_required(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise RuntimeError(f"Missing required env var: {key}")
    return val