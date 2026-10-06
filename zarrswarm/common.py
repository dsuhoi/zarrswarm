"""Hashing, canonical JSON and ed25519 node identity."""
import hashlib
import json
import os
from pathlib import Path

from cryptography.hazmat.primitives import serialization as ser
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def env(name: str, default=None):
    """Read a ZS_* setting, falling back to its legacy ZT_* name."""
    if name in os.environ:
        return os.environ[name]
    return os.environ.get(name.replace("ZS_", "ZT_", 1), default)


def state_home(home: str | Path | None = None) -> Path:
    """Use ~/.zs for new nodes; reuse existing ~/.zt state unless a home is supplied."""
    home = home if home is not None else env("ZS_HOME")
    if home:
        return Path(home).expanduser()
    current, legacy = Path("~/.zs").expanduser(), Path("~/.zt").expanduser()
    return legacy if not current.exists() and legacy.is_dir() else current


def protocol_headers(**values) -> dict[str, str]:
    """Emit ZS headers with legacy aliases so older peers can still verify replies."""
    return {f"X-{prefix}-{name}": value for name, value in values.items() for prefix in ("Zs", "Zt")}


def protocol_header(headers, name: str, default: str = "") -> str:
    """Prefer the canonical header even if its value is empty or invalid."""
    return headers.get(f"X-Zs-{name}", headers.get(f"X-Zt-{name}", default))


def cjson(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def h160(b: bytes) -> str:
    """160-bit id: DHT keys, node ids, grid/array family ids."""
    return hashlib.blake2b(b, digest_size=20).hexdigest()


def cid_of(b: bytes) -> str:
    """Content id of a stored (encoded) chunk."""
    return hashlib.blake2b(b, digest_size=16).hexdigest()


class Identity:
    def __init__(self, home: Path):
        home.mkdir(parents=True, exist_ok=True)
        p = home / "node.key"
        if p.exists():
            self.sk = Ed25519PrivateKey.from_private_bytes(p.read_bytes())
        else:
            self.sk = Ed25519PrivateKey.generate()
            raw = self.sk.private_bytes(ser.Encoding.Raw, ser.PrivateFormat.Raw, ser.NoEncryption())
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
        self.pk = self.sk.public_key().public_bytes(ser.Encoding.Raw, ser.PublicFormat.Raw).hex()
        self.id = h160(bytes.fromhex(self.pk))

    def sign(self, b: bytes) -> str:
        return self.sk.sign(b).hex()

    def signed(self, obj: dict) -> dict:
        obj = dict(obj, pk=self.pk)
        return dict(obj, sig=self.sign(cjson(obj)))


def verify(pk: str, sig: str, b: bytes) -> bool:
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(pk)).verify(bytes.fromhex(sig), b)
        return True
    except Exception:
        return False


def check_signed(obj: dict) -> bool:
    body = {k: v for k, v in obj.items() if k != "sig"}
    return verify(obj.get("pk", ""), obj.get("sig", ""), cjson(body))
