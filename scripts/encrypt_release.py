#!/usr/bin/env python
"""Encrypt the gated release files (DESIGN_DECISIONS 4.6, 11.1, 12 item 23; Jacovi et al. 2023).

    python scripts/encrypt_release.py encrypt --release data/release/v0.4                 # every canary-bearing file
    python scripts/encrypt_release.py encrypt --release data/release/v0.4 --files noilai_test.jsonl attested.jsonl
    python scripts/encrypt_release.py verify  --release data/release/v0.4                 # decrypt in memory, compare hashes
    python scripts/encrypt_release.py decrypt --key-file data/release/v0.4/release_key.txt data/release/v0.4/encrypted/noilai_test.jsonl.enc
    python scripts/encrypt_release.py decrypt --key <hex> noilai_test.jsonl.enc --out noilai_test.jsonl

What is encrypted: by default every `.jsonl` of the release directory that opens with the BIG-bench header record
(the test-split files: `noilai_test`, `noilai_core`, `noilai_main`, the C2-enriched and scope files, every derived
core subset), i.e. the files whose rows carry the canary; the public dev file carries no header and is left alone.
`--files` names the set explicitly (e.g. to add `attested.jsonl`, whose vulgar rows stay in the gated split).

Format: AES-256-GCM (the `cryptography` package), one random 256-bit key per release drawn from `secrets` and
written ONLY to `<release>/release_key.txt` (hex; git-ignored like `manifest_private.json`; the gated README is
where it is published, DESIGN_DECISIONS 11.1); a fresh 96-bit nonce per file; the file name is bound as the
associated data so a ciphertext cannot be renamed into another file. The `.enc` file is the 16-byte magic
`NOILAI-ENC-v1\\n` padded, the nonce, then ciphertext + tag. `<out-dir>/encryption.json` records, per file, the
plaintext and ciphertext SHA-256 and sizes, the algorithm, the key's SHA-256 (never the key) and the exact decrypt
command; `verify` re-derives every hash from the key file. Nothing here reads or writes `manifest_private.json`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MAGIC = b"NOILAI-ENC-v1\n\0\0"          # 16 bytes
NONCE_BYTES = 12
KEY_BYTES = 32
ALGORITHM = "AES-256-GCM"
KEY_FILE = "release_key.txt"
ENC_DIR = "encrypted"
RECORD = "encryption.json"


def _aesgcm():
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as e:                                   # pragma: no cover - the dependency is in requirements.txt
        raise SystemExit("encrypt_release.py needs the `cryptography` package (pip install cryptography)") from e
    return AESGCM


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def new_key() -> bytes:
    return secrets.token_bytes(KEY_BYTES)


def read_key(text: str) -> bytes:
    """A key as hex (the key file's content); refused unless it is exactly 256 bits."""
    key = bytes.fromhex(text.strip())
    if len(key) != KEY_BYTES:
        raise ValueError(f"the key must be {KEY_BYTES} bytes ({KEY_BYTES * 2} hex characters), got {len(key)} bytes")
    return key


def load_key(key: str | None, key_file: Path | None) -> bytes:
    if (key is None) == (key_file is None):
        raise ValueError("give exactly one of --key and --key-file")
    return read_key(key if key is not None else Path(key_file).read_text(encoding="utf-8"))


def encrypt_bytes(plain: bytes, key: bytes, name: str) -> bytes:
    nonce = secrets.token_bytes(NONCE_BYTES)
    ct = _aesgcm()(key).encrypt(nonce, plain, name.encode("utf-8"))
    return MAGIC + nonce + ct


def decrypt_bytes(blob: bytes, key: bytes, name: str) -> bytes:
    """The plaintext; raises ValueError on a wrong key, a renamed file or a tampered ciphertext."""
    if len(blob) < len(MAGIC) + NONCE_BYTES + 16 or blob[:len(MAGIC)] != MAGIC:
        raise ValueError("not a NOILAI-ENC-v1 file")
    nonce = blob[len(MAGIC):len(MAGIC) + NONCE_BYTES]
    try:
        return _aesgcm()(key).decrypt(nonce, blob[len(MAGIC) + NONCE_BYTES:], name.encode("utf-8"))
    except Exception as e:                                     # InvalidTag and friends
        raise ValueError(f"decryption of {name} failed: wrong key, renamed file or tampered ciphertext") from e


def has_header_record(path: Path) -> bool:
    """The file opens with the DESIGN_DECISIONS 4.6 header record (`_header` + `canary`): a gated test-split file."""
    try:
        with open(path, encoding="utf-8") as f:
            first = f.readline()
        d = json.loads(first)
    except (OSError, ValueError):
        return False
    return isinstance(d, dict) and "_header" in d and bool(d.get("canary"))


def default_files(release: Path) -> list[Path]:
    return sorted(p for p in release.glob("*.jsonl") if has_header_record(p))


def plain_name(enc_path: Path) -> str:
    return enc_path.name.removesuffix(".enc")


def cmd_encrypt(args) -> int:
    release = Path(args.release)
    if not release.is_dir():
        raise SystemExit(f"release directory {release} does not exist")
    files = [release / f for f in args.files] if args.files else default_files(release)
    if not files:
        raise SystemExit(f"nothing to encrypt under {release}: no file opens with the header record; name files with --files")
    missing = [f for f in files if not f.exists()]
    if missing:
        raise SystemExit("missing: " + ", ".join(str(m) for m in missing))
    key_path = Path(args.key_file) if args.key_file else release / KEY_FILE
    if key_path.exists():
        key = read_key(key_path.read_text(encoding="utf-8"))
        key_source = f"existing {key_path}"
    else:
        key = new_key()
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_text(key.hex() + "\n", encoding="utf-8")
        try:
            key_path.chmod(0o600)
        except OSError:
            pass
        key_source = f"new key written to {key_path}"
    out_dir = Path(args.out_dir) if args.out_dir else release / ENC_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {"generated_by": "scripts/encrypt_release.py encrypt", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "algorithm": ALGORITHM, "format": "16-byte magic NOILAI-ENC-v1, 12-byte nonce, ciphertext + 16-byte tag; the file name is the associated data",
              "key_file": str(key_path), "key_sha256": sha256_bytes(key), "release": str(release),
              "decrypt": f"python scripts/encrypt_release.py decrypt --key-file {key_path} <file>.enc",
              "files": []}
    for f in files:
        plain = f.read_bytes()
        blob = encrypt_bytes(plain, key, f.name)
        target = out_dir / (f.name + ".enc")
        target.write_bytes(blob)
        record["files"].append({"name": f.name, "encrypted": target.name, "plain_sha256": sha256_bytes(plain), "plain_bytes": len(plain),
                                "cipher_sha256": sha256_bytes(blob), "cipher_bytes": len(blob), "header_record": has_header_record(f)})
    (out_dir / RECORD).write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"encrypted": [r["name"] for r in record["files"]], "out_dir": str(out_dir), "key": key_source,
                      "key_sha256": record["key_sha256"], "record": str(out_dir / RECORD)}, ensure_ascii=False, indent=1))
    return 0


def cmd_decrypt(args) -> int:
    key = load_key(args.key, args.key_file)
    enc = Path(args.file)
    name = args.name or plain_name(enc)
    plain = decrypt_bytes(enc.read_bytes(), key, name)
    out = Path(args.out) if args.out else enc.with_name(name)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; pass --force to overwrite")
    out.write_bytes(plain)
    print(json.dumps({"decrypted": str(out), "bytes": len(plain), "sha256": sha256_bytes(plain)}))
    return 0


def verify_release(release: Path, key: bytes, out_dir: Path | None = None) -> dict:
    """Decrypt every file of the record in memory and compare the hashes; a dict with `ok` and the per-file verdicts."""
    out_dir = out_dir or release / ENC_DIR
    rec_path = out_dir / RECORD
    if not rec_path.exists():
        raise FileNotFoundError(f"no {rec_path}: run `encrypt` first")
    record = json.loads(rec_path.read_text(encoding="utf-8"))
    verdicts = []
    for r in record["files"]:
        enc = out_dir / r["encrypted"]
        v = {"name": r["name"], "cipher_sha256_ok": None, "plain_sha256_ok": None, "plain_matches_release": None, "error": None}
        try:
            blob = enc.read_bytes()
            v["cipher_sha256_ok"] = sha256_bytes(blob) == r["cipher_sha256"]
            plain = decrypt_bytes(blob, key, r["name"])
            v["plain_sha256_ok"] = sha256_bytes(plain) == r["plain_sha256"]
            src = release / r["name"]
            v["plain_matches_release"] = (sha256_file(src) == r["plain_sha256"]) if src.exists() else None
        except (OSError, ValueError) as e:
            v["error"] = str(e)
        verdicts.append(v)
    ok = bool(verdicts) and all(v["cipher_sha256_ok"] and v["plain_sha256_ok"] and v["error"] is None for v in verdicts) \
        and sha256_bytes(key) == record["key_sha256"]
    return {"ok": ok, "key_sha256_matches_record": sha256_bytes(key) == record["key_sha256"], "files": verdicts}


def cmd_verify(args) -> int:
    release = Path(args.release)
    key = load_key(args.key, Path(args.key_file) if args.key_file else (release / KEY_FILE if args.key is None else None))
    out = verify_release(release, key, Path(args.out_dir) if args.out_dir else None)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0 if out["ok"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encrypt", help="encrypt the gated files of a release directory")
    e.add_argument("--release", type=Path, required=True)
    e.add_argument("--files", nargs="*", default=None, help="file names inside the release directory (default: every file with the header record)")
    e.add_argument("--key-file", type=Path, default=None, help=f"reuse / write the key here (default <release>/{KEY_FILE})")
    e.add_argument("--out-dir", type=Path, default=None, help=f"where the .enc files and {RECORD} go (default <release>/{ENC_DIR})")
    e.set_defaults(fn=cmd_encrypt)
    d = sub.add_parser("decrypt", help="decrypt one .enc file")
    d.add_argument("file")
    d.add_argument("--key", default=None, help="the key as hex")
    d.add_argument("--key-file", type=Path, default=None)
    d.add_argument("--name", default=None, help="the original file name when the .enc file was renamed (the associated data)")
    d.add_argument("--out", type=Path, default=None)
    d.add_argument("--force", action="store_true")
    d.set_defaults(fn=cmd_decrypt)
    v = sub.add_parser("verify", help=f"decrypt in memory and compare every hash of {RECORD}")
    v.add_argument("--release", type=Path, required=True)
    v.add_argument("--key", default=None)
    v.add_argument("--key-file", type=Path, default=None)
    v.add_argument("--out-dir", type=Path, default=None)
    v.set_defaults(fn=cmd_verify)
    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
