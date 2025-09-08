from pathlib import Path
from einops import pack, unpack

REPO_PATH = repo_path = Path(__file__).resolve().parents[1]

def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


