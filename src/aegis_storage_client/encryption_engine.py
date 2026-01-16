import hashlib
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes


def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _hkdf_32(key_material: bytes, salt: bytes, info: bytes) -> bytes:
    hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=salt, info=info)
    return hkdf.derive(key_material)


class GF256:
    _prim = 0x11D

    def __init__(self):
        self.exp = [0] * 512
        self.log = [0] * 256
        x = 1
        for i in range(255):
            self.exp[i] = x
            self.log[x] = i
            x <<= 1
            if x & 0x100:
                x ^= self._prim
        for i in range(255, 512):
            self.exp[i] = self.exp[i - 255]

    def add(self, a: int, b: int) -> int:
        return a ^ b

    def sub(self, a: int, b: int) -> int:
        return a ^ b

    def mul(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return self.exp[self.log[a] + self.log[b]]

    def inv(self, a: int) -> int:
        if a == 0:
            raise ZeroDivisionError
        return self.exp[255 - self.log[a]]


_GF = GF256()


def _mat_mul(a: List[List[int]], b: List[List[int]]) -> List[List[int]]:
    rows = len(a)
    cols = len(b[0])
    inner = len(b)
    out = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        for k in range(inner):
            aik = a[i][k]
            if aik == 0:
                continue
            for j in range(cols):
                out[i][j] ^= _GF.mul(aik, b[k][j])
    return out


def _mat_inv(m: List[List[int]]) -> List[List[int]]:
    n = len(m)
    aug = [row[:] + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(m)]
    for col in range(n):
        pivot = None
        for r in range(col, n):
            if aug[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            raise ValueError("matrix singular")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        inv_p = _GF.inv(aug[col][col])
        for c in range(col, 2 * n):
            aug[col][c] = _GF.mul(aug[col][c], inv_p)
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0:
                continue
            for c in range(col, 2 * n):
                aug[r][c] ^= _GF.mul(factor, aug[col][c])
    return [row[n:] for row in aug]


def _parity_row(base: int, k: int) -> List[int]:
    row = [1]
    x = 1
    for _ in range(1, k):
        x = _GF.mul(x, base)
        row.append(x)
    return row


def _gen_matrix(n: int, k: int) -> List[List[int]]:
    rows: List[List[int]] = []
    for i in range(k):
        rows.append([1 if j == i else 0 for j in range(k)])
    for i in range(k, n):
        rows.append(_parity_row(i - k + 1, k))
    return rows


def reed_solomon_encode(data: bytes, k: int, n: int) -> Tuple[List[bytes], int]:
    if not (1 <= k <= n <= 255):
        raise ValueError("invalid k/n")
    if len(data) == 0:
        raise ValueError("empty data")

    shard_size = (len(data) + k - 1) // k
    padded = data + b"\x00" * (shard_size * k - len(data))
    data_shards = [padded[i * shard_size : (i + 1) * shard_size] for i in range(k)]

    gen = _gen_matrix(n, k)
    out: List[bytes] = []
    for i in range(n):
        if i < k:
            out.append(data_shards[i])
            continue
        coeffs = gen[i]
        buf = bytearray(shard_size)
        for j in range(k):
            c = coeffs[j]
            if c == 0:
                continue
            sj = data_shards[j]
            for b_idx in range(shard_size):
                buf[b_idx] ^= _GF.mul(c, sj[b_idx])
        out.append(bytes(buf))
    return out, len(data)


def reed_solomon_decode(shards: Dict[int, bytes], k: int, n: int, original_len: int) -> bytes:
    if len(shards) < k:
        raise ValueError("insufficient shards")
    shard_size = len(next(iter(shards.values())))
    used = sorted(shards.items(), key=lambda x: x[0])[:k]
    idxs = [i for i, _ in used]
    a_rows = _gen_matrix(n, k)
    a = [a_rows[i] for i in idxs]
    inv = _mat_inv(a)

    data_shards: List[bytearray] = [bytearray(shard_size) for _ in range(k)]
    for out_i in range(k):
        coeffs = inv[out_i]
        for j in range(k):
            c = coeffs[j]
            if c == 0:
                continue
            sj = shards[idxs[j]]
            for b_idx in range(shard_size):
                data_shards[out_i][b_idx] ^= _GF.mul(c, sj[b_idx])
    joined = b"".join(bytes(s) for s in data_shards)
    return joined[:original_len]


def merkle_root(leaves: Sequence[bytes]) -> bytes:
    if not leaves:
        raise ValueError("no leaves")
    level = [_sha256(x) for x in leaves]
    while len(level) > 1:
        nxt: List[bytes] = []
        it = iter(level)
        for a in it:
            b = next(it, a)
            nxt.append(_sha256(a + b))
        level = nxt
    return level[0]


def merkle_proof(leaves: Sequence[bytes], index: int) -> List[Tuple[bytes, bool]]:
    if index < 0 or index >= len(leaves):
        raise IndexError
    level = [_sha256(x) for x in leaves]
    proof: List[Tuple[bytes, bool]] = []
    idx = index
    while len(level) > 1:
        if idx % 2 == 0:
            sib_idx = idx + 1
            is_left = False
        else:
            sib_idx = idx - 1
            is_left = True
        sibling = level[sib_idx] if sib_idx < len(level) else level[idx]
        proof.append((sibling, is_left))

        nxt: List[bytes] = []
        for i in range(0, len(level), 2):
            a = level[i]
            b = level[i + 1] if i + 1 < len(level) else a
            nxt.append(_sha256(a + b))
        level = nxt
        idx //= 2
    return proof


def merkle_verify(leaf: bytes, proof: Sequence[Tuple[bytes, bool]], root: bytes) -> bool:
    h = _sha256(leaf)
    for sibling, sibling_is_left in proof:
        h = _sha256(sibling + h) if sibling_is_left else _sha256(h + sibling)
    return h == root


@dataclass(frozen=True)
class EncryptedFile:
    file_key: bytes
    chacha_nonce: bytes
    ciphertext: bytes


@dataclass(frozen=True)
class EncryptedFragment:
    index: int
    n: int
    k: int
    aes_nonce: bytes
    ciphertext: bytes

    def blob(self) -> bytes:
        return (
            self.index.to_bytes(2, "big")
            + self.n.to_bytes(2, "big")
            + self.k.to_bytes(2, "big")
            + self.aes_nonce
            + self.ciphertext
        )


def parse_fragment_blob(blob: bytes) -> EncryptedFragment:
    if len(blob) < 2 + 2 + 2 + 12 + 16:
        raise ValueError("fragment blob too small")
    idx = int.from_bytes(blob[0:2], "big")
    n = int.from_bytes(blob[2:4], "big")
    k = int.from_bytes(blob[4:6], "big")
    nonce = blob[6:18]
    ct = blob[18:]
    return EncryptedFragment(index=idx, n=n, k=k, aes_nonce=nonce, ciphertext=ct)


def encrypt_file_chacha(data: bytes) -> EncryptedFile:
    file_key = os.urandom(32)
    nonce = os.urandom(12)
    aead = ChaCha20Poly1305(file_key)
    ciphertext = aead.encrypt(nonce, data, None)
    return EncryptedFile(file_key=file_key, chacha_nonce=nonce, ciphertext=ciphertext)


def decrypt_file_chacha(enc: EncryptedFile) -> bytes:
    aead = ChaCha20Poly1305(enc.file_key)
    return aead.decrypt(enc.chacha_nonce, enc.ciphertext, None)


def _derive_fragment_key(file_key: bytes, index: int) -> bytes:
    salt = _sha256(file_key)
    info = b"aegis_storage_fragment_key:" + index.to_bytes(4, "big")
    return _hkdf_32(file_key, salt=salt, info=info)


def encrypt_fragments(
    *,
    encrypted_file_ciphertext: bytes,
    file_key: bytes,
    k: int,
    n: int,
) -> Tuple[List[EncryptedFragment], int, bytes]:
    shards, original_len = reed_solomon_encode(encrypted_file_ciphertext, k=k, n=n)
    out: List[EncryptedFragment] = []
    leaf_blobs: List[bytes] = []
    for idx, shard in enumerate(shards):
        frag_key = _derive_fragment_key(file_key, idx)
        nonce = os.urandom(12)
        aes = AESGCM(frag_key)
        ct = aes.encrypt(nonce, shard, idx.to_bytes(2, "big"))
        frag = EncryptedFragment(index=idx, n=n, k=k, aes_nonce=nonce, ciphertext=ct)
        out.append(frag)
        leaf_blobs.append(frag.blob())
    root = merkle_root(leaf_blobs)
    return out, original_len, root


def decrypt_and_reconstruct(
    *,
    fragments: Sequence[EncryptedFragment],
    file_key: bytes,
    original_len: int,
    k: int,
    n: int,
) -> bytes:
    shard_map: Dict[int, bytes] = {}
    for frag in fragments:
        if frag.index in shard_map:
            continue
        key = _derive_fragment_key(file_key, frag.index)
        aes = AESGCM(key)
        shard = aes.decrypt(frag.aes_nonce, frag.ciphertext, frag.index.to_bytes(2, "big"))
        shard_map[frag.index] = shard
    ciphertext = reed_solomon_decode(shard_map, k=k, n=n, original_len=original_len)
    return ciphertext

