import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from aegis_storage_client.encryption_engine import merkle_proof, merkle_root, merkle_verify


def _sha256_hex(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _sha256(data: bytes) -> bytes:
    import hashlib

    return hashlib.sha256(data).digest()


@dataclass(frozen=True)
class StoredFragmentMetadata:
    owner_id: str
    file_id: str
    fragment_index: int
    fragment_hash_hex: str
    file_merkle_root_hex: str
    block_size: int
    block_merkle_root_hex: str


@dataclass(frozen=True)
class PorChallenge:
    fragment_hash_hex: str
    block_index: int
    nonce_b64: str
    expected_block_merkle_root_hex: str


@dataclass(frozen=True)
class PorResponse:
    fragment_hash_hex: str
    block_index: int
    nonce_b64: str
    block_b64: str
    proof: List[Tuple[str, bool]]
    response_hash_hex: str


class StorageService:
    def __init__(self, base_dir: str, *, block_size: int = 4096):
        self.base_dir = Path(base_dir)
        self.block_size = int(block_size)
        self.fragments_dir = self.base_dir / "fragments"
        self.meta_dir = self.base_dir / "metadata"
        self.fragments_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    def put_fragment(
        self,
        *,
        owner_id: str,
        file_id: str,
        fragment_index: int,
        fragment_blob: bytes,
        file_merkle_root_hex: str,
        fragment_hash_hex: str,
    ) -> StoredFragmentMetadata:
        blob_hash = _sha256_hex(fragment_blob)
        if blob_hash != fragment_hash_hex:
            raise ValueError("fragment hash mismatch")

        blocks = self._split_blocks(fragment_blob)
        block_root = merkle_root(blocks).hex()

        meta = StoredFragmentMetadata(
            owner_id=str(owner_id),
            file_id=str(file_id),
            fragment_index=int(fragment_index),
            fragment_hash_hex=str(fragment_hash_hex),
            file_merkle_root_hex=str(file_merkle_root_hex),
            block_size=self.block_size,
            block_merkle_root_hex=block_root,
        )

        frag_path = self.fragments_dir / (fragment_hash_hex + ".bin")
        meta_path = self.meta_dir / (fragment_hash_hex + ".json")
        frag_path.write_bytes(fragment_blob)
        meta_path.write_text(json.dumps(meta.__dict__, indent=2, ensure_ascii=False), encoding="utf-8")
        return meta

    def has_fragment(self, fragment_hash_hex: str) -> bool:
        return (self.fragments_dir / (fragment_hash_hex + ".bin")).exists()

    def get_fragment(self, fragment_hash_hex: str) -> bytes:
        return (self.fragments_dir / (fragment_hash_hex + ".bin")).read_bytes()

    def get_metadata(self, fragment_hash_hex: str) -> StoredFragmentMetadata:
        raw = (self.meta_dir / (fragment_hash_hex + ".json")).read_text(encoding="utf-8")
        d = json.loads(raw)
        return StoredFragmentMetadata(**d)

    def create_por_challenge(self, fragment_hash_hex: str, *, block_index: Optional[int] = None) -> PorChallenge:
        meta = self.get_metadata(fragment_hash_hex)
        frag = self.get_fragment(fragment_hash_hex)
        blocks = self._split_blocks(frag)
        idx = int(block_index) if block_index is not None else int.from_bytes(os.urandom(2), "big") % len(blocks)
        nonce = os.urandom(16)
        return PorChallenge(
            fragment_hash_hex=fragment_hash_hex,
            block_index=idx,
            nonce_b64=base64.b64encode(nonce).decode("ascii"),
            expected_block_merkle_root_hex=meta.block_merkle_root_hex,
        )

    def answer_por_challenge(self, challenge: PorChallenge) -> PorResponse:
        meta = self.get_metadata(challenge.fragment_hash_hex)
        if meta.block_merkle_root_hex != challenge.expected_block_merkle_root_hex:
            raise ValueError("unexpected merkle root")

        frag = self.get_fragment(challenge.fragment_hash_hex)
        blocks = self._split_blocks(frag)
        if challenge.block_index < 0 or challenge.block_index >= len(blocks):
            raise ValueError("invalid block index")

        block = blocks[challenge.block_index]
        proof = merkle_proof(blocks, challenge.block_index)
        proof_ser = [(h.hex(), is_left) for (h, is_left) in proof]

        nonce = base64.b64decode(challenge.nonce_b64.encode("ascii"))
        response_hash = _sha256(
            nonce
            + challenge.block_index.to_bytes(4, "big")
            + bytes.fromhex(challenge.expected_block_merkle_root_hex)
            + _sha256(block)
        ).hex()

        return PorResponse(
            fragment_hash_hex=challenge.fragment_hash_hex,
            block_index=challenge.block_index,
            nonce_b64=challenge.nonce_b64,
            block_b64=base64.b64encode(block).decode("ascii"),
            proof=proof_ser,
            response_hash_hex=response_hash,
        )

    def verify_por_response_locally(self, challenge: PorChallenge, response: PorResponse) -> bool:
        if challenge.fragment_hash_hex != response.fragment_hash_hex:
            return False
        if challenge.block_index != response.block_index:
            return False
        if challenge.nonce_b64 != response.nonce_b64:
            return False
        block = base64.b64decode(response.block_b64.encode("ascii"))
        proof = [(bytes.fromhex(h), is_left) for (h, is_left) in response.proof]
        root = bytes.fromhex(challenge.expected_block_merkle_root_hex)
        if not merkle_verify(block, proof, root):
            return False

        nonce = base64.b64decode(challenge.nonce_b64.encode("ascii"))
        expected_hash = _sha256(
            nonce
            + challenge.block_index.to_bytes(4, "big")
            + root
            + _sha256(block)
        ).hex()
        return expected_hash == response.response_hash_hex

    def _split_blocks(self, blob: bytes) -> List[bytes]:
        if len(blob) == 0:
            return [b""]
        blocks: List[bytes] = []
        for i in range(0, len(blob), self.block_size):
            blocks.append(blob[i : i + self.block_size])
        return blocks

