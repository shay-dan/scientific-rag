"""Offline regression coverage for precommitted blind allocation."""
import copy
import json
import unittest

from app.benchmark.generation_evaluation import (
    build_blind_scoring_packet, canonical_sha256, create_blind_key, validate_blind_key,
)
from tests.test_r7d_generation_evaluation import _manifest


class BlindAllocationTests(unittest.TestCase):
    def setUp(self):
        self.manifest = _manifest()
        self.key = create_blind_key(self.manifest)
        self.commitment = canonical_sha256(self.key)
        self.records = [
            {"scenario_id": row["scenario_id"], "arms": [
                {"prompt_version": v, "status": "completed", "post_repair_output": {"claims": []}}
                for v in ("v15", "v16")]}
            for row in self.manifest["scenarios"]]

    def test_balanced_and_salted(self):
        self.assertEqual(10, sum(row["A"] == "v15" for row in self.key["arms"]))
        self.assertNotEqual(self.commitment, canonical_sha256(create_blind_key(self.manifest)))

    def test_repeat_export_and_no_metadata_leak(self):
        packet, key = build_blind_scoring_packet(self.manifest, self.records, key=self.key,
                                                  commitment=self.commitment)
        self.assertEqual(key, self.key)
        self.assertEqual(packet, build_blind_scoring_packet(self.manifest, self.records[::-1], key=self.key,
                                                            commitment=self.commitment)[0])
        for secret in ("v15", "v16", self.key["nonce"], self.commitment):
            self.assertNotIn(secret, json.dumps(packet))

    def test_tampering_and_missing_key(self):
        for key in (None, create_blind_key(self.manifest), dict(self.key, nonce="0" * 64)):
            with self.subTest(key=type(key).__name__), self.assertRaises(ValueError):
                validate_blind_key(self.manifest, key, self.commitment)

    def test_rehashed_invalid_allocations(self):
        for mode in ("duplicate", "unbalanced", "missing", "wrong_manifest"):
            key = copy.deepcopy(self.key)
            if mode == "duplicate":
                key["arms"][1] = key["arms"][0]
            elif mode == "unbalanced":
                for row in key["arms"]:
                    row.update(A="v15", B="v16")
            elif mode == "missing":
                key["arms"].pop()
            else:
                key["manifest_sha256"] = "0" * 64
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_blind_key(self.manifest, key, canonical_sha256(key))

    def test_duplicate_missing_records_refused(self):
        for records in (self.records[:-1], self.records + [self.records[0]]):
            with self.assertRaises(ValueError):
                build_blind_scoring_packet(self.manifest, records, key=self.key, commitment=self.commitment)
