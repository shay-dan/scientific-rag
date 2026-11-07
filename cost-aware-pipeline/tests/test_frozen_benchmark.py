import unittest

from app.frozen_benchmark import frozen_corpus_rows, snapshot_documents, snapshot_hash, snapshot_manifest


class FrozenBenchmarkTests(unittest.TestCase):
    def _payload(self):
        return {
            "cases": [
                {
                    "id": "one",
                    "question": "SECRET QUESTION",
                    "reference_answer": "SECRET ANSWER",
                    "reference_claims": ["SECRET CLAIM"],
                    "reference_context": [
                        {"context_id": "c1", "source_id": "PMID:1", "title": "A", "url": "https://pubmed.ncbi.nlm.nih.gov/1/", "excerpt": "evidence alpha"},
                        {"context_id": "c2", "source_id": "PMID:1", "title": "A", "url": "https://pubmed.ncbi.nlm.nih.gov/1/", "excerpt": "evidence beta"},
                    ],
                },
                {
                    "id": "two",
                    "reference_context": [
                        {"context_id": "c3", "source_id": "arXiv:2", "title": "B", "url": "https://arxiv.org/abs/2", "excerpt": "evidence gamma"}
                    ],
                },
            ]
        }

    def test_shared_corpus_excludes_questions_answers_and_claims(self):
        rows = frozen_corpus_rows(self._payload())
        serialized = repr(rows)
        self.assertNotIn("SECRET QUESTION", serialized)
        self.assertNotIn("SECRET ANSWER", serialized)
        self.assertNotIn("SECRET CLAIM", serialized)
        self.assertEqual(len(rows), 3)

    def test_manifest_is_stable_and_reports_unique_sources(self):
        first = snapshot_manifest(self._payload())
        second = snapshot_manifest(self._payload())
        self.assertEqual(first["snapshot_hash"], second["snapshot_hash"])
        self.assertEqual(first["unique_source_count"], 2)
        self.assertFalse(first["contains_reference_answers"])

    def test_documents_retain_source_identity_only(self):
        documents = snapshot_documents(self._payload())
        self.assertEqual(documents[0].metadata["source"], "pubmed")
        self.assertTrue(all("reference_answer" not in item.metadata for item in documents))

    def test_hash_changes_when_corpus_changes(self):
        rows = frozen_corpus_rows(self._payload())
        original = snapshot_hash(rows)
        rows[0]["excerpt"] += " changed"
        self.assertNotEqual(original, snapshot_hash(rows))


if __name__ == "__main__":
    unittest.main()
