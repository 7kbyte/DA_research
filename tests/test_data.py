import tempfile
from pathlib import Path
import unittest

import torch

from replay_study.data import (chunks_from_token_ids, document_hash, hash_partition,
                              load_token_dataset, reconstruct_wikitext_articles,
                              save_token_dataset, token_chunk_hash)


class DataBoundaryTests(unittest.TestCase):
    def test_article_reconstruction_keeps_sections_and_short_rows(self):
        lines = ["", " = First = \n", "short paragraph", " = = Section = = \n", "more",
                 " = Second = \n", "last"]
        docs = list(reconstruct_wikitext_articles(lines, "validation"))
        self.assertEqual(len(docs), 2)
        self.assertIn("Section", docs[0].text)
        self.assertIn("short paragraph", docs[0].text)
        self.assertIn("Second", docs[1].text)

    def test_document_chunking_never_packs_across_boundaries(self):
        a, _ = chunks_from_token_ids([1, 2, 3], 4, eos_token_id=0)
        b, discarded = chunks_from_token_ids([4, 5], 4, eos_token_id=0)
        self.assertEqual(a, [[1, 2, 3, 0]])
        self.assertEqual(b, [])
        self.assertEqual(discarded, 3)

    def test_normalization_duplicates_have_same_partition(self):
        a = document_hash("Ａ sample\ntext")
        b = document_hash("A  sample text")
        self.assertEqual(a, b)
        self.assertEqual(hash_partition(a), hash_partition(b))
        self.assertNotEqual(token_chunk_hash([1, 2]), token_chunk_hash([2, 1]))

    def test_roundtrip_counts_prediction_tokens(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "example.pt"
            manifest = save_token_dataset(path, [[1, 2, 3, 4], [5, 6, 7, 8]], ["a", "b"], {})
            data = load_token_dataset(path)
            self.assertEqual(data["input_ids"].dtype, torch.int32)
            self.assertEqual(manifest["prediction_tokens"], 6)
            self.assertEqual(manifest["raw_input_tokens"], 8)


if __name__ == "__main__":
    unittest.main()
