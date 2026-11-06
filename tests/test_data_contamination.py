from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_data_ngrams import ngram_hashes
from prepare_data import Builder
from replay_study.data import Document, load_token_dataset


class IntegerTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [int(x) for x in text.split()]


def doc(start, stop, name):
    return Document(" ".join(map(str, range(start, stop))), name, "test", "train")


class ContaminationPolicyTests(unittest.TestCase):
    def test_shifted_duplicate_rejects_entire_document_and_refills(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = Builder(IntegerTokenizer(), Path(tmp), Path(tmp), seq_len=16)
            builder.collect("example", [doc(1, 33, "train")], {"train": 2},
                            lambda d, h: "train", "test", "fixed", {})
            train = load_token_dataset(Path(tmp) / "example_train.pt")["input_ids"].numpy()
            builder.reference_ngrams = np.unique(ngram_hashes(train))
            builder.reference_names = ["example_train"]
            builder.collect("example", [doc(2, 34, "shifted"), doc(101, 133, "clean")],
                            {"validation": 2}, lambda d, h: "validation", "test", "fixed", {}, .2)
            result = load_token_dataset(Path(tmp) / "example_validation.pt")
            self.assertEqual(result["doc_ids"], ["clean", "clean"])
            self.assertEqual(len(builder.ngram_document_rejections), 1)
            self.assertEqual(builder.ngram_document_rejections[0]["doc_id"], "shifted")


if __name__ == "__main__":
    unittest.main()
