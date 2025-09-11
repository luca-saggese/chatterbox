#!/usr/bin/env python3
"""
Regression test for Italian accented letters in the multilingual tokenizer.
Verifies encode/decode round-trip preserves original text including problematic characters like 'ì'.
"""

import os
import unittest
from pathlib import Path

from huggingface_hub import snapshot_download

# Make src importable
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chatterbox.models.tokenizers import MTLTokenizer


class TestItalianAccents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Download artifacts to locate tokenizer file
        ckpt_dir = Path(
            snapshot_download(
                repo_id="ResembleAI/chatterbox",
                repo_type="model",
                revision="main",
                allow_patterns=["mtl_tokenizer.json"],
                token=os.getenv("HF_TOKEN"),
            )
        )
        cls.tokenizer = MTLTokenizer(str(ckpt_dir / "mtl_tokenizer.json"))

    def round_trip(self, text: str):
        ids = self.tokenizer.encode(text, language_id="it")
        out = self.tokenizer.decode(ids)
        self.assertEqual(out, text, f"Round-trip mismatch: '{text}' -> '{out}'")

    def test_single_chars(self):
        for ch in ["à","è","é","ì","ò","ù","À","È","É","Ì","Ò","Ù"]:
            with self.subTest(ch=ch):
                self.round_trip(ch)

    def test_common_words(self):
        words = [
            "città", "perché", "più", "così", "università", "lì", "dì", "però", "più", "andrò",
        ]
        for w in words:
            with self.subTest(word=w):
                self.round_trip(w)

    def test_phrases(self):
        phrases = [
            "È lì dov'è l'università di città.",
            "Così com'è, più tardi andrò lì.",
            "L'ìtaliano con accenti: è, à, ì, ò, ù.",
        ]
        for p in phrases:
            with self.subTest(phrase=p):
                self.round_trip(p)


if __name__ == "__main__":
    unittest.main()
