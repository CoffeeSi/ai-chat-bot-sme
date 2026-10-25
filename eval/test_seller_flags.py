"""Регрессии для роли продавца, миграции SQLite, ранжирования и промпта."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rmt_assignment3_project.config.rag import Index, Hit, build_chunks, format_record, retrieve
from rmt_assignment3_project.config.sellers import SellerIdentity, classify_answer
from rmt_assignment3_project.parsers.ozon_questions import parse_questions, save_questions, SCHEMA


class SellerFlagsTest(unittest.TestCase):
    identity = SellerIdentity("DAMELI", "https://cdn.example/shop.png", "https://ozon.kz/seller/dameli-3283894/")

    def test_explicit_false_overrides_matching_profile(self):
        answer = {"isSeller": False, "author": {"name": {"text": "DAMELI"}, "avatarUrl": self.identity.avatar_url}}
        self.assertEqual(classify_answer(answer, [self.identity]), (False, "api.isSeller"))

    def test_name_alone_and_legal_prefix_are_not_proof(self):
        for name in ("DAMELI", "ООО Покупатель"):
            self.assertEqual(classify_answer({"author": {"name": {"text": name}}}, [self.identity])[0], None)
        self.assertIsNone(classify_answer({"author": {"name": {"text": "DAMELI"}, "avatarUrl": "https://cdn.example/buyer.png"}}, [self.identity])[0])

    def test_parse_store_rebuild_rank_and_prompt(self):
        answer = {"answerUuid": "a1", "author": {"name": {"text": "DAMELI"}, "avatarUrl": self.identity.avatar_url}, "text": {"text": "Нанесите на влажную кожу."}}
        question = {"questionUuid": "q1", "itemId": "123", "text": {"text": "Способ применения?"}, "answers": [answer]}
        raw = {"widgetStates": {"webPDPListQuestions-1": {"questions": [question], "products": {"123": {"name": {"text": "Скраб"}}}}}}
        parsed = parse_questions(raw, [self.identity])
        self.assertTrue(parsed[0]["answers"][0]["is_seller"])
        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp) / "old.db")
            with sqlite3.connect(path) as con:
                con.executescript(SCHEMA)
            con.close()
            save_questions(path, parsed, "https://ozon.kz/product/test-123/")
            chunks = build_chunks(path)
        self.assertEqual(chunks[0]["prior"], 0.03)
        self.assertIn("Ответ (продавец)", format_record(1, Hit(chunks[0], .8, True)))
        other = dict(chunks[0], id="description", prior=0.0)
        idx = Index(np.array([[.8, .6], [.8, .6]], dtype=np.float32), [other, chunks[0]], {}, "test", None)

        class Embedder:
            def encode(self, texts):
                return np.array([[1., 0.]], dtype=np.float32)

        self.assertEqual(retrieve(idx, Embedder(), "test")[0].chunk["id"], "q1")


if __name__ == "__main__":
    unittest.main()
