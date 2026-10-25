"""Fast regressions for query/document roles and evaluation isolation."""
import unittest
from unittest.mock import Mock

import numpy as np

from rmt_assignment3_project.config.rag import (
    E5Embedder, Index, answer_question_detailed, load_index, retrieve,
)


class SearchTests(unittest.TestCase):
    def test_e5_uses_different_prefixes_for_query_and_passage(self):
        embedder = E5Embedder.__new__(E5Embedder)
        embedder.model = Mock()
        embedder.model.encode.return_value = [[1.0, 0.0]]
        embedder.encode(["Көлемі 25 мл"])
        args, kwargs = embedder.model.encode.call_args
        self.assertEqual(args[0], ["passage: Көлемі 25 мл"])
        self.assertTrue(kwargs["normalize_embeddings"])
        embedder.encode_query("Көлемі қанша?")
        self.assertEqual(embedder.model.encode.call_args.args[0], ["query: Көлемі қанша?"])

    def test_semantic_search_preserves_original_kazakh_query(self):
        chunk = {"id": "x", "item_id": "1", "prior": 0.0}
        idx = Index(np.array([[1.0, 0.0]], dtype=np.float32), [chunk], {}, "e5", "test")
        encoder = Mock()
        encoder.encode_query.return_value = np.array([1.0, 0.0])
        hits = retrieve(idx, encoder, "Хош иістендіргіштер бар ма?")
        self.assertEqual(len(hits), 1)
        encoder.encode_query.assert_called_once_with("Хош иістендіргіштер бар ма?")
        encoder.encode.assert_not_called()

    def test_no_data_never_calls_generator(self):
        idx = load_index("data/rag_index")
        llm = Mock()
        result = answer_question_detailed(idx, None, llm, "нет информации", min_score=1e9)
        self.assertEqual(result.method, "no_data")
        self.assertFalse(result.hits)
        llm.complete.assert_not_called()

    def test_e5_scope_threshold_and_explicit_override(self):
        idx = Index(np.array([[0.75, 0.66144]], dtype=np.float32),
                    [{"id": "x", "item_id": "1", "prior": 0.0}], {}, "e5", "test")
        encoder = Mock()
        encoder.encode_query.return_value = np.array([1.0, 0.0])
        self.assertFalse(retrieve(idx, encoder, "question"))
        self.assertEqual(len(retrieve(idx, encoder, "question", "1")), 1)
        self.assertFalse(retrieve(idx, encoder, "question", "1", min_score=0.9))

    def test_template_is_distinguishable_and_can_be_disabled(self):
        idx = load_index("data/rag_index")
        llm = Mock()
        llm.complete.return_value = "Model response"
        question = "Қас гелінде хош иістендіргіштер бар ма?"
        result = answer_question_detailed(idx, None, llm, question, "3560833226", strict=True)
        self.assertEqual(result.method, "template")
        llm.complete.assert_not_called()
        generated = answer_question_detailed(idx, None, llm, question, "3560833226",
                                             strict=True, use_templates=False)
        self.assertEqual(generated.method, "llm")
        llm.complete.assert_called_once()


if __name__ == "__main__":
    unittest.main()
