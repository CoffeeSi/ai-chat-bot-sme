"""Контракт API при перегрузке Gemini и идентификация модели ответа."""

import unittest

import numpy as np
from fastapi.testclient import TestClient

from rmt_assignment3_project.api.api import Settings, create_app
from rmt_assignment3_project.config.rag import Index, LLMError


class AvailabilityTest(unittest.TestCase):
    def client(self, llm):
        chunk = {"id": "q", "item_id": "123", "product": "Test", "date": None,
                 "question": "Use", "answers": [], "prior": 0., "source": "product", "content": "Test"}
        idx = Index(np.array([[1., 0.]], dtype=np.float32), [chunk], {"123": "Test"}, "hash", None)

        class Embedder:
            name = "test"
            def encode(self, texts):
                return np.array([[1., 0.]], dtype=np.float32)

        return TestClient(create_app(Settings(), idx=idx, embedder=Embedder(), llm=llm))

    def test_transient_upstream_is_reported_as_503(self):
        class Busy:
            model = "test-model"
            def complete(self, system, user):
                raise LLMError("upstream busy", upstream_status=503)

        with self.client(Busy()) as client:
            response = client.post("/ask", json={"question": "How to use?"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers["Retry-After"], "10")

    def test_response_identifies_actual_model(self):
        class Working:
            model = "test-model"
            def complete(self, system, user):
                return "Test answer"

        with self.client(Working()) as client:
            self.assertEqual(client.get("/health").json()["llm_model"], "test-model")
            response = client.post("/ask", json={"question": "How to use?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["llm_model"], "test-model")


if __name__ == "__main__":
    unittest.main()
