"""BM25 с положительным IDF: log(1+(N-df+0.5)/(df+0.5)).

Сохраняет токены документов. Оценка не является вероятностью и может превышать 1.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import math
import re
import threading
import unicodedata

import numpy as np
import snowballstemmer

TOKENIZER_VERSION = "unicode-snowball-ru-en-v1"
STOPWORDS = frozenset("а и в во на у к ко с со из для по о об от до при это этот эта эти того такой какая какой какие как что где когда сколько ли или же бы вы мы я он она они оно его ее их наш ваш мне меня вам пожалуйста здравствуйте добрый день туралы үшін және қандай қалай неше қай ма ме ба бе па пе бар".split())


class BM25Tokenizer:
    name = "bm25"

    def __init__(self):
        self.russian = snowballstemmer.stemmer("russian")
        self.english = snowballstemmer.stemmer("english")
        self.lock = threading.Lock()

    @lru_cache(maxsize=20000)
    def stem(self, word: str) -> str:
        if re.fullmatch(r"[а-я]+", word):
            return self.russian.stemWord(word)
        if re.fullmatch(r"[a-z]+", word):
            return self.english.stemWord(word)
        return word

    def tokenize(self, text: str) -> list[str]:
        text = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
        with self.lock:
            return [self.stem(word) for word in re.findall(r"[^\W_]+", text, flags=re.UNICODE)
                    if word not in STOPWORDS]


class BM25Index:
    def __init__(self, corpus: list[list[str]], ids: list[str], k1: float = 1.5, b: float = 0.75):
        if not corpus or len(corpus) != len(ids) or len(set(ids)) != len(ids):
            raise ValueError("Некорректный корпус BM25 или идентификаторы записей.")
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("BM25: k1 должен быть положительным, b — от 0 до 1.")
        self.corpus, self.ids, self.k1, self.b = corpus, ids, k1, b
        self.tokenizer = BM25Tokenizer()
        self.lengths = np.array([len(tokens) for tokens in corpus], dtype=np.float64)
        self.avgdl = float(self.lengths.mean()) or 1.0
        self.postings = defaultdict(list)
        for i, tokens in enumerate(corpus):
            for term, frequency in Counter(tokens).items():
                self.postings[term].append((i, frequency))

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.corpus), dtype=np.float64)
        for term in sorted(set(self.tokenizer.tokenize(query))):
            posting = self.postings.get(term, ())
            if not posting:
                continue
            idf = math.log1p((len(self.corpus) - len(posting) + 0.5) / (len(posting) + 0.5))
            for i, frequency in posting:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl)
                scores[i] += idf * frequency * (self.k1 + 1) / (frequency + norm)
        return scores

    def to_dict(self) -> dict:
        return {"version": 1, "tokenizer": TOKENIZER_VERSION, "k1": self.k1, "b": self.b,
                "ids": self.ids, "corpus": self.corpus}

    @classmethod
    def from_dict(cls, data: dict):
        if data.get("version") != 1 or data.get("tokenizer") != TOKENIZER_VERSION:
            raise ValueError("Версия BM25 изменилась: пересобери индекс.")
        return cls(data["corpus"], data["ids"], data["k1"], data["b"])
