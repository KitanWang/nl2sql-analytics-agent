"""少样本示例检索 —— 自己实现的 BM25，不引第三方依赖。

## 为什么不用向量检索

1. **没有 embedding 服务时它不可用**。演示环境只有 LLM API，多引一个 embedding 依赖
   就多一个失败点；
2. **示例库只有几十条**。在这个规模下 BM25 的召回和向量检索差别很小，
   而 BM25 是**确定性**的——同样的输入永远同样的输出，评测可复现；
3. 中文没有空格分词，这里用 **字符二元组（bigram）** 作为 token。
   实测对「上个月销售额最高的品类」这类短查询，bigram 的召回优于不装分词的按空格切分。

这个模块同时也证明了「检索增强」不一定等于「上向量库」——
选对检索方式比选贵的检索方式更重要。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

_CJK = re.compile(r"[\u4e00-\u9fff]")
_ASCII_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]+")
_DIGIT = re.compile(r"\d+")


def tokenize(text: str) -> list[str]:
    """中英混排分词：中文取字符 bigram，英文/数字取整词。

    例：「2025年6月华东区销售额」→
        ['2025', '6', '年6', '6月', '月华', '华东', '东区', '区销', '销售', '售额']
    单字也保留（unigram），避免短查询（如「退款率」）在 bigram 下匹配不到。
    """
    t = (text or "").lower()
    tokens: list[str] = []

    # 英文词与数字整体保留
    tokens += _ASCII_WORD.findall(t)
    tokens += _DIGIT.findall(t)

    # 中文连续片段 → unigram + bigram
    for seg in re.findall(r"[\u4e00-\u9fff]+", t):
        tokens += list(seg)
        tokens += [seg[i:i + 2] for i in range(len(seg) - 1)]

    return tokens


@dataclass
class Example:
    question: str
    sql: str
    why: str = ""
    tags: tuple[str, ...] = ()

    def as_prompt_pair(self) -> dict[str, str]:
        return {"question": self.question, "sql": self.sql}


class BM25Retriever:
    """标准 BM25（k1=1.5, b=0.75）。

    公式：score(q,d) = Σ idf(t) · (tf·(k1+1)) / (tf + k1·(1 − b + b·|d|/avgdl))
    Δ：对「问题本身」与「问题 + 业务标签」拼接后建索引，
    标签（如 `口径`、`排名`、`时间窗口`）让同类问题互相靠近。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: list[list[str]] = []
        self._examples: list[Example] = []
        self._df: Counter = Counter()
        self._avgdl: float = 0.0
        self._n: int = 0

    def fit(self, examples: list[Example]) -> "BM25Retriever":
        self._examples = list(examples)
        self._docs = [tokenize(e.question + " " + " ".join(e.tags)) for e in self._examples]
        self._n = len(self._docs)
        self._df = Counter()
        for d in self._docs:
            self._df.update(set(d))
        self._avgdl = sum(len(d) for d in self._docs) / max(1, self._n)
        return self

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        # 加 0.5 平滑，避免 df=0 时除零；df 越大（如「的」）idf 越小
        return math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def score(self, query: str, idx: int) -> float:
        q = tokenize(query)
        doc = self._docs[idx]
        tf = Counter(doc)
        dl = len(doc)
        s = 0.0
        for term in set(q):
            f = tf.get(term, 0)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1.0))
            s += self._idf(term) * f * (self.k1 + 1) / denom
        # 标签命中加权：问题里出现标签词时，直接给该示例加分。
        # 这是在 BM25 之上的一层确定性规则，实测对「排名」「同比」类问题效果明显。
        for tag in self._examples[idx].tags:
            if tag and tag in query:
                s += 1.5
        return s

    def search(self, query: str, k: int = 3, min_score: float = 0.0) -> list[tuple[Example, float]]:
        scored = [(self._examples[i], self.score(query, i)) for i in range(self._n)]
        scored.sort(key=lambda x: (-x[1], x[0].question))
        return [(e, s) for e, s in scored[:k] if s > min_score]
