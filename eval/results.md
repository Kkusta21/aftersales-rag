Embedder: `hashing`, reranker: `none`

| Retrieval mode | Questions | Hit@1 | Hit@3 | MRR |
|---|---|---|---|---|
| bm25 | 31 | 0.97 | 1.00 | 0.98 |
| dense | 31 | 0.94 | 0.97 | 0.96 |
| hybrid | 31 | 0.97 | 1.00 | 0.98 |

Abstention accuracy (answer vs. say "not found"): **0.94** on 16 questions
Routing accuracy (manuals / recalls / clarify): **1.00** on 10 questions
