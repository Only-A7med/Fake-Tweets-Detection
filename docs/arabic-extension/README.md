# Arabic extension — documentation

Planning documents for the bilingual (English and Arabic) extension of this project. The English
pipeline is frozen: nothing in this directory changes its code paths, outputs or results.

In these documents, "plan item N" refers to item N of the project plan, not to a GitHub
pull request number.

| Document | Contents |
|---|---|
| [datasets.md](datasets.md) | Survey of eleven candidate Arabic datasets, with the licence, size, label scheme, tweets-against-articles question and — critically — whether the tweet text is distributed or only tweet identifiers. Ends with the recommended dataset, label mapping, leakage-safe split design, code layout and per-module coverage gate. |
| [models.md](models.md) | Arabic word embeddings (AraVec variants, fastText fallback) and pretrained transformers (MARBERT, MARBERTv2, AraBERT Twitter, CAMeLBERT) with exact Hugging Face identifiers, licences, parameter counts and weight sizes, judged against the free Hugging Face Space limits. |

Both were written as a decision gate: they record evidence and recommend. **The gate has since
been decided — the maintainer's choices of 21 November 2026 are recorded in
[datasets.md section 13.10](datasets.md#1310-decisions-21-november-2026), which governs wherever
it differs from the recommendation.** Every identifier, licence, size and citation was read
from a primary source on 21 November 2026, and each is linked. Items that could not be reached are
marked **unverified**, with the reason.

Nothing in this directory reports a measured result of this project. Those live in `reports/`.
