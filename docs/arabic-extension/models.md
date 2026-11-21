# Arabic embeddings and transformers — survey and recommendation

Companion to [datasets.md](datasets.md). Same rules: **every identifier, size, licence and citation
below was read from a primary source on 21 November 2026**, each entry links the page that was
read, and anything unreachable is marked **unverified**.

The recommendation is in [datasets.md section 13.6](datasets.md#136-primary-embedding-and-transformer);
this document is the evidence behind it, plus the Hugging Face Space limits the deployment has to
fit.

**Headline finding.** On technical merit the ranking is clear: MARBERTv2 is pretrained on a billion
Arabic tweets and is the obvious best fit for tweet classification. On licence it is the opposite
way round: **CAMeLBERT is the only candidate with an unambiguous licence, and MARBERT's repository
restricts its checkpoints to research use.** The same tension appears in the embeddings — AraVec is
the right corpus and the right dimension, and has no licence at all.

---

## 1. Word embeddings for the Bi-LSTM

The paper's Bi-LSTM initialises its embedding layer with GloVe 100-dimensional vectors (stated in
its equations 1 and 2, not its prose). The Arabic counterpart should match that shape so the
architecture comparison stays honest.

### 1.1 AraVec 3.0 — first candidate

| | |
|---|---|
| Source | <https://github.com/bakrianoo/aravec> |
| Citation, quoted from the README | "Abu Bakr Soliman, Kareem Eisa, and Samhaa R. El-Beltagy, "AraVec: A set of Arabic Word Embedding Models for use in Arabic NLP", in proceedings of the 3rd International Conference on Arabic Computational Linguistics (ACLing 2017), Dubai, UAE, 2017." |
| Corpora | Twitter tweets, and Arabic Wikipedia articles |
| Total tokens, quoted | "By total tokens of more than `1,169,075,128` tokens." |
| Framework | gensim `Word2Vec`, loaded with `gensim.models.Word2Vec.load` |
| Licence | **None.** The GitHub repositories application programming interface reports no licence and the root has no `LICENSE` file. The repository description says the project "aims to provide the Arabic NLP research community with free to use and powerful word embedding models" — an intention, not terms. |

The sixteen AraVec 3.0 variants, transcribed from the README's own download tables. All are hosted
at `https://bakrianoo.ewr1.vultrobjects.com/aravec/`.

**Unigram models**

| Variant | Documents | Vocabulary | Dimensions | File |
|---|---|---|---|---|
| Twitter CBOW | 66,900,000 | 1,259,756 | 300 | `full_uni_cbow_300_twitter.zip` |
| **Twitter CBOW** | **66,900,000** | **1,259,756** | **100** | **`full_uni_cbow_100_twitter.zip`** |
| Twitter SkipGram | 66,900,000 | 1,259,756 | 300 | `full_uni_sg_300_twitter.zip` |
| Twitter SkipGram | 66,900,000 | 1,259,756 | 100 | `full_uni_sg_100_twitter.zip` |
| Wikipedia CBOW | 1,800,000 | 320,636 | 300 / 100 | `full_uni_cbow_{300,100}_wiki.zip` |
| Wikipedia SkipGram | 1,800,000 | 320,636 | 300 / 100 | `full_uni_sg_{300,100}_wiki.zip` |

**N-gram models** (same corpora; vocabularies include multi-word units joined by underscore)

| Variant | Documents | Vocabulary | Dimensions | File |
|---|---|---|---|---|
| Twitter CBOW | 66,900,000 | 1,476,715 | 300 / 100 | `full_grams_cbow_{300,100}_twitter.zip` |
| Twitter SkipGram | 66,900,000 | 1,476,715 | 300 / 100 | `full_grams_sg_{300,100}_twitter.zip` |
| Wikipedia CBOW | 1,800,000 | 662,109 | 300 / 100 | `full_grams_cbow_{300,100}_wiki.zip` |
| Wikipedia SkipGram | 1,800,000 | 662,109 | 300 / 100 | `full_grams_sg_{300,100}_wiki.zip` |

**Availability and download sizes, measured.** The host was probed with HTTP HEAD requests only —
no file was downloaded:

| File | Status | Compressed size |
|---|---|---|
| `full_uni_cbow_100_twitter.zip` | HTTP 200 | 962,251,835 bytes (about 962 megabytes) |
| `full_uni_sg_100_twitter.zip` | HTTP 200 | 963,874,258 bytes |
| `full_grams_cbow_100_twitter.zip` | HTTP 200 | 1,131,904,811 bytes |
| `full_uni_cbow_300_twitter.zip` | HTTP 200 | 2,833,686,412 bytes |

So the host is alive and the 100-dimensional unigram Twitter model is about 962 megabytes
compressed. There is **no mirror on the Hugging Face Hub** — a Hub model search for "aravec"
returns nothing — so the Vultr object store is a single point of failure for availability as well
as the source of the licence question.

**Recommended variant: `full_uni_cbow_100_twitter`.** Reasons: Twitter rather than Wikipedia,
matching the domain; 100 dimensions, matching the paper's GloVe exactly so the Arabic Bi-LSTM is
the same architecture rather than a bigger one; unigrams rather than n-grams, because the n-gram
vocabulary expects underscore-joined multi-word tokens produced by the authors' own `clean_str`
and n-gram construction, which would add an untested preprocessing dependency between our
normalisation and theirs. CBOW over SkipGram is a weak preference and a fair ablation if the
embedding coverage turns out poor.

**Risks.** The licence is unconfirmed, which is a section 8 stop. A single uncommon host. The
vectors load through gensim, which is **not currently a project dependency** — pull request 7 would
add it, or convert the vectors once to a plain text format and keep gensim out of the runtime
requirements, which is the better option for the deployed image. Pull request 7 must report the
measured share of our Arabic vocabulary found in AraVec, as the English pipeline reports GloVe
coverage (25,365 of 50,000 tokens, paper protocol).

### 1.2 fastText Arabic — fallback with a clear licence

| | |
|---|---|
| Source | <https://fasttext.cc/docs/en/crawl-vectors.html> |
| Licence, quoted | "The word vectors are distributed under the Creative Commons Attribution-Share-Alike License 3.0" |
| Corpora | Common Crawl and Wikipedia |
| Coverage | 157 languages |
| Dimensions | 300 |
| Citation | E. Grave, P. Bojanowski, P. Gupta, A. Joulin and T. Mikolov, "Learning Word Vectors for 157 Languages," in *Proceedings of LREC 2018* |
| Arabic files | `cc.ar.300.bin.gz` and `cc.ar.300.vec.gz`, at `https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/` |

**This is the fallback if the AraVec licence is unacceptable.** It is explicitly licensed, from a
stable host, and its subword model handles Arabic morphology and out-of-vocabulary words better
than word-level vectors. Three costs. It is 300-dimensional, so the Arabic Bi-LSTM would no longer
mirror the paper's 100-dimensional embedding — report that as a deliberate, stated deviation
rather than truncating the vectors. It is Common Crawl and Wikipedia, not Twitter, so it is weaker
on dialect and tweet register. And CC BY-SA 3.0 is a share-alike licence: it governs the vectors
and any adapted copy of them, which is a reason not to redistribute a derived vector file in the
model repository. Trained Bi-LSTM weights are a different artifact, but the cautious route is
simply never to publish a modified copy of the vectors.

A second fallback, if a 100-dimensional Arabic vector set is wanted with a clear licence, is
training word2vec on our own Arabic corpus — honest, reproducible, and far weaker than either
pretrained option on 3,584 tweets. Not recommended except as an ablation.

---

## 2. Pretrained Arabic transformers

All four candidates are BERT-base-shaped: 12 layers, 12 attention heads, hidden size 768,
intermediate size 3,072, maximum position embeddings 512. They differ in vocabulary size,
pretraining corpus, and licence.

Parameter counts below are computed from each model's own `config.json`, fetched from the Hub, using
the standard BERT accounting (embeddings + 12 encoder layers + pooler). Where the Hub publishes an
authoritative tensor count the computed figure is cross-checked against it, and against the figures
the MARBERT repository publishes for itself and for AraBERT. Weight-file sizes are read from the
Hub tree application programming interface.

### 2.1 Comparison

| | MARBERT | MARBERTv2 | AraBERTv0.2-Twitter | CAMeLBERT-mix / -DA |
|---|---|---|---|---|
| Hugging Face identifier | `UBC-NLP/MARBERT` | `UBC-NLP/MARBERTv2` | `aubmindlab/bert-base-arabertv02-twitter` | `CAMeL-Lab/bert-base-arabic-camelbert-mix`, `CAMeL-Lab/bert-base-arabic-camelbert-da` |
| Licence on the model card | **none** | **none** | **none** | **`apache-2.0`** |
| Licence on the code repository | **none** ([UBC-NLP/marbert](https://github.com/UBC-NLP/marbert), no `LICENSE`) | same | **none** ([aub-mind/arabert](https://github.com/aub-mind/arabert), no `LICENSE`, no licence wording in the README) | **MIT** ([CAMeL-Lab/CAMeLBERT](https://github.com/CAMeL-Lab/CAMeLBERT)) |
| Extra restriction | **"exclusively for research"** — see 2.2 | same | — | — |
| Vocabulary | 100,000 | 100,000 | 64,000 | 30,000 |
| Parameters, computed | 162,841,344 | 162,841,344 | 135,193,344 | 109,081,344 |
| Parameters, as published | 163M (authors' own table) | — | 136M (card), 135M (MARBERT's table) | — |
| Hub tensor total | — | — | 135,259,392 (safetensors) | — |
| `pytorch_model.bin` size | 654.2 MB | 654.2 MB | 541.1 MB | 438.9 MB |
| Pretraining data | 1B Arabic tweets sampled from about 6B; 128GB, 15.6B tokens; no next-sentence-prediction objective | MARBERT further pretrained on ARBERT's Modern Standard Arabic data plus AraNews, sequence length 512, 40 epochs; **29B tokens** | AraBERTv0.2 (Wikipedia, OSIAN, 1.5B Arabic Corpus, OSCAR, **Assafir (private)**) continued on about 60M Arabic tweets filtered from 100M | mix: classical + dialectal + Modern Standard, 167GB, 17.3B words. DA: dialectal only, 54GB, 5.8B words |
| Sequence-length caveat | pretrained at length 128 | 512 | **trained at length 64; the card warns longer "might result in degraded performance"** | 512 |
| Paper | M. Abdul-Mageed, A. Elmadany and E. M. B. Nagoudi, "ARBERT & MARBERT: Deep Bidirectional Transformers for Arabic," *ACL-IJCNLP 2021*, pp. 7088–7105, doi: [10.18653/v1/2021.acl-long.551](https://doi.org/10.18653/v1/2021.acl-long.551) | same | [arXiv:2003.00104](https://arxiv.org/abs/2003.00104) | "The Interplay of Variant, Size, and Task Type in Arabic Pre-trained Language Models," [arXiv:2103.06678](https://arxiv.org/abs/2103.06678), EACL 2021 WANLP |

The computed and published parameter counts agree: 162,841,344 against the authors' "163M", and
135,193,344 against the Hub's safetensors total of 135,259,392 — the 66,048 difference is the
masked-language-model head, which a classifier replaces anyway.

### 2.2 The MARBERT restriction, quoted

From [UBC-NLP/marbert's README](https://github.com/UBC-NLP/marbert), section 6, "Download ARBERT
and MARBERT Checkpoints":

> ARBERT and MARBERT Pytorch and Tenserflow checkpoints are available on Huggingface website for
> direct download and use ```exclusively for research```.

And from its section 5, "Ethics":

> Our models are developed using data from the public domain. We provide access to our models to
> accelerate scientific research with no liability on our part. Please use our models and
> benchmark only ethically.

This is a **custom restriction, not a standard licence**, and the brief's section 8 says to stop
and ask on exactly this. Three consequences if MARBERTv2 is chosen:

1. It needs the maintainer's explicit, recorded acceptance.
2. The restriction plausibly follows the fine-tuned weights, since they are a derivative of the
   checkpoint. The Arabic weights would then need their own model repository carrying the
   restriction in its card, rather than going into the MIT-licensed
   `Only-A7med/fake-tweets-detection` beside the English artifacts.
3. A public demonstration Space serving it is arguably use beyond research. That is a judgement
   the maintainer makes, not one to make silently.

AraBERT has no stated licence at all, which is a different problem of the same kind, and its
pretraining data includes two private corpora (`Assafir(private)`, `Twitter(private)`), so the
pretraining is not fully reproducible or auditable.

### 2.3 Recommendation: CAMeLBERT

**Primary: `CAMeL-Lab/bert-base-arabic-camelbert-mix`.** Reasons, in order:

1. **It is the only candidate with an unambiguous licence** — `apache-2.0` declared in the model
   card front matter, MIT on the code repository. Nothing has to be asked, accepted, or
   quarantined, and the fine-tuned Arabic weights can sit in `arabic/` in the existing MIT model
   repository with no licence conflict.
2. **It is the smallest** — 109 million parameters, 438.9 megabytes of weights against MARBERT's
   654.2 — which matters directly for the free Space, and the saving is almost entirely the
   embedding table, not depth, so capacity on the encoder is unchanged.
3. **The mix variant covers dialectal, classical and Modern Standard Arabic** (167 gigabytes,
   17.3 billion words), which is the right coverage for tweets about a global news event that mix
   registers.
4. It supports sequence length 512, so our 64-token tweets are comfortably inside its range.

`CAMeL-Lab/bert-base-arabic-camelbert-da` (dialect-only, 54 gigabytes, 5.8 billion words) is the
same size and licence and is the natural second run: a one-line change, a clearly labelled
ablation, and a real test of whether dialect-only pretraining helps on this data.

**State the trade-off honestly in the final report.** CAMeLBERT was not pretrained predominantly on
tweets; MARBERT was, on a billion of them, and would most likely score higher. Choosing CAMeLBERT
is choosing a clean licence over expected accuracy. If the maintainer prefers the stronger model,
MARBERTv2 under its research-only terms is a defensible choice for an academic follow-up paper —
it simply has to be decided, recorded, and reflected in where the weights are published.

A fifth option worth one line: fine-tuning nothing and using the Bi-LSTM alone. That is already
the plan for pull request 7; the transformer is the *improvement* the extension promises, so
skipping it is not on the table.

---

## 3. Fitting the free Hugging Face Space

Read from the Hub documentation on 21 November 2026:
[spaces-overview](https://huggingface.co/docs/hub/spaces-overview),
[spaces-gpus](https://huggingface.co/docs/hub/spaces-gpus).

| Free tier, quoted | Value |
|---|---|
| Hardware name | CPU Basic |
| Processor | 2 vCPU |
| Memory | 16 GB |
| Disk | 50 GB, not persistent |
| Hourly price | Free |
| Sleep | "if your Space runs on the default `cpu-basic` hardware, it will go to sleep if inactive for more than a set time (currently, 48 hours)" |

The overview page states it as: "Each Spaces environment is limited to 16GB RAM, 2 CPU cores and
50GB of (not persistent) disk space by default."

**Does the model fit? Comfortably, on memory and disk.** CAMeLBERT's weights are 438.9 megabytes
and inference in float32 needs roughly that again for activations and framework overhead — well
inside 16 gigabytes even with the existing English models loaded alongside. Disk is not a concern:
the four English artifacts plus one Arabic transformer are under two gigabytes against a 50-gigabyte
allowance. Even MARBERTv2 at 654.2 megabytes fits.

**Latency on 2 vCPU is the real question, and it is not answered here.** A BERT-base forward pass
on two cores is plausibly in the hundreds of milliseconds for a single short tweet, but that is an
expectation, not a measurement, and this project does not publish expectations as numbers. Pull
request 12 measures it on the Space and reports it. Mitigations to try there, in order of
preference: cap the tokeniser at 64 tokens to match the tweet length already used for English;
load the model once at start-up rather than per request; consider dynamic int8 quantisation, which
would roughly quarter both the memory and the time at some cost in accuracy and would have to be
measured as its own variant rather than swapped in silently.

**A deployment risk that affects pull request 12 and is worth flagging now.** The Spaces overview
states: "Static Spaces are free for everyone. Gradio and Docker Spaces run on compute and require
a paid plan to create: PRO for personal accounts, Team or Enterprise for organizations. Free
personal accounts in good standing can still host up to 2 Gradio Spaces running on ZeroGPU." The
repository's plan is a **Docker** Space. On that wording, creating it needs a PRO account, free
CPU Basic hardware notwithstanding. No Space has been created yet, so nothing is broken today, but
pull request 12 should not assume a free Docker Space is available. The options are a paid PRO
account, a Gradio Space on the free ZeroGPU allowance instead of Docker, or no public demonstration
at all — which is the README's current, accurate position.

---

## 4. Preprocessing: the transformer does not get our normalisation

Section 4 of the brief is explicit, and it matters enough to restate. The Arabic normalisation
written in pull request 4 — unifying alef forms, alef maqsura, taa marbuta, stripping tashkeel and
tatweel, collapsing repeated letters — is for the TF-IDF baseline and the Bi-LSTM. A pretrained
transformer expects the preprocessing it was pretrained with, and aggressive normalisation it never
saw can only lose information its tokeniser was built to use.

Concretely:

- **CAMeLBERT** ships its own tokeniser; use `AutoTokenizer.from_pretrained` and apply no
  normalisation beyond handling uniform resource locators, mentions and whitespace. The CAMeL Tools
  preprocessing the authors used is a documented option to test as a variant, not a default.
- **AraBERT** requires its own `ArabertPreprocessor` — the card says "It is recommended to apply
  our preprocessing function before training/testing on any dataset" and that the preprocessor
  "will keep and space out emojis when used with a 'twitter' model". Skipping it would be using
  the model wrongly. It also adds `arabert` as a dependency.
- **MARBERT** was pretrained on raw tweets with only a three-Arabic-word filter, so it is the most
  tolerant of unnormalised input.

If our normalisation is applied to the transformer at all, it must be as a **measured, clearly
labelled ablation** with both arms reported, never as an unstated default.

---

## 5. Open licence questions for the maintainer

Collected from this document; the dataset equivalents are in
[datasets.md section 14](datasets.md#14-limitations).

| Resource | Status | Decision needed |
|---|---|---|
| AraVec 3.0 | No licence; "free to use" in prose only | Accept, or fall back to fastText CC BY-SA 3.0 at 300 dimensions |
| MARBERT / MARBERTv2 | No licence, plus "exclusively for research" | Only if explicitly accepted, with the Arabic weights then published separately under that restriction |
| AraBERT, Twitter variant | No licence at all; private pretraining corpora | Not recommended |
| CAMeLBERT | `apache-2.0` card, MIT code repository | **No decision needed — this is why it is recommended** |
| fastText `cc.ar.300` | CC BY-SA 3.0, explicit | Usable; do not redistribute a modified copy of the vectors |
