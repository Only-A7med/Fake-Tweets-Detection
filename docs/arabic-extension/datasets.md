# Arabic datasets for the bilingual extension — survey and recommendation

This document surveys the Arabic datasets that could support a fake-tweet detector alongside the
existing English pipeline, and recommends one. Embeddings and pretrained transformers are surveyed
separately in [models.md](models.md).

**Status: decision gate.** No pipeline code and no dataset download belongs to this document. The
primary dataset is chosen by the maintainer on the strength of what is recorded here.

**Everything below was read from a primary source on 21 November 2026.** Each entry links the page
that was read. Where a file-level claim is made — "the text is distributed", "the labels are these
values", "there are this many rows" — the file itself was fetched and parsed, and the measured
figure is given next to the figure the authors publish. Anything that could not be reached is
marked **unverified** and says why.

Two reasons this matters more than usual for Arabic:

- **Tweet hydration is no longer possible on this project's budget.** A dataset that ships tweet
  identifiers without text requires the paid X application programming interface to turn the
  identifiers back into text. A dataset in that shape is unusable here unless the same text is
  available elsewhere under a clear licence.
- **Licences are frequently absent rather than restrictive.** Several of the best-known Arabic
  resources carry no licence file and no licence statement at all. That is not the same as
  permissive, and it is recorded as unconfirmed rather than assumed.

---

## 1. Summary table

Ordered by usefulness for this project. "Text?" answers the hydration question.

| # | Dataset | Content | Veracity labels? | Text? | Size (measured) | Licence | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | [ArCOV19-Rumors](#2-arcov19-rumors) | tweets | yes | **yes** | 3,584 labelled tweets | none stated | **recommended primary** |
| 2 | [AFND](#3-afnd-arabic-fake-news-dataset) | news articles | source-level | yes | 606,912 articles | CC BY 4.0 | **recommended backup** |
| 3 | [ANS](#4-ans-arabic-news-stance-corpus) | news claims | yes (synthetic) | yes | 4,547 claims | none on origin | possible backup |
| 4 | [AraStance](#5-arastance) | claims + articles | stance, not veracity | yes | see entry | CC BY 4.0 (mirror) | not suitable |
| 5 | [Arabic satirical fake news](#6-arabic-satirical-fake-news) | news articles | satire vs real | yes | 3,185 + 3,710 | none on repository | not suitable |
| 6 | [Matsda2sh scrape](#7-matsda2sh-fact-check-scrape) | fact-check titles | yes | yes | 3,004 + 3,124 titles | Apache-2.0 | not suitable |
| 7 | [AraFacts](#8-arafacts) | claims | yes | yes | 6,121 claims | **CC BY-NC 4.0** | blocked by licence |
| 8 | [AraNews](#9-aranews) | news articles | manipulated vs real | on request | 5,187,957 articles | none stated | blocked by access |
| 9 | [CheckThat! Arabic](#10-clef-checkthat-arabic-subtasks) | tweets | **no** | unverified | 4,121–6,214 per subtask | CC BY 4.0 (papers) | wrong label set |
| 10 | [Alqurashi COVID-19](#11-alqurashi-covid-19-misinformation) | tweets | yes | **no — identifiers only** | 8,786 identifiers | none stated | unusable |
| 11 | [COVID-19-FAKES](#12-covid-19-fakes) | tweets | machine-generated | **no — identifiers only** | 216,208 identifiers | none stated | unusable |

Only one candidate is Arabic tweets, labelled for veracity by people, and distributed with the text
intact: **ArCOV19-Rumors**.

---

## 2. ArCOV19-Rumors

| | |
|---|---|
| Source | <https://gitlab.com/bigirqu/ArCOV-19> — subdirectory [`ArCOV19-Rumors`](https://gitlab.com/bigirqu/ArCOV-19/-/tree/master/ArCOV19-Rumors) |
| Paper | F. Haouari, M. Hasanain, R. Suwaileh and T. Elsayed, "ArCOV19-Rumors: Arabic COVID-19 Twitter Dataset for Misinformation Detection," in *Proceedings of the Sixth Arabic Natural Language Processing Workshop*, Kyiv, Ukraine (Virtual), 2021, pp. 72–81. Anthology identifier [2021.wanlp-1.8](https://aclanthology.org/2021.wanlp-1.8/). No digital object identifier is listed. |
| Team | bigIR, Qatar University |
| Content | **Tweets**, plus the free-text claims the tweets relate to |
| Licence | **None stated.** See the licence note below. |
| Period | 27 January to the end of April 2020 |
| Topic | COVID-19, and claims in other categories influenced by it |

### What is in it, as measured

The dataset has two subsets. Both ship the tweet text.

**Tweet-verification subset** — one tab-separated file,
[`ArCOV19-Rumors/tweet_verification/Tweets.txt`](https://gitlab.com/bigirqu/ArCOV-19/-/raw/master/ArCOV19-Rumors/tweet_verification/Tweets.txt).
Parsed with Python's `csv` module, tab delimiter, double-quote quoting:

| Measured | Value |
|---|---|
| Columns | `tweetID`, `label`, `tweetText` |
| Rows | 3,584 |
| Label `True` | 1,831 |
| Label `False` | 1,753 |
| Rows with non-empty `tweetText` | 3,584 (all of them) |
| Unique `tweetID` | 3,578 (6 identifiers repeat) |
| Unique `tweetText`, verbatim | 3,464 |
| Unique `tweetText` after removing uniform resource locators and collapsing whitespace | 3,180 |
| Largest cluster of identical normalised texts | 46 rows |
| `tweetText` length in characters | minimum 22, median 173, maximum 330 |

The row and label counts match the figures in the subset's own
[README](https://gitlab.com/bigirqu/ArCOV-19/-/blob/master/ArCOV19-Rumors/README.md) exactly
(3,584 total; 1,753 false at 48.9 per cent; 1,831 true at 51.1 per cent).

**Claim-verification subset** — 138 verified claims in
[`claim_verification/claims`](https://gitlab.com/bigirqu/ArCOV-19/-/raw/master/ArCOV19-Rumors/claim_verification/claims),
with one file per claim under `claim_verification/relevant_tweets/claimNNN.txt`. Measured from the
claims file:

| Measured | Value |
|---|---|
| Columns | `ClaimID`, `Claim`, `ClaimLabel`, `Category`, `ClaimSource` |
| Claims | 138 |
| `ClaimLabel` | `FALSE` 95, `TRUE` 43 |
| `Category` | Health 45, Social 38, Politics 22, Religious 18, Entertainment 9, Sports 6 |

Each `claimNNN.txt` carries `tweetID`, `label`, `tweetText`, with labels `True`, `False` and
`Other`. The subset README states 9,414 tweets in total, of which 5,830 are `Other`.

### The text really is distributed — and the README understates it

The tweet-verification README says the file "stores tweet IDs and veracity labels (either TRUE or
FALSE)". **The file also has a `tweetText` column, populated on every one of the 3,584 rows.** The
same is true of the per-claim files. This was checked by fetching and parsing the files, not by
reading the description, and it is the single most consequential finding in this survey: it means
no hydration, and therefore no paid application programming interface, is required.

### A claim identifier for grouping exists, but has to be reconstructed

`Tweets.txt` has no claim column. The claim for each tweet lives in the per-claim files. Of the 138
per-claim files, **47 were retrieved** before gitlab.com began returning HTTP 429 rate-limit
responses; the rest could not be fetched in this session. From those 47:

| Measured | Value |
|---|---|
| Tweet rows in the 47 files | 2,529, every one with non-empty text |
| Labels | `Other` 1,422, `False` 954, `True` 153 |
| Unique tweet identifiers | 2,523 |
| Identifiers appearing under more than one claim | 2 |
| `True`/`False` rows in those files | 1,107 |
| …of which also present in `Tweets.txt` | **1,107 — all of them** |
| Share of the 3,578 tweet-verification identifiers whose claim is known from 47 of 138 files | 30.9 per cent |

Two conclusions follow. First, the tweet-verification subset appears to be exactly the `True` and
`False` rows of the per-claim files, so **a claim identifier is recoverable for every labelled
tweet** by joining the 138 per-claim files. Second, claims are very nearly a partition: only 2 of
2,523 identifiers sat under more than one claim. That makes claim a sound grouping key.

Group sizes are skewed. Across the 47 sampled claims, counting only `True`/`False` rows: minimum 1,
median 13, mean 23.6, maximum 202 tweets per claim. The implied mean over all 138 claims is 26.0.

### The authors' own folds are not claim-grouped — do not reuse them

`ArCOV19-Rumors/baselines_folds/` holds five files, `split1` to `split5`, each a list of tweet
identifiers. `split1` was fetched and parsed: 223 identifiers, all unique, all present in
`Tweets.txt`, 131 `True` and 92 `False` — 6.2 per cent of the labelled tweets, so the five files do
not partition the dataset.

Crossing `split1` against the claim mapping recovered from the 47 sampled files: of the 6 sampled
claims with any member inside `split1`, **all 6 straddle the boundary** — part of the claim inside,
part outside. None was wholly inside or wholly outside. On this evidence the published folds permit
exactly the same-claim leakage the brief warns about, so this project should build its own
claim-grouped split rather than reuse them. (Based on 6 claims from a 47-file sample; the direction
is unambiguous, the exact rate is not.)

### Licence — unconfirmed, and the maintainer must decide

There is **no licence file and no licence statement**:

- The repository's root listing, read through the GitLab tree application programming interface,
  contains `ArCOV19-Rumors`, `code`, `dataset`, `stats` and `README.md`. **No `LICENSE` file.**
- The root [README](https://gitlab.com/bigirqu/ArCOV-19/-/blob/master/README.md) and the
  [ArCOV19-Rumors README](https://gitlab.com/bigirqu/ArCOV-19/-/blob/master/ArCOV19-Rumors/README.md)
  were both read end to end. Neither mentions a licence, terms of use, or redistribution.
- The paper says the dataset supports research on misinformation detection and the ACL Anthology
  page carries the usual CC BY 4.0 notice **for the paper**, which says nothing about the data.
- The project's GitLab licence metadata field could not be read: the application programming
  interface returned HTTP 429 throughout the later part of this session. **Unverified.**

Mirrors exist on the Hugging Face Hub —
[`khalidalt/ArCOV19-Rumors_tweet_verification`](https://huggingface.co/datasets/khalidalt/ArCOV19-Rumors_tweet_verification),
[`asas-ai/ArCOV19-Rumors`](https://huggingface.co/datasets/asas-ai/ArCOV19-Rumors) (tagged
`license:unknown`), [`arbml/ArCOV19_claims`](https://huggingface.co/datasets/arbml/ArCOV19_claims) —
but none declares a licence either, and a third-party mirror cannot grant rights the origin did not.
The GitLab project stays the source of record.

Practical reading: the data is published openly by an academic group for research use, with a paper
to cite, and this project never redistributes it — the download script caches outside the
repository and nothing is committed. That is the same posture already taken for TruthSeeker 2023,
where the Kaggle mirror declares no licence and the Canadian Institute for Cybersecurity terms
govern. But it is an unconfirmed licence, so under section 8 of the brief it is **the maintainer's
call, recorded here rather than assumed.** Asking the corresponding author for an explicit licence
is cheap and would settle it permanently.

---

## 3. AFND (Arabic Fake News Dataset)

| | |
|---|---|
| Data | <https://data.mendeley.com/datasets/67mhx6hhzd/1>, digital object identifier [10.17632/67mhx6hhzd.1](https://doi.org/10.17632/67mhx6hhzd.1), published 21 February 2022 |
| Paper | A. Khalil, M. Jarrah, M. Aldwairi and M. Jaradat, "AFND: Arabic fake news dataset for the detection and classification of articles credibility," *Data in Brief*, vol. 42, 108141, June 2022, doi: [10.1016/j.dib.2022.108141](https://doi.org/10.1016/j.dib.2022.108141). Author list, journal, volume, article number and date read from the digital object identifier's own metadata record. |
| Content | **News articles**, Modern Standard Arabic |
| Size | 606,912 articles from 134 public Arabic news websites across 19 Arab countries |
| Labels | Source-level credibility — credible, not credible, undecided — assigned using the Arabic fact-checking platform Misbar |
| Format | JavaScript Object Notation; `sources.json` plus one `scraped_articles.json` per source directory; each article carries title, text and publication date. Source identities are anonymised as `source_1`, `source_2` and so on. |
| Licence | **CC BY 4.0**, as named on the Mendeley Data page |
| Dialect | Modern Standard Arabic, stated by the authors |
| Grouping field | the source website, which is the natural grouping key |

**Why it is the recommended backup.** It is large, it is clearly licensed, and the licence is
permissive — the only candidate of real size with no licence doubt at all. Two caveats, both
serious:

- **Domain.** News articles, not tweets. The whole point of the English work is tweets. An
  Arabic model trained on newspaper prose is not measuring the same task, and the comparison with
  the English numbers would be looser still than the size difference already makes it.
- **The labels are source-level, not article-level.** Every article inherits its website's
  credibility rating. That is distant supervision: a credible outlet publishes some false stories
  and a non-credible one publishes much that is true. Treating it as article-level ground truth
  overstates what the labels support, and a grouped split by source is then mandatory — otherwise
  the model can learn to recognise the outlet's style and score very highly while learning nothing
  about veracity. A result above roughly 99 per cent on this dataset should be read as outlet
  identification, not fake-news detection.

Of incidental interest: two of its authors, Jarrah and Aldwairi, are the authors of the Arabic
fake-news work the original paper cites in its related-work section.

---

## 4. ANS (Arabic News Stance corpus)

| | |
|---|---|
| Source | <https://github.com/latynt/ans> |
| Paper | J. Khouja, "Stance Prediction and Claim Verification: An Arabic Perspective," in *Proceedings of the Third Workshop on Fact Extraction and VERification (FEVER)*, 2020. Anthology identifier [2020.fever-1.2](https://aclanthology.org/2020.fever-1.2.pdf); preprint [arXiv:2005.10410](https://arxiv.org/pdf/2005.10410). |
| Content | **News titles** in Arabic, with paraphrased and corrupted variants |
| Size | 4,547 true and false claims; 3,786 (claim, evidence) pairs |
| Licence | **None** — the GitHub repository reports no licence through the repositories application programming interface, and holds only `ans.png`, `data/` and `readme.md`. The mirror [`strombergnlp/ans-stance`](https://huggingface.co/datasets/strombergnlp/ans-stance) is tagged `license:apache-2.0`, which is a third party's claim about someone else's data. |
| Verdict | Possible backup only |

The decisive objection is how the false claims were made: real news titles were **paraphrased and
corrupted** to create them. A classifier trained on that learns to spot the corruption procedure,
not misinformation. It is a reasonable auxiliary or pretraining signal and a poor primary target.

---

## 5. AraStance

| | |
|---|---|
| Source | <https://github.com/Tariq60/arastance>; mirror [`strombergnlp/ara-stance`](https://huggingface.co/datasets/strombergnlp/ara-stance) |
| Paper | T. Alhindi *et al.*, "AraStance: A Multi-Country and Multi-Domain Dataset of Arabic Stance Detection for Fact Checking," preprint [arXiv:2104.13559](https://arxiv.org/pdf/2104.13559) |
| Content | Claims from fact-checking websites paired with articles |
| Labels | **Stance** — agree, disagree, discuss, unrelated — not veracity of the text |
| Licence | The origin repository reports **no licence**; the mirror is tagged `license:cc-by-4.0` |
| Verdict | Not suitable |

Stance between a claim and an article is a different task from classifying one tweet as real or
fake. Listed because it recurs in Arabic fake-news surveys and is easy to mistake for a veracity
dataset.

---

## 6. Arabic satirical fake news

| | |
|---|---|
| Source | <https://github.com/sadanyh/Satirical-Fake-News-Dataset> (`fake_news.zip`, 2,800,346 bytes) |
| Paper | H. Saadany, C. Orasan and E. Mohamed, "Fake or Real? A Study of Arabic Satirical Fake News," in *Proceedings of the 3rd International Workshop on Rumours and Deception in Social Media (RDSM)*, 2020, pp. 70–80. Anthology identifier [2020.rdsm-1.7](https://aclanthology.org/2020.rdsm-1.7/). No digital object identifier listed. |
| Content | **News articles.** 3,185 satirical articles from Al-Hudood and Al-Ahram Al-Mexici; 3,710 real articles from BBC Arabic, CNN Arabic and Al Jazeera |
| Licence | The repository reports **no licence**. The paper is CC BY 4.0, which covers the paper. |
| Verdict | Not suitable |

Satire is written to be recognised as satire; misinformation is written to be believed. The label
is "satirical versus mainstream outlet", so a classifier can separate the classes on house style
alone. The paper's reported accuracy up to 98.6 per cent should be read with that in mind.

---

## 7. Matsda2sh fact-check scrape

| | |
|---|---|
| Source | [`HeshamHaroon/Arabic_fake_news_dataset`](https://huggingface.co/datasets/HeshamHaroon/Arabic_fake_news_dataset) |
| Content | Titles scraped from the Egyptian fact-checking platform [متصدقش (Matsda2sh)](https://matsda2sh.com/) |
| Size, measured | the single file `arabic_fake_news_dataset.json` parsed: 3,133 entries, each `link`, `fakes`, `trues`; **3,004 fake titles and 3,124 true titles** in total |
| Licence | **Apache-2.0**, declared in the dataset card front matter |
| Dialect | Egyptian Arabic in places, by the nature of the platform |
| Verdict | Not suitable as primary |

Clearly licensed and genuinely Arabic misinformation, but short titles rather than tweets, no
paper, no documented annotation procedure, and the card itself warns "this dataset needs more
preprocessing". A single contributor's web scrape. Worth revisiting only as extra training signal.

---

## 8. AraFacts

| | |
|---|---|
| Source | [`asas-ai/AraFacts`](https://huggingface.co/datasets/asas-ai/AraFacts) |
| Paper | Z. Sheikh Ali, W. Mansour, T. Elsayed and A. Al-Ali, "AraFacts: The First Large Arabic Dataset of Naturally Occurring Claims," in *Proceedings of the Sixth Arabic Natural Language Processing Workshop*, Kyiv, Ukraine (Virtual), 2021, pp. 231–236. Anthology identifier [2021.wanlp-1.26](https://aclanthology.org/2021.wanlp-1.26/). No digital object identifier listed. |
| Content | **Claims** gathered from five Arabic fact-checking websites, among them Fatabyyano and Misbar, from 2016 onward |
| Size | the paper's abstract says 6,121 claims; the Hugging Face card reports about 4.67 thousand rows in its default split. **The discrepancy is unexplained and is recorded, not reconciled.** |
| Labels | `normalized_label`, with values including `correct`, `False`, `Partly-false` and two Arabic-script values |
| Licence | **`cc-by-nc-4.0`** — Creative Commons Attribution-NonCommercial |
| Verdict | **Blocked by licence** |

Section 8 of the brief says to stop and ask on any non-commercial licence. This is one. It is also
claims rather than tweets, so the licence question is academic unless the maintainer wants it for
an auxiliary task. **Not used without an explicit decision.**

---

## 9. AraNews

| | |
|---|---|
| Source | <https://github.com/UBC-NLP/wanlp2020_arabic_fake_news_detection> |
| Paper | E. M. B. Nagoudi, A. Elmadany, M. Abdul-Mageed, T. Alhindi and H. Cavusoglu, "Machine Generation and Detection of Arabic Manipulated and Fake News," in *Proceedings of the Fifth Arabic Natural Language Processing Workshop*, Barcelona, Spain (Online), 2020, pp. 69–84. Anthology identifier [2020.wanlp-1.7](https://aclanthology.org/2020.wanlp-1.7/); no digital object identifier listed. The venue name, year, pages and author list are from the Anthology page; the BibTeX entry in the repository README calls the venue the "Fourth" workshop, which the Anthology contradicts. Recorded, not reconciled. |
| Content | **News articles.** 5,187,957 articles from 50 newspapers across 15 Arab countries, the United States and the United Kingdom |
| Access | **A Google Forms registration — "Registration Form for Arabic News and Machine Manipulated News Datasets" — not a direct download.** |
| Licence | **None stated.** No licence file in the repository root; the README makes no licence statement. |
| Verdict | **Blocked by access, and the wrong task anyway** |

Three reasons to set it aside. It is news articles. Its "fake" examples are **machine-generated
manipulations**, so a detector trained on them detects a particular text generator. And it is
gated behind a registration form with no stated licence, which means the terms are whatever the
reply says — the brief's rule 7 cannot be satisfied in advance.

---

## 10. CLEF CheckThat! Arabic subtasks

| | |
|---|---|
| Overview paper | P. Nakov, A. Barrón-Cedeño, G. Da San Martino, F. Alam, R. Míguez, T. Caselli, M. Kutlu, W. Zaghouani, C. Li, S. Shaar, H. Mubarak, A. Nikolov and Y. S. Kartal, "Overview of the CLEF-2022 CheckThat! Lab Task 1 on Identifying Relevant Claims in Tweets," in *Working Notes of CLEF 2022*, CEUR Workshop Proceedings vol. 3180, paper 28. Read in full from <https://ceur-ws.org/Vol-3180/paper-28.pdf>. The paper carries "Use permitted under Creative Commons License Attribution 4.0 International (CC BY 4.0)". |
| Data | <https://gitlab.com/checkthat_lab/clef2022-checkthat-lab/>, as named in the overview paper |
| Content | **Tweets**, on COVID-19 and politics, Arabic among six languages |

The 2022 subtask definitions, quoted from the overview:

- **1A Check-worthiness of tweets** — "Given a tweet, predict whether it is worth fact-checking."
- **1B Verifiable factual claims detection** — "Given a tweet, predict whether it contains a
  verifiable claim or not."
- **1C Harmful tweet detection** — "Given a tweet, predict whether it is harmful to society."
- **1D Attention-worthy tweet detection** — "Given a tweet, predict whether it should get the
  attention of policymakers and why."

**None of these is veracity.** A tweet can be check-worthy and true, or harmful and true. There is
no real/fake label to map, which rules the Arabic Task 1 data out as a primary dataset however
convenient its size and curation. Arabic totals from Table 2 of the overview — 4,121 tweets for
1A, 6,214 for 1B, 6,155 for 1C, 6,140 for 1D — are recorded for completeness.

CheckThat! 2022 **Task 3** did target veracity of news, and 2023 Task 1 included about three
thousand Arabic tweets for check-worthiness; neither gives Arabic tweet-level veracity labels.

**Unverified:** whether the GitLab distribution ships tweet text or identifiers only could not be
checked. gitlab.com returned HTTP 429 rate-limit responses for every request to the CheckThat!
project during this session, including a `Retry-After: 1686`. The overview paper implies text is
released with the data, but the files were not inspected, so this stays unverified. It does not
change the verdict, which rests on the label set.

---

## 11. Alqurashi COVID-19 misinformation

| | |
|---|---|
| Source | <https://github.com/SarahAlqurashi/COVID19-Misinformation-dataset-> |
| Paper | "Eating Garlic Prevents COVID-19 Infection: Detecting Misinformation on the Arabic Content of Twitter," preprint [arXiv:2101.05626](https://arxiv.org/pdf/2101.05626) |
| Content | **Tweets**, March to April 2020, human-annotated into misinformation and not misleading |
| Licence | **None** — the repository reports no licence |
| Verdict | **Unusable: identifiers only** |

The repository's whole data directory is one file,
`Dataset/COVID19-Misinformation-Dataset.txt`, 193,301 bytes. It was fetched and parsed:

| Measured | Value |
|---|---|
| Columns | `id`, `label` — **and nothing else** |
| Rows | 8,786 |
| Label `0` | 7,475 |
| Label `1` | 1,311 |
| Any text column | **no** |

The row count matches the paper's 8,786. The repository also ships a hydration notebook, which is
exactly the problem: turning 8,786 identifiers back into text needs the paid X application
programming interface. **Unusable here.** Which is a real loss — human-labelled Arabic
misinformation tweets, and more than twice ArCOV19-Rumors' labelled count.

---

## 12. COVID-19-FAKES

| | |
|---|---|
| Source | <https://github.com/mohaddad/COVID-FAKES>, branch `COVID-FAKES-A` for Arabic |
| Paper | M. K. Elhadad, K. F. Li and F. Gebali, "COVID-19-FAKES: A Twitter (Arabic/English) Dataset for Detecting Misleading Information on COVID-19," in *Proceedings of the 12th International Conference on Intelligent Networking and Collaborative Systems (INCoS-2020)*, Springer, 2020 |
| Content | **Tweets**, 4 February to 10 March 2020 |
| Licence | **None** — the repository reports no licence |
| Verdict | **Unusable: identifiers only, and the labels are machine-generated** |

All 23 Arabic files were downloaded and parsed together:

| Measured | Value |
|---|---|
| Files on `COVID-FAKES-A` | 23, about 44 megabytes |
| Rows | 216,208 |
| Fields per row | 93, semicolon-separated, with no header |
| First field | a tweet identifier |
| Remaining 92 fields | binary values |
| Any text column | **no** |

Two disqualifications rather than one. There is no tweet text, so hydration would be needed. And
the 92 label columns are the output of the authors' 13 machine-learning algorithms crossed with 7
feature-extraction techniques, scored against official World Health Organization, UNICEF and United
Nations sources — **not human veracity judgements**. Training on another system's predictions
would measure agreement with that system.

---

## 13. Recommendation

### 13.1 Primary dataset: ArCOV19-Rumors, tweet-verification subset

The only candidate that is simultaneously Arabic tweets, labelled for veracity by human
annotators, and distributed with the text. The claim-level subset supplies the grouping key that a
leakage-safe split needs. It comes from an established academic group with a peer-reviewed paper to
cite.

**Risks, in order of seriousness.**

1. **The licence is unconfirmed** (section 2). Needs the maintainer's decision; worth an email to
   the corresponding author.
2. **It is small.** 3,584 labelled tweets against the English pipeline's 200,000 augmented rows —
   a factor of roughly 56. Every Arabic number will carry a wide confidence interval, which is why
   pull request 9 already mandates at least three seeds with mean and standard deviation. Arabic
   and English results must never be presented as directly comparable.
3. **Narrow domain and period.** COVID-19 and claims it influenced, between January and April 2020.
   Nothing can be said about Arabic tweets on other topics or in other years.
4. **Near-duplicates are real and measured.** 3,584 rows collapse to 3,180 distinct texts once
   uniform resource locators are stripped, with one cluster of 46. Removal before splitting is
   mandatory, not optional.
5. **Only 138 groups, with skewed sizes** (median 13, maximum 202 tweets per claim). A single
   80/20 grouped split will be lumpy. See the split design below.
6. **Reconstructing the claim mapping needs all 138 per-claim files**, and gitlab.com rate-limits
   bulk fetching. The download script must fetch politely and cache; a repository archive in one
   request is better than 138 requests.

### 13.2 Backup dataset: AFND

If the maintainer rules the unconfirmed ArCOV19-Rumors licence unacceptable, AFND is the fallback:
CC BY 4.0, no licence doubt, very large. The price is a change of task — news articles with
source-level labels instead of tweets with tweet-level labels — and it must then be framed as a
different experiment rather than the Arabic counterpart of the English one. A grouped split by
source website would be compulsory, and any very high accuracy treated as suspected outlet
identification.

ANS is a distant third and only as auxiliary signal; its false claims are synthetic.

### 13.3 Label mapping

ArCOV19-Rumors uses `True` and `False` in the `label` column. Map to the English convention in
`src/config.py`, where `BinaryNumTarget` is 1 for real:

| Source label | Project label | Integer |
|---|---|---|
| `True` | real | 1 |
| `False` | fake | 0 |
| `Other` | **dropped** | — |

`Other` appears only in the claim-verification subset and means the tweet is related to the claim
without asserting or denying it. It is not a third veracity class and must not be folded into
either side. Dropping it is what reduces the 9,414 claim-relevant tweets to the 3,584 labelled
ones, and it is exactly how the authors define the tweet-verification subset.

Case matters: the tweet files use `True`/`False`, the claims file uses `TRUE`/`FALSE`. Parse
case-insensitively and reject anything unrecognised loudly rather than defaulting it.

### 13.4 Leakage-safe split

There is no paper protocol for Arabic, so leakage-safe from the start, as rule 4 of the brief
requires. The design, in order:

1. **Join the claim identifier onto every labelled tweet** from the 138 per-claim files. Fail
   loudly if any labelled tweet cannot be assigned a claim — do not silently drop it, and do not
   invent a singleton group for it, because either choice quietly changes the split.
2. **Remove exact duplicates by tweet identifier** — 6 identifiers repeat.
3. **Remove near-duplicates** before splitting: strip uniform resource locators, collapse
   whitespace, normalise per the agreed Arabic preprocessing, then drop repeats, keeping the first
   occurrence by tweet identifier so the choice is deterministic. Measured effect: 3,584 rows down
   to approximately 3,180. Report the exact number removed in the exploratory data analysis
   notebook.
4. **Split by group, not by row.** Whole claims go to one side. `StratifiedGroupKFold` from
   scikit-learn, with `groups` = claim identifier and `y` = the binary label, stratifies the label
   while keeping groups intact.
5. **Fixed seed 42**, as everywhere else in the repository.
6. **Augmentation, if any, on the training side only** — and note that the English word-swap
   augmentation is a paper-fidelity device, not an improvement; for Arabic it is optional and
   should be justified by a measurement rather than copied.
7. **A validation split carved from the training side only**, for the transformer's early
   stopping (pull request 8). Also grouped, so no claim spans training and validation.

**Why five folds rather than one split.** With 138 groups and a maximum group of 202 tweets, a
single 20 per cent test set is one roll of the dice: one large claim landing in test can move
accuracy by whole percentage points. Recommended: **report five-fold grouped cross-validation as
the headline, with mean and standard deviation across folds**, and additionally hold one fixed
grouped split for the published demonstration weights. This sits naturally inside what pull
request 9 already requires and costs little on a dataset this size. It also replaces the authors'
`baselines_folds`, which section 2 shows are not claim-grouped.

### 13.5 Expected size and imbalance

From the measured figures, before any augmentation:

| | Value |
|---|---|
| Labelled tweets as distributed | 3,584 |
| After duplicate and near-duplicate removal | approximately 3,180 |
| Class balance as distributed | 1,831 real / 1,753 fake — 51.1 / 48.9 per cent |
| Groups (claims) | 138 |
| Per fold of five, approximately | 636 test tweets, 2,544 training |
| Single 80/20 grouped split, approximately | 2,544 training, 636 test |

The classes are **near-balanced**, so accuracy is not misleading here in the way it would be on a
skewed set. Macro F1 should still be reported, per pull request 9, because fold-level balance will
drift once whole claims move together. The exact post-cleaning counts and the realised fold balance
are to be measured and committed in pull request 5, not predicted here.

### 13.6 Primary embedding and transformer

Full reasoning and the licence evidence are in [models.md](models.md). In short:

- **Bi-LSTM embedding:** AraVec 3.0 Twitter unigram CBOW at 100 dimensions
  (`full_uni_cbow_100_twitter`) — the closest Arabic analogue of the paper's GloVe 100-dimensional
  vectors, trained on 66.9 million tweets. **Its licence is unconfirmed**, so the fallback is
  fastText `cc.ar.300`, which is explicitly CC BY-SA 3.0 but 300-dimensional and not Twitter-only.
- **Transformer:** **CAMeLBERT** `CAMeL-Lab/bert-base-arabic-camelbert-mix` (dialect-and-Modern-
  Standard mix) or `-da` (dialectal). It is the **only** candidate with an unambiguous licence —
  `apache-2.0` on the model card, MIT on the code repository. MARBERTv2 would almost certainly fit
  the task better, being pretrained on a billion Arabic tweets, but its repository states the
  checkpoints are for download and use "exclusively for research", which is a restriction the
  maintainer must accept explicitly before it can be used.

### 13.7 Where the Arabic weights go

**`arabic/` inside the existing public model repository
[`Only-A7med/fake-tweets-detection`](https://huggingface.co/Only-A7med/fake-tweets-detection)**,
beside the existing `paper/` and `safe/` folders. The repository's model card already reserves it:
"`arabic/` is reserved for the bilingual extension and is not present yet."

One repository is right because the English artifacts are MIT-licensed and so is the repository;
the Arabic weights are our own trained parameters, and no dataset or embedding is redistributed by
publishing them. A separate repository would only be needed if a chosen embedding or base model
carried terms that propagate to fine-tuned weights. Of the candidates, **CAMeLBERT (Apache-2.0)
does not create that problem; MARBERT's research-only wording might**, and that is one more reason
the transformer choice is a licence decision before it is a quality decision. If MARBERT is chosen,
the Arabic weights should go to their own repository carrying the upstream restriction in its model
card, and the English repository stays MIT and untouched.

Suggested layout, mirroring the English convention:

```
arabic/
  baseline.joblib          TF-IDF + Logistic Regression
  bilstm.keras             Bi-LSTM + AraVec
  bilstm_history.json      per-epoch loss and accuracy
  transformer/             fine-tuned transformer, in whatever form it saves
```

The Docker image stays weight-free; the start-up download gains the Arabic paths.

### 13.8 Proposed code layout

Following section 4 of the brief. English modules do not move, and no English file is edited except
`src/config.py`, which gains a new section at the end, and the application, which gains routing in
pull request 10.

```
src/
  config.py                    + an "Arabic" section at the end; English constants untouched
  language.py                  NEW  shared language detection (Arabic-script share, threshold)
  data/
    download.py                unchanged (English)
    preprocess.py              unchanged (English)
    split.py                   unchanged (English)
    arabic/
      __init__.py
      normalise.py             NEW  pull request 4: alef/alef-maqsura/taa-marbuta, tashkeel,
                                    tatweel, repeated letters, hashtags, mentions, uniform
                                    resource locators, emojis, Arabic-Indic digits, mixed script
      download.py              NEW  pull request 5: fetch ArCOV19-Rumors, cache outside the repository
      prepare.py               NEW  pull request 5: join the claim identifier, clean, de-duplicate
      split.py                 NEW  pull request 5: grouped, stratified, seeded; five folds
  models/
    baseline.py                unchanged (English)
    bilstm.py                  unchanged (English)
    predict.py                 unchanged (English)
    arabic/
      __init__.py
      baseline.py              NEW  pull request 6: TF-IDF + Logistic Regression
      bilstm.py                NEW  pull request 7: Bi-LSTM + AraVec
      transformer.py           NEW  pull request 8: fine-tuned transformer
      predict.py               NEW  Arabic inference, mirroring models/predict.py
  training/
    train_baseline.py          unchanged (English)
    train_bilstm.py            unchanged (English)
    evaluate.py                extended in pull request 9, per language; English output identical
    arabic/
      __init__.py
      train_baseline.py        NEW  pull request 6
      train_bilstm.py          NEW  pull request 7
      train_transformer.py     NEW  pull request 8
  app/
    app.py                     extended in pull requests 10 and 11; existing fields preserved
    download_artifacts.py      extended in pull request 12 to fetch arabic/

docs/arabic-extension/         this survey
notebooks/arabic_eda.ipynb     NEW  pull request 5
reports/
  metrics/arabic.json          NEW  pull request 6 onward
  arabic_error_analysis.md     NEW  pull request 9
  arabic_extension.md          NEW  pull request 13
scripts/check_arabic_coverage.py   NEW  pull request 4; see below
tests/
  arabic/                      NEW  tests mirroring each module above
```

Two deliberate choices worth stating. **`src/language.py` sits at the top level, not inside
`arabic/`**, because routing is a bilingual concern and putting it under `arabic/` would imply
Arabic owns it. **Arabic modules duplicate rather than generalise the English ones.** Factoring
`train_baseline.py` into a shared generic trainer would be tidier and would touch frozen English
code paths; rule 3 makes that a refactor needing proof of output-identity from the regression
guard. Duplication is the cheaper, safer trade here, and it keeps the English pipeline literally
unchanged. If the duplication becomes painful later, that is its own pull request with the guard
as the gate.

The new Arabic entries in `src/config.py` — vocabulary size, sequence length, learning rate,
batch size, epochs, early-stopping patience, the Arabic-script threshold for routing, which
normalisation rules default to on — are all **our choices, not the paper's**, and each gets the
same `# not specified in the paper` style marker the English section already uses, or a comment
naming the source where one exists.

### 13.9 How the per-module 90 per cent coverage check will work

Section 6 of the brief: the total floor stays 87 per cent, and **every new Arabic module reaches at
least 90 per cent line coverage on its own.** `pytest-cov`'s `--cov-fail-under` is a single total
and cannot express that, so pull request 4 adds a second, independent gate.

Mechanism:

1. The existing continuous-integration pytest step gains one report format and nothing else:
   `--cov-report=json:coverage.json` alongside the current `--cov-report=term-missing`. The
   `--cov-fail-under=87` total gate is untouched, so the English gate cannot change behaviour.
2. A new step runs `scripts/check_arabic_coverage.py`, which reads `coverage.json`, selects every
   file matching the guarded patterns, and fails with a per-file table if any is below 90 per cent.
3. The guarded patterns are globs, not a hand-maintained list — `src/*/arabic/*.py` and
   `src/language.py` — so a new Arabic module is covered by the gate the moment it is added, and
   nobody has to remember to register it.
4. Files with zero statements, such as `__init__.py`, are skipped rather than counted as failures.
5. The script is also runnable locally after any `pytest --cov=src --cov-report=json:coverage.json`,
   so the gate is reproducible off continuous integration.
6. `coverage.json` is added to `.gitignore`; it is a build artifact.

Why a separate script and step rather than a test inside the suite: a test that inspects the
coverage of the run it is part of has to reason about its own measurement, which is fragile.
A post-processing step on the finished report is deterministic and its failure message can name
every offending file and percentage at once.

---

## 14. Limitations

Stated plainly, as rule 5 requires.

**Size, and what it does to the comparison.** The recommended Arabic dataset has 3,584 labelled
tweets, roughly 3,180 after de-duplication. The English pipeline trains on 200,000 augmented rows
from 134,198 cleaned tweets. Arabic results will have materially wider uncertainty and **must not
be placed in the same table as the English numbers without a column making the difference
explicit.** A one or two point difference between two Arabic models on a 636-tweet fold is likely
to be noise; hence at least three seeds, five-fold grouped cross-validation, mean and standard
deviation, and the paired significance test pull request 9 already requires.

**Domain and period.** ArCOV19-Rumors is COVID-19-era claims from January to April 2020. The
English TruthSeeker data is also narrow, but differently narrow. Neither model's accuracy
transfers to general Arabic or English tweets, and the Arabic one is additionally pinned to one
pandemic's information environment.

**Dialect coverage is not documented, and we should not guess it.** The dataset's topical
categories are published; its dialect composition is not. Arabic tweets about a global news event
mix Modern Standard Arabic with Gulf, Egyptian, Levantine and Maghrebi varieties in unknown
proportion. Pull request 5's exploratory data analysis should measure dialect signals rather than
assume them, and pull request 9's error analysis groups errors by dialect against Modern Standard
Arabic — those will be the first real evidence. Nothing in this survey supports a claim about
dialect balance.

**News-article datasets are a different task.** AFND, AraNews and the satire corpus are all
articles. If the maintainer picks one, the extension stops being "the same task in a second
language" and becomes a second experiment. That is defensible but it must be said out loud in the
README and the final report.

**Identifier-only datasets are a genuine loss, not a technicality.** Two of the three Arabic
tweet datasets with human veracity labels — Alqurashi's 8,786 and COVID-19-FAKES' 216,208 — ship
identifiers without text. Hydration needs the paid X application programming interface. Together
they hold more labelled Arabic misinformation tweets than the dataset we can actually use, and
they are out of reach. This is the central reason Arabic tweet work is so much thinner than
English, and it belongs in the limitations of the final write-up.

**Licences that could not be confirmed.** Recorded individually above; collected here because
rule 7 requires it before anything is downloaded into the pipeline.

| Resource | Licence status |
|---|---|
| ArCOV19-Rumors | **No licence file, no licence statement.** Verified by reading the root tree listing and both READMEs. The GitLab project licence metadata field is **unverified** — HTTP 429. |
| AraVec | **No licence.** GitHub reports none; the README says "free to use" without naming terms. |
| MARBERT, MARBERTv2, ARBERT | **No licence**, plus a repository statement that the checkpoints are for use "exclusively for research". |
| AraBERT, including the Twitter variant | **No licence** anywhere in the repository or the model cards. |
| AraFacts | **CC BY-NC 4.0** — non-commercial, a section 8 stop. |
| AraNews | **No licence**, and access only through a registration form. |
| ANS, AraStance, Arabic satire | **No licence** on the origin repositories; third-party Hugging Face mirrors assert CC BY 4.0 or Apache-2.0, which the origins do not. |
| AFND | **CC BY 4.0** — confirmed on the Mendeley Data page. |
| Matsda2sh scrape | **Apache-2.0** — declared on the dataset card. |
| TruthSeeker 2023 (existing English data) | Canadian Institute for Cybersecurity terms govern; the Kaggle mirror declares none. Already recorded in the README as a limitation. |

**What this document did not verify.** Three things, each marked where it appears: the CheckThat!
Arabic file format, because gitlab.com rate-limited every request; the ArCOV19-Rumors GitLab
project licence metadata field, for the same reason; and 91 of the 138 ArCOV19-Rumors per-claim
files, so the claim mapping and the finding about the published folds rest on a 47-file sample
rather than the whole set. The 47-file evidence is consistent and one-directional, but it is a
sample, and pull request 5 will have the complete join.

**No measurement in this document is a result.** Nothing here was trained or evaluated. The counts
above are properties of files published by other people, parsed to check the authors' own
descriptions. The project's own measured results stay in `reports/`.
