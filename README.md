# LLM Evaluation Harness: Llama 3.1 8B on Bioinformatics

A small, honest evaluation of whether a "say *I don't know* if unsure" system prompt reduces hallucinations in a local Llama 3.1 8B model, using 30 hand-written bioinformatics questions, a scoring rubric, and a held-out test split.

## Problem

> Does a "say 'I don't know' if unsure" system prompt reduce Llama 3.1 8B's hallucination rate on bioinformatics questions without lowering accuracy?

## Data

- **30 hand-written questions** in [`questions.csv`](questions.csv), on bioinformatics fundamentals (FASTA/FASTQ, SAM/BED, BLAST, alignment algorithms, genetic code, Phred scores, short calculations).
- **4 question types:** 8 easy factual, 8 hard factual (specific numbers, flags, formats), 6 reasoning/calculation, 8 **trap** questions (invented tool, paper or function, or a false premise).
- **Reference answers** were written before running the model. Each question has source URLs in the `source` column (NCBI, BLAST+ manual, SAM/BED specs, UCSC, Bioconductor, Cock et al. 2010). Calculation questions were checked by hand.
- **Trap checks:** the invented items (`-quantum_mode` for blastn, the "Quantum-Accelerated Sequence Alignment" paper, `GenomicRanges::autoAlignReads()`, the tool SeqNebulaPro) were searched on the web on 2026-11-23 and no matches were found. Searches were web-only, not direct GitHub, Bioconda or PyPI searches.
- **Split:** 10 **dev** questions (used to sanity-check the prompt) and 20 **held-out test** questions (run once). Traps appear in both splits (3 dev, 5 test).
- **Privacy:** all data is public or hand-written. No student, staff or patient records were used.

## Method

- **Model:** `llama3.1:8b` run locally through [Ollama](https://ollama.com) (quantised).
- **Versions:** Ollama `<run: ollama --version>`, `ollama` Python package `<run: pip show ollama>`, Python `<run: python --version>`. Run metadata is logged in `run_meta.jsonl`.
- **Config A (baseline):** no system prompt.
- **Config B:** system prompt *"You are a careful bioinformatics assistant. If you are not sure, or you cannot verify that what is asked about exists, say "I don't know". Never invent details."* Prompt B was **not tuned**: it was run on dev, looked reasonable, and was kept as written.
- **Settings:** temperature 0, seed 0, one run per question per config.
- **Rubric:**
  - `score` 0 (wrong), 1 (partial or vague), 2 (correct and complete). For trap questions, abstaining or correcting the false premise scores 2. Abstaining on an answerable question scores 0.
  - `halluc` (0/1): the answer asserts something invented or confidently false about a named thing (a made-up tool, option, function, citation or quote, or a wrong fact about a format, flag or spec). Arithmetic or reasoning slips are wrong but not flagged.
  - `abstain` (0/1): the answer's overall stance is "I don't know / can't verify".
- **Metrics:**
  - accuracy = mean score / 2
  - hallucination rate = flagged answers / total answers, with Wilson 95% confidence intervals
  - abstain rate, over-abstention (abstaining on non-trap questions), and trap score (accuracy on trap questions only)
  - an exact sign test on paired A/B hallucination flags
- **Scoring:** manual rubric, **AI-assisted** (scored with Claude, then reviewed by me). Scoring was not blind to the config.
- **Code:** [`run_eval.py`](run_eval.py) runs the model, [`analyze.py`](analyze.py) computes the metrics, chart and error analysis, and [`results_app.py`](results_app.py) is a small FastAPI page that displays the results.

## Results

### Summary table

| Split | Config | n | Accuracy | Hallucinated | Halluc. rate (95% CI) | Abstain rate | Trap score |
|---|---|---|---|---|---|---|---|
| dev | A | 10 | 0.55 | 5/10 | 0.50 (0.24 to 0.76) | 0.00 | 0.50 |
| dev | B | 10 | 0.65 | 3/10 | 0.30 (0.11 to 0.60) | 0.20 | 0.83 |
| **test** | **A** | 20 | **0.60** | **9/20** | **0.45 (0.26 to 0.66)** | 0.10 | 0.40 |
| **test** | **B** | 20 | **0.90** | **0/20** | **0.00 (0.00 to 0.16)** | 0.25 | 1.00 |
| all | A | 30 | 0.58 | 14/30 | 0.47 (0.30 to 0.64) | 0.07 | 0.44 |
| all | B | 30 | 0.82 | 3/30 | 0.10 (0.04 to 0.26) | 0.23 | 0.94 |

### Chart (held-out test split)

![Accuracy and hallucination rate, A vs B, test split](results_chart.png)

*Config B's hallucination bar is empty because it hallucinated on 0 of 20 test questions.*

### Results page (FastAPI)

The same numbers displayed by `results_app.py` (`uvicorn results_app:app --reload`):

![Results table served by the FastAPI page](results_page.png)

### Paired comparison (test split)

Config A hallucinated on 9 test questions where B did not, and B hallucinated on 0 where A did not. The exact sign test gives **p = 0.004**.

### By question type (all 30 questions)

| Type | Config A accuracy | A halluc. | Config B accuracy | B halluc. |
|---|---|---|---|---|
| easy (8) | 0.88 | 3 | 1.00 | 0 |
| hard (8) | 0.75 | 5 | 0.62 | 2 |
| reasoning (6) | 0.17 | 0 | 0.67 | 0 |
| trap (8) | 0.44 | 6 | 0.94 | 1 |

**Reading the results:**
- On the held-out test split, prompt B had higher accuracy and fewer hallucinations, mostly because it declined to invent answers to trap questions.
- On **hard** factual questions B was *less* accurate than A (0.62 vs 0.75), because it sometimes abstained when A happened to know the answer (for example, B said "I don't know" for the blastn default word size, which A answered correctly).
- **Answer length is a confound.** Config B's answers are far shorter (median about 155 characters vs about 890 for A). A's longer answers contained more invented extras such as fabricated documentation quotes and made-up technical details, so part of the hallucination drop may come from B saying less, not from B knowing more.

## Error analysis

26 problem answers (score below 2 or hallucination flagged) from dev and test pooled, each tagged with one category. The full list is in [`error_analysis.csv`](error_analysis.csv).

| Category | Config A | Config B | Total |
|---|---|---|---|
| wrong fact | 5 | 3 | 8 |
| fabricated entity | 4 | 0 | 4 |
| confident wrong reasoning | 3 | 2 | 5 |
| wrong number | 2 | 1 | 3 |
| accepted false premise | 2 | 0 | 2 |
| fabricated citation | 2 | 0 | 2 |
| partial answer | 1 | 0 | 1 |
| over-abstention | 0 | 1 | 1 |
| **Total** | **19** | **7** | **26** |

**Examples:**
- **Fabricated entity (A, Q26):** asked what `GenomicRanges::autoAlignReads()` returns (a function that does not exist), A confidently described it as returning a `GRanges` object with alignment scores. B said "I don't know."
- **Accepted false premise (A, Q29):** asked for BLAST's minimum percent identity, A invented defaults ("30% for protein, 90% for nucleotide"). BLAST has no such cutoff. B said "I don't know" rather than correcting the premise.
- **Wrong fact (A and B, Q12):** both configs misread SAM FLAG 4 (it means *unmapped*). A said "reverse orientation" and B said "paired in the sequencing", so no prompt fixed a plain knowledge gap.
- **Confident wrong reasoning (A and B, Q18):** both produced wrong reverse complements for `AACGTTGC` (correct: `GCAACGTT`) with confident step-by-step working.
- **Over-abstention (B, Q10):** B answered "I don't know" to a question A got right (blastn word size 11). This is the cost of the abstention prompt.
- **Fabricated citation (A, Q10 and Q11):** A gave correct answers but added invented "quotes from the NCBI documentation".

## Limitations

- **Small sample:** only 30 questions, 20 in the held-out test split. A difference of a few answers is noise, and even the significant sign test (p = 0.004) applies only to this question set.
- **Single annotator, AI-assisted, not blind.** There is no inter-rater agreement. Some judgement calls (for example, whether a correct answer with an invented quote counts as a hallucination) could reasonably be scored differently.
- **Hallucination definition matters:** flagging confidently false facts about formats and tools makes long answers riskier than short ones, which favours config B.
- **Answer length confound** (see above): B's shorter answers have fewer chances to hallucinate.
- **One model, one prompt, one run:** a single quantised 8B model, a single prompt B, temperature 0, seed 0. Results may not transfer to other models, prompts or sampling settings. No repeated runs were scored.
- **Topic-specific:** bioinformatics fundamentals only. The trap questions are my own design, and the web checks that they do not exist were not exhaustive.
- **No tuning:** prompt B was not tuned, so these are first-attempt results rather than the best an abstention prompt could do.

## Reproduce

```bash
# 1. Install Ollama, then:
ollama pull llama3.1:8b
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Run dev, score scores_dev.csv by hand, then run test ONCE and score scores_test.csv
python run_eval.py --split dev
python run_eval.py --split test

# 3. Metrics, chart and error analysis
python analyze.py

# 4. Optional results page
uvicorn results_app:app --reload
```

## Files

| File | Purpose |
|---|---|
| `questions.csv` | 30 questions, reference answers, sources, dev/test split |
| `run_eval.py` | Runs configs A and B through Ollama |
| `scores_dev.csv`, `scores_test.csv` | Model answers plus manual scores |
| `analyze.py` | Metrics, sign test, chart, error analysis |
| `results_summary.json`, `results_chart.png`, `error_analysis.csv` | Analysis outputs |
| `results_app.py`, `results_page.png` | FastAPI results page and screenshot |
| `run_meta.jsonl` | Run metadata (model, versions, settings) |
