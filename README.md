Hugging Face CPU NLP Toolkit (Tasks 1–3)

Small, CPU-only NLP scripts using Hugging Face pipelines to:

Task 1: Instruction rewrite (FLAN-T5) + Sentiment (DistilBERT).

Task 2: Batch rewrite with beam vs sampling + CSV export + mini-report.

Task 3: Guardrails + metrics across rewrite, summarize, sentiment (retry on failure, neutral band).

1) Environment (Windows / PowerShell)

# Create & activate Python 3.11 venv (adjust path if needed)
& "C:\Users\ayish\AppData\Local\Programs\Python\Python311\python.exe" -m venv hf311
.\hf311\Scripts\activate

# Install packages
python -m pip install --upgrade pip
pip install torch transformers huggingface_hub

# (optional) for notebooks
pip install ipykernel
python -m ipykernel install --user --name hf311 --display-name "Python 3.11.9 (hf311)"


CPU-only: the scripts set CUDA_VISIBLE_DEVICES="" and use device=-1.

2) Files

.
├── hug_face.ipynb               # Task 1
├── task2.py                  # Task 2
├── task3.py                  # Task 3
├── textfile.txt              # inputs for rewrite (1 sentence per line)
├── lines_summarize.txt       # inputs for summarize (2–3 sentences per line)
├── lines_sentiment.txt       # inputs for sentiment (1 sentence per line)
├── results.csv               # Task 2 output
└── README.md

3) Task 1 

A) Instruction rewrite (FLAN-T5-base)

Input: one English sentence (prompted in the script).

Output constraints: exactly 1 sentence, 8–24 words, ends with a period.

Decoding: deterministic (no sampling).

If constraints fail → prints a clear note.

B) Sentiment (DistilBERT SST-2)

Input: 3 sentences (prompted in the script).

Output: label + score for each.

Prints a short note explaining why SST-2 has no Neutral.

4) Task 2 — task2.py (Batch rewrite + CSV + mini-report)

* Input file: --infile textfile.txt (one sentence per line).

* Runs two decoding modes on each line:

    Beam (deterministic)

    Sampling (temperature/top-p)

* Exports results.csv with columns:
    input, output, decoding, tokens_out, constraint_passed, notes

* Prints a mini-report per mode:

    % rows that passed constraints

    avg tokens_out

    one-sentence reflection

Run:

python task2.py --infile textfile.txt --outfile results.csv

5) Task 3 — task3.py (Guardrails + evaluation)

Supports three tasks via --task:

A) --task rewrite (FLAN-T5)

* Beam first; if constraints fail or output unchanged, retry once with sampling (different temperature/top-p).

* Prints per-line status and a metrics block:

   Beam pass-rate, Sampling pass-rate

   Avg output length (words) for beam & sampling

   Retry triggered & retry fix-rate

Run:

python task3.py --task rewrite --infile textfile.txt

B) --task summarize (DistilBART CNN)

Input lines are short paragraphs (2–3 sentences).

Beam first; if fail, retry with sampling.

Post-processing forces exactly one sentence ending with a period.

Prints metrics like rewrite.

Run:

python task3.py --task summarize --infile lines_summarize.txt

C) --task sentiment (DistilBERT SST-2 + neutral band)

Adds a neutral band around 0.5 to map uncertain scores to NEUTRAL.

Shows raw label/score and neutralized label; prints distributions.

Run (default ±0.10 band):

python task3.py --task sentiment --infile lines_sentiment.txt --neutral_band 0.10
(Widen to 0.20 if you want to see more NEUTRALs.)

6) Sample inputs (short)

textfile.txt (rewrite & summarize):

Given the sudden change in schedule, the meeting originally planned for Friday afternoon has been rescheduled to Monday morning to accommodate key participants.
Although the software update promised improved performance, many users reported slower load times and occasional crashes during routine tasks after installing it.
...
lines_sentinment.txt (sentiment):

I love how quickly this phone launches apps and switches between tasks.
This update is fine; I don’t notice much difference either way.
...

7) Reproducibility & determinism

Seed: scripts call set_seed(42) for repeatable sampling.

Beam decoding: deterministic with do_sample=False.

CPU-only: all pipelines use device=-1.

8) Troubleshooting

“temperature ignored”: expected if do_sample=False (beam/greedy).

return_full_text error: remove it for text2text-generation/summarization pipelines.

Hugging Face cache + Windows symlink warning: harmless; enable Windows “Developer Mode” to silence it or set HF_HUB_DISABLE_SYMLINKS_WARNING=1.

Summarization too long / multiple sentences: lower max_length, raise length_penalty, and post-process to the first sentence (as in task3.py).

9) Requirements
Minimal: 

torch
transformers
huggingface_hub

10) Models used (acknowledgments)

Rewrite: google/flan-t5-base

Summarize: sshleifer/distilbart-cnn-12-6

Sentiment: distilbert-base-uncased-finetuned-sst-2-english

All models are accessed via the Hugging Face transformers pipelines and run locally on CPU.

