# vocabulary sheet 

# ————— Core concepts —————
# - **Model**: a trained neural network that maps input text to output text/labels.
# - **Weights / checkpoint**: the large files with the model’s learned parameters.
# - **Inference** vs **training**: using a model to predict vs. updating its weights with data.

# ————— Tokenization —————
# - **Tokenizer**: converts text ↔ tokens (integers). Loaded with `AutoTokenizer`.
# - **Token / Token ID**: a subword unit represented as an integer (what the model actually reads).
# - **Vocabulary**: the set of all tokens a tokenizer knows.
# - **WordPiece**: tokenizer family used by BERT/DistilBERT (often uncased; adds special tokens).
# - **SentencePiece**: tokenizer family used by T5/FLAN & Marian/OPUS-MT (language-agnostic subwords).
# - **Detokenize**: convert token IDs back into human-readable text (optionally hiding special tokens).

# ————— Special tokens you’ll see —————
# - **[CLS]**: “classification” token at the start for BERT-style models (pooled summary for classifiers).
# - **[SEP]**: “separator/end” token at the end (and between paired sentences) for BERT-style models.
# - **</s> (EOS)**: end-of-sequence token used by T5/FLAN to signal “stop generating”.
# - **<pad>**: padding token used to equalize sequence lengths inside a batch.
# - **Pad token id**: the integer ID representing `<pad>` (e.g., 0 for many T5/FLAN models).

# ————— Tensors & shapes —————
# - **Shape [B, L]**: tensors print as `torch.Size([batch_size, sequence_length])`.
# - **Batch size (B)**: how many sequences we process at once.
# - **Sequence length (L)**: number of **tokens** after tokenization (not characters/words).
# - **attention_mask**: `[B, L]` tensor with 1=real token, 0=padding; tells the model to ignore pads.
# - **Padding**: add `<pad>` tokens so all sequences in a batch share the same length.
# - **Truncation**: cut inputs longer than a chosen `max_length` (protects CPU time/memory).

# ————— Pipelines & Auto classes —————
# - **Pipeline**: a prebuilt function (e.g., `pipeline("summarization")`) bundling tokenizer+model+decoding.
# - **Auto classes**: factory loaders that pick correct components, e.g.:
#   `AutoTokenizer.from_pretrained("google/flan-t5-base")`,
#   `AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")`.

# ————— Model families we use —————
# - **FLAN-T5 (encoder–decoder / seq2seq)**: instruction-tuned T5 (great for text-to-text tasks).
# - **DistilBERT (encoder-only)**: lightweight BERT for classification (expects [CLS]…[SEP]).
# - **DistilBART-CNN (encoder–decoder)**: distilled BART fine-tuned for news summarization.
# - **OPUS-MT / Marian (encoder–decoder)**: translation models (e.g., EN→SV), SentencePiece-based.
# - **Distillation**: compressing a large “teacher” into a smaller, faster “student” model.

# ————— Generation (decoding) —————
# - **`.generate(...)`**: turns input token IDs into new output token IDs.
# - **Greedy**: always pick the top next token (deterministic, safe).
# - **Beam search (`num_beams`)**: explore several high-probability paths; often more fluent; can repeat.
# - **Sampling (`do_sample=True`)**: add randomness; tune with **temperature** and **top_p** for creativity.
# - **`max_new_tokens`**: cap on how many **generated** tokens to produce.
# - **`no_repeat_ngram_size`**: discourages repeating short phrases (reduces loops/copypasta).
# - **`length_penalty`**: bias beams toward shorter/longer outputs.
# - **`skip_special_tokens=True`**: hide special tokens when decoding to text.

# ————— Performance & timing —————
# - **`time.perf_counter()`**: high-resolution wall-clock timer for quick benchmarks.
# - **Latency vs throughput**: time per call vs items per second (batching improves throughput).
# - **Warm-up / caches**: first call is slower; later calls are faster (weights loaded, kernels warmed).
# - **Scaling with L²**: attention cost grows roughly with the square of sequence length—keep prompts short on CPU.

# part5 : Token & Tokenizer

from transformers import AutoTokenizer

text = "Transformers make local demo easy. Python is great for teaching"

#FLAN-T%-base tokenizer(Sentencepiece)

tok_flan = AutoTokenizer.from_pretrained("google/flan-t5-base")

ids_flan = tok_flan.encode(text)

print("\n-- FLAN-T5-base Tokenizer (SentencePiece) --\n")

print("Token IDs:", ids_flan[:20], "...")  # print first 20 ids

print("Decoded:", tok_flan.decode(ids_flan))

#DistilBERT tokenizer(WordPiece)

tok_bert = AutoTokenizer.from_pretrained("distilbert-base-uncased") #[CLS]=101 [SEP]=102

ids_bert = tok_bert.encode(text)

print("\n-- DistilBERT Tokenizer (WordPiece) --\n")

print("Token IDs:", ids_bert[:20], "...")  # print first 20 ids

print("Decoded:", tok_bert.decode(ids_bert))

# Part6 : From pipeline to Auto classes

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

device = torch.device("cpu")
model_id= "google/flan-t5-base"
tok= AutoTokenizer.from_pretrained(model_id)
model= AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
prompt = (
    "Rewrite the sentence in simpler English. Output exactly one sentence, ending with a period.\n\n"
    "Sentence: 'Transformers lets us try modern AI models locally for teaching.'\n"
    "Output: "
)
# Encode
enc = tok(prompt, return_tensors="pt").to(device)

print("\n --- Shapes(input) ---\n")

print("Input IDs:", enc["input_ids"].shape)          # [1, L]
print("Attention mask:", enc["attention_mask"].shape)  # [1, L]

#generate

with torch.no_grad():
    out = model.generate(
        **enc,
        do_sample=False,        # deterministic
        num_beams=4,           # beam search
        early_stopping=True,   # stop when best beam ends
        no_repeat_ngram_size=2,# no repeating phrases
        max_new_tokens=48      # cap on generated tokens
    )

    print("\n--- Shapes(output) ---\n")
    print("Output IDs:", out.shape)  # [1, L_out]

    out_text= tok.decode(out[0], skip_special_tokens=True)
    print("\n--- Generated text ---\n")
    print(out_text)