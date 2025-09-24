import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

'''model_id= "google/flan-t5-base"
device = torch.device("cpu")
tok= AutoTokenizer.from_pretrained(model_id)
model= AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)

sentences= [
    "Show one-sentence summary of why python is used in education.",
    "Summarize the benefit of running models locally on CPU for teaching.",
    "Explain in one sentence what a tokenizer does.",
]

enc_batch=tok(
    [f"Response in one sentence: {s}" for s in sentences],
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length = 96
).to(device)

print("\n--- Shapes (batch input) ---\n")
print("Input IDs shape:", enc_batch["input_ids"].shape)          # [B, L
print("Attention mask shape:", enc_batch["attention_mask"].shape)  # [B, L]
print("Padding token ID:", tok.pad_token_id)

with torch.no_grad():
    out_batch = model.generate(
        **enc_batch,   #unpacks input_ids and attention_mask
        do_sample=False,        # deterministic
        num_beams=4,           # beam search
        no_repeat_ngram_size=3,
        encoder_no_repeat_ngram_size=4,
        repetition_penalty=1.1,
        max_new_tokens=32
    )

    print("\n--- Shapes (batch output) ---\n")
    for i, out in enumerate(out_batch):
        print(f"{i+1}.", tok.decode(out, skip_special_tokens=True))'''

# Part- Decoding strategies : greedy, beam search, sampling

model_id= "google/flan-t5-base"
device = torch.device("cpu")
tok= AutoTokenizer.from_pretrained(model_id)
model= AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
'''
def run_decode(prompt: str):
    enc= tok(prompt, return_tensors="pt").to(device)

    print("\n--- Shapes(input) ---\n")

    print("Input IDs:", enc["input_ids"].shape)          # [1, L]
    print("Attention mask:", enc["attention_mask"].shape)  # [1, L]

    with torch.no_grad():
        g = model.generate(
            **enc,
            do_sample=False,        # deterministic
            num_beams=1,           # beam search pure greedy
            no_repeat_ngram_size=2,# no repeating phrases
            max_new_tokens=48      # cap on generated tokens
        )

        b = model.generate(
            **enc,
            do_sample=False,        # deterministic
            num_beams=5,           # beam search
            no_repeat_ngram_size=2,# no repeating phrases
            max_new_tokens=48      # cap on generated tokens
        )
        s= model.generate(
            **enc,
            do_sample=True,        # enable sampling
            temperature=1.0,       # default temp
            top_p=0.9,             # nucleus sampling
            top_k=50,              # top-k sampling
            no_repeat_ngram_size=2,# no repeating phrases
            max_new_tokens=48      # cap on generated tokens
        )

        return(
            tok.decode(g[0], skip_special_tokens=True),
            tok.decode(b[0], skip_special_tokens=True),
            tok.decode(s[0], skip_special_tokens=True)
        )
cmp_prompt=(
    "Create One playful single-sentence analogy that explains how text is split"
    "for language models to understand. Do not use the words 'token' or 'tokenizer'."
    "End with a period."
)
greedy_text, beam_text, sample_text= run_decode(cmp_prompt)
print("\n--- Decode Comparison ---\n")
print("Greedy:", greedy_text)
print("Beam:", beam_text)
print("Sample:", sample_text)'''

#Part-9- Timing(simple prompt ve small batch on CPU)

import time
sentences= [
    "Show one-sentence summary of why python is used in education.",
    "Summarize the benefit of running models locally on CPU for teaching.",
    "Explain in one sentence what a tokenizer does.",
]
batch_prompts= [f"Response in one sentence: {s}" for s in sentences]*2
# Time: single imput
t0= time.perf_counter()
_=model.generate(
    **tok("Respond in one sentence: What is a tokenizer?", return_tensors="pt").to(device),
    max_new_tokens=24

)
t1 = time.perf_counter()
# Time: small batch

enc2 = tok(
    batch_prompts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=96
).to(device)
_= model.generate(
    **enc2,
    max_new_tokens=24
)
t2 = time.perf_counter() #t2-t1

print(f"Single input time: {t1 - t0:.3f} sec")
print(f"Small batch time: {t2 - t1:.3f} sec")
