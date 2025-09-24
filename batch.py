from transformers import AutoTokenizer
import torch

model_name= "google/flan-t5-base"

max_new_tokens = 32

torch.manual_seed(42)

raw_prompts = [
    # Rewrite (4)
    "Rewrite the sentence in simpler English. End with a period. Sentence: 'Python’s clear syntax helps beginners focus on problem-solving.' Output:",
    "Rewrite the sentence in simpler English. End with a period. Sentence: 'Version control lets teams track changes and work safely together.' Output:",
    "Rewrite the sentence in simpler English. End with a period. Sentence: 'Preprocessing text often includes lowercasing and removing extra spaces.' Output:",
    "Rewrite the sentence in simpler English. End with a period. Sentence: 'Short prompts run faster on CPU because attention scales with length.' Output:",
    # Explain (4)
    "Explain in one sentence what a learning rate does. End with a period.",
    "Explain in one sentence what an API key is used for. End with a period.",
    "Explain in one sentence what a unit test checks. End with a period.",
    "Explain in one sentence what a tokenizer does in NLP. End with a period.",
    # Summarize (4)
    "Summarize in one sentence: 'Pipelines bundle tokenization, the model, and decoding. They are great for quick demos on CPU.' Output:",
    "Summarize in one sentence: 'Batching several prompts can improve throughput. Padding and masks keep shapes compatible.' Output:",
    "Summarize in one sentence: 'Beam search is deterministic and often fluent. Sampling adds creativity but may drift.' Output:",
    "Summarize in one sentence: 'SentencePiece and WordPiece split text into subwords. This keeps vocabulary small and improves coverage.' Output:",
]

batch_prompts = [f"Respond in one sentence: {p}" for p in raw_prompts]
tokenizer = AutoTokenizer.from_pretrained(model_name)
enc_batch = tokenizer(
    batch_prompts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=128
)
#input_ids = enc_batch.input_ids           # shape: (batch, seq_len)
#attention_mask = enc_batch.attention_mask # 1 for real tokens, 0 for padding

print("Input IDs shape:", enc_batch["input_ids"].shape)  # [B, L]
print("Attention mask shape:", enc_batch["attention_mask"].shape)  # [B, L]
print("Padding token ID:", tokenizer.pad_token_id)

print("attention_mask last row:", enc_batch["attention_mask"][-1].tolist())