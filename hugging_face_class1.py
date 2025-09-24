#   Vocabulary

# - **Model**: a trained neural network that maps input text to output text or labels.
# - **Weights / checkpoint**: the learned parameters of that model (the big files we download).
# - **Tokenizer**: converts text → **tokens** (subwords/IDs) for the model, and back again.
# - **Tokens**: pieces of text (not necessarily full words). Length limits and speeds are
#   measured in tokens, so “max_new_tokens” is about subwords, not words.
# - **Transformer (architecture)**: the neural network design (self-attention, etc.) used
#   by most state-of-the-art language models.
# - **Transformers (library)**: the Hugging Face Python library we import to use models.
# - **Pipeline**: a prebuilt function (e.g., `pipeline("sentiment-analysis")`) that handles
#   tokenizer + model + decoding for a common task.
# - **Inference** vs **training**:
#   * Inference = using a trained model to make predictions.
#   * Training/fine-tuning = updating weights with data.
# - **Encoder–decoder** vs **decoder-only**:
#   * Encoder–decoder (e.g., T5/BART): good for text-to-text tasks (summarize/translate).
#   * Decoder-only (e.g., GPT-style): good for next-token text continuation.
# - **Deterministic decoding** (beam search, no sampling): stable, repeatable outputs.
# - **Sampling** (temperature, top-p): more creative/varied but less predictable.

# "How a pipeline call works (mental model):"
#   1) Your input text → **tokenizer** → token IDs.
#   2) Token IDs → **model** (forward pass on CPU) → output logits.
#   3) **Decoding** turns logits into text (beam search or sampling).
#   4) Output tokens → detokenize → final string.


# Setup
# text-to-text generation
# Sentiment analysis
# Summarization
# Translation

# Transformers
# accelerate

# Part1: Text Generation with Flan-t5-base
'''from transformers import pipeline

generator = pipeline("text2text-generation", model="google/flan-t5-base")

prompt= ("Produce exactly one family-friendly joke."
         "One sentence, 10-20 words, end with a period.")

print ("\n-- Text Generation with Flan-t5-base ---\n")
print(generator(prompt, max_new_tokens=32, num_beams=5, no_repeat_ngram_size=3, do_sample= False)[0]['generated_text'])

#Part2- Sentiment Analysis with DistilBERT

sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
print ("\n-- Sentiment Analysis with DistilBERT ---\n")
examples= ["I love using Hugging Face models!", "I hate this movie.", "It's an average day."]

for text in examples:
    result = sentiment_analyzer(text)[0]
    print(f"Text: {text}\nLabel: {result['label']}, Score: {result['score']:.4f}\n")'''

# Part3- summarization with BART

from transformers import pipeline

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
print ("\n-- Summarization with BART ---\n")

article = (
    "Python is a high-level, interpreted programming language known for its readability and versatility. "
    "It supports multiple programming paradigms, including procedural, object-oriented, and functional programming. "
    "Python has a large standard library and a vibrant ecosystem of third-party packages, making it suitable for a wide range of applications. "
    "Python is widely used in web development, data science, artificial intelligence, scientific computing, and automation. "
    "Its simple syntax and dynamic typing make it a popular choice for beginners and experienced developers alike."
)

print(summarizer(article, max_length=130, min_length=30, do_sample=False)[0]['summary_text'])

# Part4- Translation En to sv (OPUS-MT)

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-sv")
print ("\n-- Translation En to sv (OPUS-MT) ---\n")

text = "Hugging Face is creating a tool that democratizes AI."
print(translator(text, max_length=40)[0]['translation_text'])
