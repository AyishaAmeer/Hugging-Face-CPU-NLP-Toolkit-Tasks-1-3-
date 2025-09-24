import os
import argparse
import re
from transformers import pipeline

os.environ["CUDA_VISIBLE_DEVICES"] = ""
WORD_RE = re.compile(r"\b[\w'-]+\b")
def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))

def validate_constraints(text: str):
    t = " ".join(text.strip().split())
    notes = []
    if not t.endswith("."):
        notes.append("no period")
    enders = re.findall(r"[.!?]", t)

    if not (len(enders) == 1 and t.endswith(".")):
        notes.append(">1 sentence or punctuation")
    wc = count_words(t)
    if wc < 8:
        notes.append("too short")
    if wc > 24:
        notes.append("too long")

    passed = (len(notes) == 0)

    return passed, "; ".join(notes)


def read_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln != ""]

def build_prompt(user_line: str) -> str:
    return (
        "Rewrite the sentence in simpler English. "
        "Output exactly one sentence, 8 to 24 words, ending with a period.\n\n"
        f"Sentence: {user_line}"
    )
def generate_with_beam(pipe, prompt: str) -> str:
    result = pipe(
        prompt,                
        do_sample=False,        
        num_beams=4,            
        early_stopping=True,    
        no_repeat_ngram_size=2, 
        max_new_tokens=48       
    )
    
    text = result[0]["generated_text"]
    
    return " ".join(text.strip().split())

def main():
    parser = argparse.ArgumentParser(description="Step 1: Read inputs from a .txt file")
    parser.add_argument("--infile", required=True, help="Path to input .txt (each line = one input)")
    args = parser.parse_args()

    inputs = read_lines(args.infile)
    if not inputs:
        print("No non-empty lines found in infile.")
        return
    
    pipe = pipeline(
        "text2text-generation",        
        model="google/flan-t5-base",   
        device=-1                      
    )
    print(f"Loaded {len(inputs)} lines from: {args.infile}")
    print("Decoding mode: BEAM SEARCH (deterministic)\n")
    for idx, line in enumerate(inputs, start=1):
        
        prompt = build_prompt(line)
        
        output = generate_with_beam(pipe, prompt)
        
        tokens_out = count_words(output)
        
        passed, notes = validate_constraints(output)
        
        print(f"{idx:02d}. tokens_out={tokens_out:02d} | passed={passed!s:<5} | notes={notes or '-'}")
        print(f"    input : {line}")
        print(f"    output: {output}\n")

if __name__ == "__main__":
    main()