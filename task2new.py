import os
import re
import csv
import argparse
from transformers import pipeline, set_seed

os.environ["CUDA_VISIBLE_DEVICES"] = ""
set_seed(42)

def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() != ""]
    
def word_count(text):
    return len(text.split())

def validate(text):
    t = " ".join(text.strip().split())
    notes = []
    if not t.endswith("."):
        notes.append("no period")
    enders = re.findall(r"[.!?]", t)
    if not (len(enders) == 1 and t.endswith(".")):
        notes.append(">1 sentence or punctuation")
    wc = word_count(t)
    if wc < 8:
        notes.append("too short")
    if wc > 24:
        notes.append("too long")
    passed = (len(notes) == 0)
    return passed, "; ".join(notes)

def prompt_for(line):
    return (
        "Rewrite the sentence using simpler English and different wording. "
        "Do not copy the original sentence; rephrase it with simpler words. "
        "Output exactly one sentence, 8 to 24 words, ending with a period.\n\n"
        f"Original: {line}\n"
        "Simplified:"
    )

def main():

    parser = argparse.ArgumentParser(description="Task 2: batch rewrite + beam vs sampling + CSV + reporting")
    parser.add_argument("--infile", required=True, help="Path to input .txt (each line = one input)")
    parser.add_argument("--outfile", required=True, help="Path to output .csv")
    args = parser.parse_args()

    # Read input lines
    inputs = read_lines(args.infile)
    if not inputs:
        print("No input lines found.")
        return
    pipe = pipeline(
        "text2text-generation", 
        model="google/flan-t5-base", 
        device=-1
    )
    rows=[]
    print(f"\nLoaded {len(inputs)} lines from: {args.infile}\n")  
    for i, line in enumerate(inputs, start=1): 
        pr = prompt_for(line)    
        beam_out = pipe(                          
            pr,                                   
            do_sample=False,                      
            num_beams=4,                          
            early_stopping=True,                  
            no_repeat_ngram_size=2,              
            max_new_tokens=48,                    
            #return_full_text=False                
        )[0]["generated_text"] 

        beam_out = " ".join(beam_out.strip().split())  # normalize spaces
        unchanged = (beam_out.strip().lower() == line.strip().lower())
        
        beam_tokens = word_count(beam_out)        # count words
        beam_pass, beam_notes = validate(beam_out)# validate constraints
        if unchanged and not beam_notes:
            beam_notes = "unchanged" 
        
        rows.append({
            "input": line,
            "output": beam_out,
            "decoding": "beam",
            "tokens_out": str(beam_tokens),
            "constraint_passed": "True" if beam_pass else "False",
            "notes": beam_notes
        })

        # --------- SAMPLING  ---------

        sample_out = pipe(                          
            pr,
            do_sample=True,                       
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            max_new_tokens=48,
            #return_full_text= False
        )[0]["generated_text"]

        sample_out = " ".join(sample_out.strip().split())  # normalize spaces
        
        unchanged = (sample_out.strip().lower() == line.strip().lower())
        sample_tokens = word_count(sample_out)        # count words
        sample_pass, sample_notes = validate(sample_out)# validate constraints
        if unchanged and not sample_notes:
            sample_notes = "unchanged"

        rows.append({
            "input": line,
            "output": sample_out,
            "decoding": "sampling",
            "tokens_out": str(sample_tokens),
            "constraint_passed": "True" if sample_pass else "False",
            "notes": sample_notes
        })
        print(f"{i:02d}. beam tokens={beam_tokens:02d} pass={beam_pass} | sampling tokens={sample_tokens:02d} pass={sample_pass}")

    # Write to CSV
    with open(args.outfile, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "input", "output", "decoding", "tokens_out", "constraint_passed", "notes"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Results written to: {args.outfile}")
    valid_rows = [r for r in rows if isinstance(r, dict) and r.get("decoding") in ("beam", "sampling")]
    
    if len(valid_rows) != len(rows):
        print(f"Note: {len(rows) - len(valid_rows)} row(s) skipped because they lacked expected keys.")
    
    # ---------- MINI-REPORT (per mode) -------
    for mode in ["beam", "sampling"]:                                  # loop modes
        mode_rows = [r for r in rows if r["decoding"] == mode]         # pick rows of this mode
        if not mode_rows:                                              # safety check
            print(f"\n[{mode}] No rows.")
            continue
        total = len(mode_rows)                                         # number of rows
        passed = sum(1 for r in mode_rows if r["constraint_passed"] == "True")  # count passes
        pct = 100.0 * passed / total                                   # percent passed
        avg_tokens = sum(int(r["tokens_out"]) for r in mode_rows) / total  # average tokens_out
        # choose a simple reflection sentence
        reflection = (
            "Beam search gives more control and passes constraints more often, but can sound stiff."
            if mode == "beam"
            else "Sampling sounds more varied, but fails constraints more often due to randomness."
        )
        # print the mini-report block
    print(f"\n=== Mini-report: {mode} ===")
    print(f"Rows: {total}")
    print(f"Constraint pass rate: {pct:.1f}%")
    print(f"Average tokens_out: {avg_tokens:.2f}")
    print(f"Reflection: {reflection}")
if __name__ == "__main__":
    main()