import os
import argparse
from transformers import pipeline, set_seed
import re

os.environ['CUDA_VISIBLE_DEVICES'] = ''
set_seed(42)
def read_lines(path):
    with open(path, 'r', encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def word_count(text):
    return len(text.strip().split())

def one_sentence_ending_period(text):
    t = " ".join(text.strip().split())               
    if not t.endswith("."):                         
        return False
    enders = sum(ch in ".!?" for ch in t)            
    return enders == 1

def validate_rewrite(text):
    """Return (passed, notes) for: one sentence, ends '.', 8–24 words."""
    t = " ".join(text.strip().split())               
    notes = []                                       # collect reasons if fails
    if not t.endswith("."): 
        notes.append("no period")
    else:
        if not one_sentence_ending_period(t): 
            notes.append(">1 sentence or punctuation")
    wc = word_count(t)
    if wc < 8: notes.append("too short")
    if wc > 24: notes.append("too long")
    return (len(notes) == 0), "; ".join(notes)
def validate_rewrite_guarded(original: str, text: str):
    """
    Same as validate_rewrite, but ALSO fails if output == input (case/space-insensitive).
    Returns (passed_bool, notes_str).
    """
    # run your existing checks first
    ok, notes = validate_rewrite(text)

    
    same = text.strip().lower() == original.strip().lower()
    if same:
        ok = False
        notes = (notes + "; " if notes else "") + "unchanged"

    return ok, notes

def prompt_rewrite(line):
    
    return (
        "Paraphrase the sentence using simpler English and different wording. "
        "Do NOT repeat any phrase longer than 3 words from the original. "
        "Keep the same meaning. Output exactly one sentence, 8–24 words, ending with a period.\n\n"
        f"Sentence: \"{line}\""
        
    )

def validate_summary(text):
    t = " ".join(text.strip().split())                 # normalize spaces
    notes = []                                         # collect reasons if fails
    if not t.endswith("."):                            # must end with '.'
        notes.append("no period")
    else:
        if not one_sentence_ending_period(t):          # reuse your helper
            notes.append(">1 sentence or punctuation")
    return (len(notes) == 0), "; ".join(notes)

def prompt_summarize(line: str):
    return (
        "Summarize the following in exactly one sentence ending with a period:\n\n"
        f"{line}"
    )
def force_one_sentence(t: str) -> str:
    """Post-process model text to a single clean sentence that ends with '.'."""
    t = " ".join(t.strip().split())
    # Drop any echoed instructions like 'Summarize ...'
    t = re.sub(r'(?i)summarize.*?:\s*[“"“”]?', "", t)
    # Take first sentence by splitting on . ! ?
    parts = re.split(r"[.!?]\s+", t)
    first = parts[0].strip(' "\'“”')
    return (first + ".") if first else "."

def main():
    parser = argparse.ArgumentParser(description="Task 3 — Step 2: simple REWRITE with beam search")
    parser.add_argument("--task", required=True, choices=["rewrite","summarize","sentiment"],
                        help="In this step, only 'rewrite' is implemented.")
    parser.add_argument("--infile", required=True, 
                        help="Path to input .txt (one input per line)")
    parser.add_argument("--neutral_band", type=float, default=0.10,
                    help="± band around 0.5 mapped to NEUTRAL (default: 0.10)")
    
    args = parser.parse_args()

    lines = read_lines(args.infile)                  # load inputs
    if not lines:                                    # empty file guard
        print("No non-empty lines found in infile.")
        return

    
        # --- SUMMARIZE task (DistilBART) ---
    if args.task == "summarize":
        # Create the summarization pipeline on CPU
        summ = pipeline("summarization",
                        model="sshleifer/distilbart-cnn-12-6",
                        device=-1)

        print(f"CPU mode ON. Loaded {len(lines)} lines from: {args.infile}")
        print("\n--- SUMMARIZE (beam; retry with sampling if needed) ---\n")

        beam_passes = 0              # how many passed on beam
        retry_triggered = 0          # how many needed a retry
        retry_fixed = 0              # how many the retry fixed
        beam_lengths = []            # word counts of beam summaries

        samp_passes = 0
        samp_lengths = []

        for i, line in enumerate(lines, 1):
            

            # ---- BEAM (deterministic) ----
            beam_raw = summ(
                line,
                do_sample=False,          
                num_beams=4,               
                no_repeat_ngram_size=3,    
                encoder_no_repeat_ngram_size=4,
                length_penalty=2.0,
                min_length=10,
                max_length=20
            )[0]["summary_text"].strip()

            beam =force_one_sentence(beam_raw)                  # normalize spaces
            beam_ok, beam_notes = validate_summary(beam) # one-sentence + period
            wc = len(beam.split())                        # count words
            beam_lengths.append(wc)                      # collect for avg
            beam_passes += int(beam_ok)                  # count pass

            samp_raw = summ(
                line,                         # IMPORTANT: pass raw text to summarization pipeline
                do_sample=True,
                temperature=1.0,
                top_p=0.92,
                top_k=50,
                no_repeat_ngram_size=3,
                encoder_no_repeat_ngram_size=4,  
                length_penalty=2.0,
                min_length=10,
                max_length=20,
                early_stopping=True
            )[0]["summary_text"].strip()

            samp = force_one_sentence(samp_raw)
            samp_ok, _ = validate_summary(samp)
            samp_wc = len(samp.split())
            samp_lengths.append(samp_wc)
            samp_passes += int(samp_ok)

            retried = False
            retry_pass = False

            # ---- RETRY (sampling) IF beam failed ----
            if not beam_ok:
                retried = True
                retry_triggered += 1
                retry_raw = summ(
                    line,
                    do_sample=True,            
                    temperature=1.1,           
                    top_p=0.90,                
                    top_k=50,                  
                    no_repeat_ngram_size=3,
                    encoder_no_repeat_ngram_size=4,
                    length_penalty=2.0,
                    min_length=10,
                    max_length=20,
                    early_stopping=True
                )[0]["summary_text"].strip()
                retry = force_one_sentence(retry_raw)
                retry_pass, _ = validate_summary(retry)
                if retry_pass:
                    retry_fixed += 1

            # ---- print a short per-line report ----
            print(f"{i:02d}. beam_pass={beam_ok} | words={wc:02d} | notes={beam_notes or '-'}"
                  f"{' | retry_triggered' if retried else ''}"
                  f"{' | retry_fixed' if retry_pass else ''}")
            print(f"    input : {line[:120]}{'...' if len(line)>120 else ''}")
            print(f"    output: {beam}")
            if retried:
                print(f"    retry : {retry}\n")
            else:
                print()

        # ---- tiny metrics summary ----
        n = len(lines)
        print("\n=== METRICS (Summarize) ===")
        print(f"Beam pass-rate       : {100.0 * beam_passes / n:.1f}%")
        print(f"Sampling pass-rate   : {100.0 * samp_passes / n:.1f}%")
        print(f"Avg length — beam    : {sum(beam_lengths)/n:.2f} words")
        print(f"Avg length — sampling: {sum(samp_lengths)/n:.2f} words")
        print(f"Retry triggered      : {retry_triggered} case(s)")
        print(f"Retry fix-rate       : {0.0 if retry_triggered==0 else 100.0 * retry_fixed / retry_triggered:.1f}%")
        return


    if args.task == "rewrite":
        print(f"CPU mode ON. Loaded {len(lines)} lines from: {args.infile}")
        print("\n--- REWRITE (beam, deterministic) ---\n")

    
         # --- REWRITE task (FLAN-T5) ---
         # Create the FLAN-T5 pipeline on CPU
        rew = pipeline("text2text-generation",           # create FLAN-T5 pipeline once
                    model="google/flan-t5-base",
                    device=-1) 
        
        
        beam_passes = 0          # how many passed on beam
        retry_triggered = 0      # how many needed a retry
        retry_fixed = 0          # how many the retry fixed
        beam_lengths = []  

        samp_passes = 0              # sampling baseline pass counter
        samp_lengths = []                    

        for i, line in enumerate(lines, 1):              # process each input line
            prompt = prompt_rewrite(line)                # build prompt

            beam_out = rew(                                   # generate with BEAM (no sampling)
                prompt,
                do_sample=False,
                num_beams=6,
                no_repeat_ngram_size=3,
                encoder_no_repeat_ngram_size=4,
                repetition_penalty=1.1,
                max_new_tokens=64
            )[0]["generated_text"].strip()               # take text and strip spaces

            beam_out = " ".join(beam_out.split())  
                
            beam_ok, beam_notes = validate_rewrite_guarded(line, beam_out)  # check constraints
            
            beam_wc = word_count(beam_out)               # word count
            beam_lengths.append(beam_wc)                 # save for average
            beam_passes += int(beam_ok)  
            retried = False
            retry_pass = False

            samp_out = rew(
                prompt,                        # same prompt as beam
                do_sample=True,                # enable sampling
                temperature=0.9,               # moderate creativity
                top_p=0.95,                    # nucleus sampling
                top_k=50,                      # cap candidate pool
                no_repeat_ngram_size=3,        # decoder anti-repeat
                
                encoder_no_repeat_ngram_size=4,
                repetition_penalty=1.15,
                max_new_tokens=64
            )[0]["generated_text"].strip()

            samp_out = " ".join(samp_out.split())
            samp_ok, _ = validate_rewrite_guarded(line, samp_out)
            samp_wc = word_count(samp_out)
            samp_lengths.append(samp_wc)
            samp_passes += int(samp_ok)

            if not beam_ok:
                retried = True                           # mark that we retried
                retry_triggered += 1                     # count retries
                retry_out = rew(                         # sampling with slightly different settings
                    prompt_rewrite(line),
                    do_sample=True,                      # enable sampling
                    temperature=1.2,                     # a bit spicier than default
                    top_p=0.92,                          # slightly tighter nucleus
                    top_k=40, 
                    no_repeat_ngram_size=3,
                    encoder_no_repeat_ngram_size=4,
                    repetition_penalty=1.2,                                                    
                    max_new_tokens=64                    # same cap
                )[0]["generated_text"].strip()

                retry_out = " ".join(retry_out.split())  # normalize spaces
                retry_pass, _ = validate_rewrite_guarded(line, retry_out)  # check constraints
                if retry_pass:
                    retry_fixed += 1 


            # print a tiny, clear report line + the texts
            print(f"{i:02d}. beam_pass={beam_ok} | words={beam_wc:02d} | notes={beam_notes or '-'}"
                    f"{' | retry_triggered' if retried else ''}"
                    f"{' | retry_fixed' if retry_pass else ''}")
            print(f"    input : {line}")
            print(f"    output: {beam_out}")
            if retried:
                print(f"    retry : {retry_out}\n")
            else:
                print()

        n = len(lines)
        print("\n=== METRICS (Rewrite) ===")
        print(f"Beam pass-rate       : {100.0 * beam_passes / n:.1f}%")
        print(f"Sampling pass-rate   : {100.0 * samp_passes / n:.1f}%")
        print(f"Avg length — beam    : {sum(beam_lengths)/n:.2f} words")
        print(f"Avg length — sampling: {sum(samp_lengths)/n:.2f} words")
        print(f"Retry triggered      : {retry_triggered} case(s)")
        print(f"Retry fix-rate       : {0.0 if retry_triggered==0 else 100.0 * retry_fixed / retry_triggered:.1f}%")


    # --- SENTIMENT task (DistilBERT + neutral band) ---
    if args.task == "sentiment":
        # Load the classifier on CPU
        sent = pipeline("text-classification",
                        model="distilbert-base-uncased-finetuned-sst-2-english",
                        device=-1)

        band = args.neutral_band            # e.g., 0.10 -> neutral if score within [0.40, 0.60]
        raw_counts  = {"POSITIVE": 0, "NEGATIVE": 0}
        neut_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}

        print(f"CPU mode ON. Loaded {len(lines)} lines from: {args.infile}")
        print(f"Neutral band: ±{band:.2f} around 0.50\n")
        print("--- SENTIMENT (raw vs neutralized) ---\n")

        for i, line in enumerate(lines, 1):
            res = sent(line, truncation=True)[0]   # {'label': 'POSITIVE'/'NEGATIVE', 'score': float}
            label = res["label"]
            score = float(res["score"])
            raw_counts[label] += 1

            # Neutralize if score is close to 0.5 by ±band
            if abs(score - 0.5) <= band:
                neutralized = "NEUTRAL"
            else:
                neutralized = label
            neut_counts[neutralized] += 1

            print(f"{i:02d}. raw={label} ({score:.3f}) -> neutralized={neutralized}")
            print(f"    text: {line}\n")

        n = len(lines)
        # Simple metrics block
        print("=== METRICS (Sentiment) ===")
        print(f"Raw distribution        : POS={raw_counts['POSITIVE']}, NEG={raw_counts['NEGATIVE']}")
        print(f"Neutralized distribution: POS={neut_counts['POSITIVE']}, "
            f"NEG={neut_counts['NEGATIVE']}, NEU={neut_counts['NEUTRAL']}")
        print(f"Neutral band used       : ±{band:.2f} around 0.50")
        print("\nNote: SST-2 is a binary model (POS/NEG). "
            "The neutral band maps uncertain scores near 0.5 to NEUTRAL so ambiguous lines aren’t forced into POS/NEG.")
        return

if __name__ == "__main__":
    main()

