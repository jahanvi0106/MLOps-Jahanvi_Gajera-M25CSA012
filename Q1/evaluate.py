import os
from striprtf.striprtf import rtf_to_text
from sacrebleu.metrics import BLEU
from sacrebleu import corpus_bleu

# ----------------------------------------------------------------------
# 1. Locate and read the reference RTF file
# ----------------------------------------------------------------------
ref_file = "m25csa012_major/Q1/output.rtf"
for file in os.listdir("."):
    if file.startswith("output") and file.endswith(".rtf"):
        ref_file = file
        break

if ref_file is None:
    raise FileNotFoundError("Could not find the reference RTF file in the downloaded folder.")

with open(ref_file, "r", encoding="utf-8") as f:
    rtf_content = f.read()
    ref_text = rtf_to_text(rtf_content)

# ----------------------------------------------------------------------
# 2. Read the generated translations
# ----------------------------------------------------------------------
with open("output.txt", "r", encoding="utf-8") as f:
    gen_text = f.read()

# ----------------------------------------------------------------------
# 3. Compute and print the BLEU score
# ----------------------------------------------------------------------
# bleu = BLEU(None)
# score = bleu.corpus_score(gen_text, [ref_text])
score = corpus_bleu([gen_text], [[ref_text]])
print(f"BLEU score: {score.score}")

# print(f"BLEU score: {score}")