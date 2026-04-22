import os
import gdown
from striprtf.striprtf import rtf_to_text
from transformers import MarianMTModel, MarianTokenizer
import torch



# ----------------------------------------------------------------------
#  Locate and read the input RTF file
# ----------------------------------------------------------------------

input_file = "m25csa012_major/Q1/input.rtf"
for file in os.listdir("."):
    if file.startswith("input") and file.endswith(".rtf"):
        input_file = file
        break

if input_file is None:
    raise FileNotFoundError("Could not find the input RTF file in the downloaded folder.")

with open(input_file, "r", encoding="utf-8") as f:
    rtf_content = f.read()
    input_text = rtf_to_text(rtf_content)          # convert RTF to plain text

# Split the text into lines (each line will be translated separately)
lines = input_text.splitlines()

# ----------------------------------------------------------------------
# Load the pretrained MarianMT model and tokenizer
# ----------------------------------------------------------------------
model_name = "Helsinki-NLP/opus-mt-bn-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# ----------------------------------------------------------------------
# Translate each line and collect the results
# ----------------------------------------------------------------------
translations = []
for line in lines:
    if line.strip():                               # skip empty lines
        inputs = tokenizer([line], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            translated = model.generate(**inputs)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        translations.append(translated_text)
    else:
        translations.append("")                    # preserve blank lines

# ----------------------------------------------------------------------
# 5. Save the translated output to output.txt
# ----------------------------------------------------------------------
with open("output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(translations))

print("Translation completed. Output saved to output.txt")