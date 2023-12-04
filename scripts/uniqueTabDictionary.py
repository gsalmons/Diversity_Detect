"""Objectives: Generate a file that contains every unique tri-gram present in labeled and unlabeled BioProject metadata. 
Standardize punctuation and spaces in order to prevent extra noise. 
Inputs: Every bioproject metadata tsv
Outputs: uniquePhrases.tsv
"""


import os
import re

#load in each metadata file
uniquePhrases = set()
for current_file in os.listdir('/bioProjectIds/oracleColumns'):
    if current_file.endswith(".tsv"):
        with open("/bioProjectIds/oracleColumns/" + current_file, "r") as readFile:
            for line in readFile:
                line = line.rstrip("\n")
                line = line.split("\t")

                #Clean and store each unique tab separated value in a set.
                for phrase in line:
                    if ":" in line:
                        line = re.sub(":", "", line)
                    if ";" in line:
                        line = re.sub(";", "", line)
                    if "," in phrase:
                        phrase = re.sub(",", "", phrase)
                    if "." in phrase:
                        phrase = re.sub(".", "", phrase)
                    if " " in phrase:
                        phrase = re.sub(" ", "_", phrase)
                    if "®" in phrase:
                        phrase = re.sub("®", "_", phrase)
                    if "__" in phrase:
                        phrase = re.sub("__", "_", phrase)
                    phrase = phrase.lower()
                    uniquePhrases.add(phrase)

for current_file in os.listdir('/bioProjectIds/unlabeledColumns'):
    if current_file.endswith(".tsv"):
        with open("/bioProjectIds/unlabeledColumns/" + current_file, "r") as readFile:
            for line in readFile:
                line = line.rstrip("\n")
                line = line.split("\t")
                #Clean and store each unique tab separated value in a set.
                for phrase in line:
                    if "," in phrase:
                        phrase = re.sub(",", "", phrase)
                    if "." in phrase:
                        phrase = re.sub(".", "", phrase)
                    if ":" in line:
                        line = re.sub(":", "", line)
                    if "®" in phrase:
                        phrase = re.sub("®", "_", phrase)
                    if ";" in line:
                        line = re.sub(";", "", line)
                    if " " in phrase:
                        phrase = re.sub(" ", "_", phrase)
                    if "__" in phrase:
                        phrase = re.sub("__", "_", phrase)
                    phrase = phrase.lower()
                    uniquePhrases.add(phrase)

print("phrases:", len(uniquePhrases))
if "" in uniquePhrases:
    uniquePhrases.remove("")
else:
    print("none empty")
ngrams = set()

#Extract trigrams/bigrams
for phrase in uniquePhrases:
    for i, character in enumerate(phrase):
        if character == "_":
            continue
        if len(phrase) >= i+3:
            ngrams.add(phrase[i:i+3])
        elif len(phrase) == i+2:
            ngrams.add(phrase[i:i+2]+"_")
print("Num grams", len(ngrams))

with open("/bioProjectIds/uniquePhrases.tsv", "w") as writeFile:
    for gram in ngrams:
        writeFile.write(gram + "\n")