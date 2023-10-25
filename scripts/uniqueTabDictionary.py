import os
import re
import sys
# I want to load in each metadata file
uniquePhrases = set()
for current_file in os.listdir('/bioProjectIds/oracleColumns'):
    if current_file.endswith(".tsv"):
        # try:
        with open("/bioProjectIds/oracleColumns/" + current_file, "r") as readFile:
            for line in readFile:
                line = line.rstrip("\n")
                line = line.split("\t")
                for phrase in line:
                    # I want to store unique tabs in a set
                    # Change anything about the phrases here?
                    # phrase = re.sub("\"", " ", phrase)
                    # phrase = re.sub("!", " ", phrase)
                    # phrase = re.sub("?", " ", phrase)
                    # phrase = re.sub(">", " ", phrase)
                    # phrase = re.sub("\\", " ", phrase)
                    # phrase = re.sub("#", " ", phrase)
                    # phrase = re.sub("<", " ", phrase)
                    # phrase = re.sub("=", " ", phrase)
                    # phrase = re.sub("+", " ", phrase)
                    # phrase = re.sub("-", " ", phrase)
                    # phrase = re.sub("_", " ", phrase)
                    # phrase = re.sub("*", " ", phrase)
                    # phrase = re.sub("$", " ", phrase)
                    # phrase = re.sub("@", " ", phrase)
                    # phrase = re.sub("]", " ", phrase)
                    # phrase = re.sub("[", " ", phrase)
                    # phrase = re.sub("{", " ", phrase)
                    # phrase = re.sub(")", " ", phrase)
                    # phrase = re.sub("}", " ", phrase)
                    # phrase = re.sub("(", " ", phrase)
                    # phrase = re.sub("%", " ", phrase)
                    # phrase = re.sub("^", " ", phrase)
                    # phrase = re.sub("~", " ", phrase)
                    # phrase = re.sub("&", " ", phrase)
                    # phrase = re.sub("'", " ", phrase)
                    # phrase = re.sub("|", " ", phrase)
                    # phrase = re.sub(":", " ", phrase)
                    # phrase = re.sub(",", " ", phrase)
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
        # except:
        #     print("This file did not open right", current_file)
for current_file in os.listdir('/bioProjectIds/unlabeledColumns'):
    if current_file.endswith(".tsv"):
        # try:
        with open("/bioProjectIds/unlabeledColumns/" + current_file, "r") as readFile:
            for line in readFile:
                line = line.rstrip("\n")
                line = line.split("\t")
                for phrase in line:
                    # I want to store unique tabs in a set
                    # Change anything about the phrases here?
                    # phrase = re.sub("\"", " ", phrase)
                    # phrase = re.sub("!", " ", phrase)
                    # phrase = re.sub("?", " ", phrase)
                    # phrase = re.sub(">", " ", phrase)
                    # phrase = re.sub("\\", " ", phrase)
                    # phrase = re.sub("#", " ", phrase)
                    # phrase = re.sub("<", " ", phrase)
                    # phrase = re.sub("=", " ", phrase)
                    # phrase = re.sub("+", " ", phrase)
                    # phrase = re.sub("-", " ", phrase)
                    # phrase = re.sub("_", " ", phrase)
                    # phrase = re.sub("*", " ", phrase)
                    # phrase = re.sub("$", " ", phrase)
                    # phrase = re.sub("@", " ", phrase)
                    # phrase = re.sub("]", " ", phrase)
                    # phrase = re.sub("[", " ", phrase)
                    # phrase = re.sub("{", " ", phrase)
                    # phrase = re.sub(")", " ", phrase)
                    # phrase = re.sub("}", " ", phrase)
                    # phrase = re.sub("(", " ", phrase)
                    # phrase = re.sub("%", " ", phrase)
                    # phrase = re.sub("^", " ", phrase)
                    # phrase = re.sub("~", " ", phrase)
                    # phrase = re.sub("&", " ", phrase)
                    # phrase = re.sub("'", " ", phrase)
                    # phrase = re.sub("|", " ", phrase)
                    # phrase = re.sub(":", " ", phrase)
                    # phrase = re.sub(",", " ", phrase)
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
        # except:
        #     print("This file did not open right", current_file)

print("phrases:", len(uniquePhrases))
if "" in uniquePhrases:
    uniquePhrases.remove("")
else:
    print("none empty")
ngrams = set()

for phrase in uniquePhrases:
    for i, character in enumerate(phrase):
        if character == "_":
            continue
        if len(phrase) >= i+3:
            ngrams.add(phrase[i:i+3])
        elif len(phrase) == i+2:
            ngrams.add(phrase[i:i+2]+"_")
print("Num grams", len(ngrams))
# Save results
with open("/bioProjectIds/uniquePhrases.tsv", "w") as writeFile:
    for gram in ngrams:
        writeFile.write(gram + "\n")