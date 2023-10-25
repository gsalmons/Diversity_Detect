###OLD NOT IN USE. RETIRE THIS SCRIPT?
import os
import re
import sys
import json
path = "/bioProjectIds/ngramedScored/"
if not os.path.exists(path):
    os.mkdir(path)
# I want to load in each metadata file
for current_file in os.listdir('/bioProjectIds/oracleColumns'):
    uniquePhrases = set()
    ngramDict = dict()
    with open("/bioProjectIds/uniquePhrases.tsv", "r") as dictFile:
        for line in dictFile:
            ngramDict[line.rstrip("\n")] = 0
    if current_file.endswith(".tsv"):
        try:
            with open("/bioProjectIds/oracleColumns/" + current_file, "r") as readFile:
                for line in readFile:
                    line.rstrip("\n")
                    line = line.split("\t")
                    for phrase in line:
                        # I want to store unique tabs in a set
                        # Change anything about the phrases here?
                        phrase = re.sub("\"", " ", phrase)
                        phrase = re.sub("!", " ", phrase)
                        phrase = re.sub("?", " ", phrase)
                        phrase = re.sub(">", " ", phrase)
                        phrase = re.sub("\\", " ", phrase)
                        phrase = re.sub("#", " ", phrase)
                        phrase = re.sub("<", " ", phrase)
                        phrase = re.sub("=", " ", phrase)
                        phrase = re.sub("+", " ", phrase)
                        phrase = re.sub("-", " ", phrase)
                        phrase = re.sub("_", " ", phrase)
                        phrase = re.sub("*", " ", phrase)
                        phrase = re.sub("$", " ", phrase)
                        phrase = re.sub("@", " ", phrase)
                        phrase = re.sub("]", " ", phrase)
                        phrase = re.sub("[", " ", phrase)
                        phrase = re.sub("{", " ", phrase)
                        phrase = re.sub(")", " ", phrase)
                        phrase = re.sub("}", " ", phrase)
                        phrase = re.sub("(", " ", phrase)
                        phrase = re.sub("%", " ", phrase)
                        phrase = re.sub("^", " ", phrase)
                        phrase = re.sub("~", " ", phrase)
                        phrase = re.sub("&", " ", phrase)
                        phrase = re.sub("'", " ", phrase)
                        phrase = re.sub("|", " ", phrase)
                        phrase = re.sub(":", " ", phrase)
                        phrase = re.sub(",", " ", phrase)
                        phrase = re.sub("  ", " ", phrase)
                        phrase = phrase.lower()
                        uniquePhrases.add(phrase)
        except:
            print("This file did not open right", current_file)
        uniquePhrases.remove("")
        ngrams = set()

        for phrase in uniquePhrases:
            for i, character in enumerate(phrase):
                if character == " ":
                    continue
                if len(phrase) >= i+3:
                    ngramDict[phrase[i:i+3]] += 1
                elif len(phrase) == i+2:
                    ngramDict[phrase[i:i+2]+" "] += 1
        print(ngrams)
        # Save results
        current_file = current_file.split(".")[0] + ".json"
        with open(f"/bioProjectIds/ngramedScored/{current_file}", "w") as writeFile:
            writeFile.write(json.dumps(ngramDict))