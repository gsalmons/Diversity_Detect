import os
import re
ngrams = dict()
numProj = 0
def reset0(dictionary):
    for key in dictionary:
        dictionary[key] = 0
    return(dictionary)

with open("/bioProjectIds/uniquePhrases.tsv", "r") as dictFile:
    for line in dictFile:
        line = line.rstrip("\n")
        ngrams[line] = 0
with open("/bioProjectIds/masterInputOracle.tsv", "w") as writeFile:
    writeFile.write("BioProjectID\tFieldName\tUniqueValues")
    for gram in sorted(list(ngrams.keys())):
        writeFile.write("\t" + gram)
    writeFile.write("\n")
    for current_file in os.listdir('/bioProjectIds/oracleColumns'):
        if current_file.endswith(".tsv"):
            bioProjectId = current_file[:-4]
            with open("/bioProjectIds/oracleColumns/" + current_file, "r") as readFile:
                for line in readFile:
                    ngrams = reset0(ngrams)

                    line = line.rstrip("\n")
                    if line == "":
                        continue
                    if "," in line:
                        line = re.sub(",", "", line)
                    if "." in line:
                        line = re.sub(".", "", line)
                    if " " in line:
                        line = re.sub(" ", "_", line)
                    if ":" in line:
                        line = re.sub(":", "", line)
                    if ";" in line:
                        line = re.sub(";", "", line)
                    if "®" in line:
                        line = re.sub("®", "_", line)
                    if "__" in line:
                        line = re.sub("__", "_", line)
                    
                    line = line.lower()
                    line = line.split("\t")
                    fieldName = line[0]
                    uniqueValues = line[1:]
                    for m in line:
                        for i, character in enumerate(m):
                            if character == "_":
                                continue
                            if len(m) >= i+3:
                                if (m[i:i+3]) not in ngrams:
                                    print("Couldnt find", (m[i:i+3]))
                                    continue 
                                ngrams[(m[i:i+3])] = 1
                            elif len(m) == i+2:
                                if m[i:i+2]+"_" not in ngrams:
                                    print("Couldnt find", m[i:i+2]+"_")
                                    continue
                                ngrams[(m[i:i+2]+"_")] = 1
                    writeFile.write(f"{bioProjectId}\t{fieldName}\t")
                    uniqueValues = ' '.join(uniqueValues)
                    writeFile.write(f"{uniqueValues}")
                    for gram in sorted(list(ngrams.keys())):
                        writeFile.write("\t" + str(ngrams[gram]))
                    writeFile.write("\n")