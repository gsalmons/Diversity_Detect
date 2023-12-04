"""Objectives: Remove tri-grams that occur less than two times across all labeled BioProjects because they will not be informative.
Inputs: masterInputOracle.tsv
Outputs: masterInputOracle2.tsv
"""


#Only include ngrams that occur more than one time total in the file. 
countingDict = dict()
import sys
import re
header = ""
with open("/bioProjectIds/masterInputOracle.tsv", "r") as readFile:
    header = readFile.readline()
    header = header.rstrip("\n")
    header = header.split("\t")
    header = header[3:]
    print(len(header), "H length")
    for h in header:
        countingDict[h] = 0
    for k, line in enumerate(readFile):
       
        line = line.rstrip("\n")
        line = line.split("\t")
        counts = line[3:]
        for i, num in enumerate(counts):
            try:
                if num == "":
                    print(i)
                    num = 0
                num = int(num)
                countingDict[header[i]] += num
            except:
                print(i)
                sys.exit()
toRemove = list()
for key in countingDict:
    if countingDict[key] < 2: 
        toRemove.append(key)
        print(len(key))
        print(key)
print((toRemove))

importantSet = sorted(list(set(countingDict.keys()) - set(toRemove)))
print(len(importantSet))
with open("/bioProjectIds/masterInputOracle.tsv", "r") as readFile:
    with open("/bioProjectIds/masterInputOracle2.tsv", "w") as writeFile:
        head = readFile.readline()
        head = head.rstrip("\n")
        head = head.split("\t")
        writeFile.write(f"{head[0]}\t{head[1]}\t{head[2]}")
        for column in importantSet:
            writeFile.write(f"\t{column}")
        importantSet = set(importantSet)
        writeFile.write("\n")
        head = head[3:]
       
        for k, line in enumerate(readFile):
            line = line.rstrip("\n")
            line = line.split("\t")
            counts = line[3:]
            writeFile.write(f"{line[0]}")
            for initialPart in line[1:3]:
                writeFile.write(f"\t{initialPart}")
            for i, num in enumerate(counts):
                try:
                    if header[i] in importantSet:
                        writeFile.write(f"\t{num}")
                except:
                    continue
            writeFile.write("\n")