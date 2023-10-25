# Code modified from https://www.geeksforgeeks.org/stratified-k-fold-cross-validation/
import sys

yTruthDict = dict()
with open("bioProjectIds/yTruthRandomSample.tsv", "r") as readFile:
    header = readFile.readline()
    for line in readFile:
        line = line.rstrip("\n")
        line = line.split("\t")
        tempDict = dict()
        if line[1] == "0":
            tempDict["overall"] = 0
            yTruthDict[line[0]] = tempDict
        elif line[1] == "1":
            tempDict["overall"] = 1
            tempDict["goodColumns"] = line[2].split(" ")
            yTruthDict[line[0]] = tempDict 
        else:
            print("Minor problem....", line[0], line[1])               
bioProjectList = []
xRandomSample = []
yTruthList = []
ngrams = []
num1 = 0
allnums = 0
with open("bioProjectIds/masterInputOracle2.tsv", "r") as readFile:
    header = readFile.readline()
    ngrams = header.split("\t")[3:]
    for line in readFile:
        line = line.rstrip("\n")
        line = line.split("\t")
        bioProjid = line[0]
        if bioProjid not in yTruthDict:
            continue
        columnName = line[1]
        futureTensor = line[3:]
        xRandomSample.append(futureTensor)
        bioProjectList.append(bioProjid + columnName)
        yl = 0
        if yTruthDict[bioProjid]["overall"] == 1:
            if columnName in yTruthDict[bioProjid]["goodColumns"]:
                yl = 1
                num1 += 1
        yTruthList.append(yl)
        allnums += 1
# print(xRandomSample)
# print(bioProjectList)        
# print(sum(yTruthList))
listedLists = xRandomSample

#Save the ngrams by importance with their frequencies in race and nonrace. 
nonraceAverages = [0] * len(listedLists[0])
numDivN = 0
numDivR = 0
raceAverages = [0] * len(listedLists[0])
firstRaceOne = True
for i, columnInfo in enumerate(yTruthList):
    if columnInfo == 0:
        numDivN += 1
        for j, value in enumerate(listedLists[i]):
            nonraceAverages[j] += int(value)
    else:
        if firstRaceOne:
            print(bioProjectList[i])
        for j, value in enumerate(listedLists[i]):
            if value == "1" and firstRaceOne:
                print(ngrams[j])
            raceAverages[j] += int(value)
        numDivR += 1
        firstRaceOne = False
for k, value in enumerate(nonraceAverages):
    nonraceAverages[k] = value / numDivN
for k, value in enumerate(raceAverages):
    raceAverages[k] = value / numDivR

with open("bioProjectIds/ngramFreqByCategory.tsv", "w") as writeFile:
    writeFile.write("Importance\tNgram\tFrequency in Race Columns\tFrequency in Nonrace Columns\n")
    for index in range(len(yTruthList)):
        writeFile.write(f"\t{ngrams[index]}\t{raceAverages[index]}\t{nonraceAverages[index]}\n")