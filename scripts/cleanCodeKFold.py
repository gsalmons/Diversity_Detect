import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from statistics import mean, stdev

def loadLabeledData():
    y_truth_dict = {}
    with open("/bioProjectIds/yTruthRandomSample.tsv", "r") as file:
        header = file.readline()
        for line in file:
            line = line.rstrip("\n").split("\t")
            temp_dict = {"overall": int(line[1])}
            if line[1] == "1":
                temp_dict["goodColumns"] = line[2].split(" ")
            y_truth_dict[line[0]] = temp_dict
    return y_truth_dict
def loadNgramInfo():
    bioProjectList = []
    xRandomSample = []
    yTruthList = []
    ngrams = []
    num1 = 0
    allnums = 0
    with open("/bioProjectIds/masterInputOracle2.tsv", "r") as readFile:
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
    return (bioProjectList, xRandomSample, yTruthList, ngrams, num1, allnums)

y_truth_dict = loadLabeledData()
bioProjectList, xRandomSample, yTruthList, ngrams, num1, allnums = loadNgramInfo()

print(sum(yTruthList))
listedLists = xRandomSample
xRandomSample = np.array(xRandomSample)