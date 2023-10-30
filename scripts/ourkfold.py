# Code modified from https://www.geeksforgeeks.org/stratified-k-fold-cross-validation/
import sys
# Import Required Modules.
from statistics import mean, stdev
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier  # Import Random Forest Classifier
import numpy as np 

# FEATCHING FEATURES AND TARGET VARIABLES IN ARRAY FORMAT.
yTruthDict = dict()
with open("/bioProjectIds/yTruthRandomSample.tsv", "r") as readFile:
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
# print(xRandomSample)
# print(bioProjectList)        
print(sum(yTruthList))
listedLists = xRandomSample
xRandomSample = np.array(xRandomSample)

# Create classifier object.
rf = RandomForestClassifier(n_estimators=100, random_state=1)  # You can adjust the parameters as needed

# Create StratifiedKFold object.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
lst_accu_stratified = []
train_index = 0
test_index = 0
bestShape = xRandomSample.shape
# random_array = np.random.randint(2, size=bestShape)
# Define the probabilities for 0 and 1
probability_0 = (allnums - num1) / allnums  # Probability for 0
probability_1 =  num1 / allnums # Probability for 1
print(probability_0, probability_1)
# Generate a random array based on the specified probabilities
random_array = np.random.choice([0, 1], size=bestShape, p=[probability_0, probability_1])

print(bestShape)
yTruthList = np.array(yTruthList)
print(yTruthList.shape)
try:
    #Swap out random_array with xRandomSample!
    for train_index, test_index in skf.split(xRandomSample, yTruthList):
        print(train_index, test_index)
        x_train_fold, x_test_fold = xRandomSample[train_index], xRandomSample[test_index]
        y_train_fold, y_test_fold = yTruthList[train_index], yTruthList[test_index]
        print('x_train_fold shape:', x_train_fold.shape)
        print('x_test_fold shape:', x_test_fold.shape)
        rf.fit(x_train_fold, y_train_fold)
        lst_accu_stratified.append(rf.score(x_test_fold, y_test_fold))
    # Print the output.
    print('List of possible accuracy:', lst_accu_stratified)
    print('\nMaximum Accuracy That can be obtained from this model is:',
        max(lst_accu_stratified) * 100, '%')
    print('\nMinimum Accuracy:',
        min(lst_accu_stratified) * 100, '%')
    print('\nOverall Accuracy:',
        mean(lst_accu_stratified) * 100, '%')
    print('\nStandard Deviation is:', stdev(lst_accu_stratified))
except:
    print(train_index, test_index)

##TODO: use predictproba to get probabilities. Do box plot as sanity check. 

from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_scores = rf.predict_proba(x_test_fold)[:, 1]  # Probability estimates of the positive class

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test_fold, y_scores)
roc_auc = auc(fpr, tpr)


# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig("/bioProjectIds/auroc.png")
plt.show()

y_pred = rf.predict(x_test_fold)

# Compute confusion matrix
cm = confusion_matrix(y_test_fold, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])

# Plot confusion matrix
plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.savefig('/bioProjectIds/confusion_matrix.png')
plt.show()

###We are attempting to find the most imporant ngrams
feature_importances = rf.feature_importances_

# Get the names of the features
feature_names = np.array(ngrams)

# Sort features based on importance
sorted_indices = np.argsort(feature_importances)[::-1]

# Select the top 100 n-grams
top_ngrams = feature_names[sorted_indices][:100]
top_importances = feature_importances[sorted_indices][:100]

# Plot the top 100 feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(top_importances)), top_importances)
plt.xticks(range(len(top_importances)), top_ngrams, rotation=45, ha="right")
plt.xlabel('N-gram')
plt.ylabel('Feature Importance')
plt.title('Top 100 Feature Importances in Random Forest')
plt.tight_layout()
plt.savefig('/bioProjectIds/mostRelevantNgrams.png')
plt.show()

#Save the ngrams by importance with their frequencies in race and nonrace. 
nonraceAverages = [0] * len(listedLists[0])
numDivN = 0
numDivR = 0
raceAverages = [0] * len(listedLists[0])
for i, columnInfo in enumerate(yTruthList):
    if columnInfo == 0:
        numDivN += 1
        for j, value in enumerate(listedLists[i]):
            nonraceAverages[j] += int(value)
    else:
        # print(bioProjectList[i], listedLists[i])
        for j, value in enumerate(listedLists[i]):
            raceAverages[j] += int(value)
        numDivR += 1
for k, value in enumerate(nonraceAverages):
    nonraceAverages[k] = value / numDivN
for k, value in enumerate(raceAverages):
    raceAverages[k] = value / numDivR

with open("/bioProjectIds/ngramFrequencyByCategory.tsv", "w") as writeFile:
    writeFile.write("Importance\tNgram\tFrequency in Race Columns\tFrequency in Nonrace Columns\n")
    for i, index in enumerate(sorted_indices):
        writeFile.write(f"{i+1}\t{ngrams[index]}\t{raceAverages[index]}\t{nonraceAverages[index]}\n")

#############################################################################
######REMOVING THE TOP 100 FEATURES WHAT WOULD HAPPEN?????####################
#############################################################################

# Sort features based on importance
# sorted_indices = np.argsort(feature_importances)

# Remove the top 100 n-grams. Tweak this. It could be the top 50, 100, 150, till it possibly break
#already did 50. So try 100 or so. 
top_ngrams_to_remove = sorted_indices[:100]
xRandomSample_reduced = np.delete(xRandomSample, top_ngrams_to_remove, axis=1)

# Re-create the StratifiedKFold object
skf = StratifiedKFold(n_splits=5, shuffle=True)

# Initialize the list for accuracy scores
lst_accu_stratified = []

try:
    for train_index, test_index in skf.split(xRandomSample_reduced, yTruthList):
        x_train_fold, x_test_fold = xRandomSample_reduced[train_index], xRandomSample_reduced[test_index]
        y_train_fold, y_test_fold = yTruthList[train_index], yTruthList[test_index]
        rf.fit(x_train_fold, y_train_fold)
        lst_accu_stratified.append(rf.score(x_test_fold, y_test_fold))

    # Print the output.
    print('List of possible accuracy without top 100 n-grams:', lst_accu_stratified)
    print('\nMaximum Accuracy That can be obtained without top 100 n-grams is:', max(lst_accu_stratified) * 100, '%')
    print('\nMinimum Accuracy without top 100 n-grams:', min(lst_accu_stratified) * 100, '%')
    print('\nOverall Accuracy without top 100 n-grams:', mean(lst_accu_stratified) * 100, '%')
    print('\nStandard Deviation without top 100 n-grams is:', stdev(lst_accu_stratified))
except:
    print(train_index, test_index)
y_scores = rf.predict_proba(x_test_fold)[:, 1]  # Probability estimates of the positive class

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test_fold, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve for the new model
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve (after removing top 100 n-grams)')
plt.legend(loc='lower right')
plt.savefig("/bioProjectIds/auroc_removed_top100.png")
plt.show()

y_pred = rf.predict(x_test_fold)

# Compute confusion matrix
cm = confusion_matrix(y_test_fold, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])

# Plot confusion matrix
plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.savefig('/bioProjectIds/confusion_matrix_removed_top100.png')
plt.show()