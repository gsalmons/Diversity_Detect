# Code modified from https://www.geeksforgeeks.org/stratified-k-fold-cross-validation/
import sys
from statistics import mean, stdev
from sklearn import preprocessing
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier 
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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
      
print(sum(yTruthList))
listedLists = xRandomSample
xRandomSample = np.array(xRandomSample)

# Create classifier object.
rf = RandomForestClassifier(n_estimators=100, random_state=1) 

# Create StratifiedKFold object.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
lst_accu_stratified = []
train_index = 0
test_index = 0
bestShape = xRandomSample.shape

# Define the probabilities for 0 and 1
probability_0 = (allnums - num1) / allnums  # Probability for 0
probability_1 =  num1 / allnums # Probability for 1
print(probability_0, probability_1)

# Generate a random array based on the specified probabilities
random_array = np.random.choice([0, 1], size=bestShape, p=[probability_0, probability_1])

print(bestShape)
yTruthList = np.array(yTruthList)
print(yTruthList.shape)
all_y_scores_0 = []
all_y_scores_1 = []

# Initialize empty lists to store probabilities for different cases
prob_0_when_true_0 = []  # Probability of predicting 0 when true label is 0
prob_0_when_true_1 = []  # Probability of predicting 0 when true label is 1
prob_1_when_true_1 = []  # Probability of predicting 1 when true label is 1
prob_1_when_true_0 = []  # Probability of predicting 1 when true label is 0

try:
    for train_index, test_index in skf.split(xRandomSample, yTruthList):
        x_train_fold, x_test_fold = xRandomSample[train_index], xRandomSample[test_index]
        y_train_fold, y_test_fold = yTruthList[train_index], yTruthList[test_index]
        rf.fit(x_train_fold, y_train_fold)
        y_scores = rf.predict_proba(x_test_fold)
        
        for i in range(len(y_scores)):
            if y_test_fold[i] == 0:
                if y_scores[i][0] > y_scores[i][1]:
                    prob_0_when_true_0.append(y_scores[i][0])  # Predicted 0 when true label is 0
                else:
                    prob_1_when_true_0.append(y_scores[i][1])  # Predicted 1 when true label is 0
                    print("Confused on", bioProjectList[i])
            else:
                if y_scores[i][0] > y_scores[i][1]:
                    prob_0_when_true_1.append(y_scores[i][0])  # Predicted 0 when true label is 1
                    print("Confused on ", bioProjectList[i])
                else:
                    prob_1_when_true_1.append(y_scores[i][1])  # Predicted 1 when true label is 1
except:
    print(train_index, test_index)
    # Create boxplots for the different cases
plt.figure(figsize=(8, 6))
boxplot = plt.boxplot([prob_0_when_true_0, prob_0_when_true_1, prob_1_when_true_1, prob_1_when_true_0],
patch_artist = True,
labels=['Predict 0 when True 0', 'Predict 0 when True 1', 'Predict 1 when True 1', 'Predict 1 when True 0'])
colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightpink']
for box, color in zip(boxplot['boxes'], colors):
    box.set_facecolor(color)
plt.xticks([0, 1, 2, 3], ['0 when True 0', '0 when True 1', '1 when True 1', 'Predict 1 when True 0'])
plt.ylabel("Probability")
plt.title("Probability Distribution")
plt.savefig("/bioProjectIds/probabilityDistribution.png")
plt.show()

with open("/bioProjectIds/kFoldTsvs/confidences.tsv", "w") as writeFile:
    writeFile.write("Category\tConfidence\n")
    for s in prob_0_when_true_0:
        writeFile.write(f"00\t{s}\n")
    for s in prob_0_when_true_1:
        writeFile.write(f"01\t{s}\n")
    for s in prob_1_when_true_0:
        writeFile.write(f"10\t{s}\n")
    for s in prob_1_when_true_1:
        writeFile.write(f"11\t{s}\n")

y_scores = rf.predict_proba(x_test_fold)[:, 1]  # Probability estimates of the positive class

# Calculate the AUC-ROC score
roc_auc = roc_auc_score(y_test_fold, y_scores)

# Print or save the AUC-ROC score
print(f'AUC-ROC Score: {roc_auc:.2f}')

with open("/bioProjectIds/kFoldTsvs/y_scores.tsv", "w") as writeFile:
    for s in y_scores:
        writeFile.write(f"{s}\t")

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test_fold, y_scores)
with open("/bioProjectIds/kFoldTsvs/faslePositiveRate.tsv", "w") as writeFile:
    for s in fpr:
        writeFile.write(f"{s}\t")

with open("/bioProjectIds/kFoldTsvs/truePositiveRate.tsv", "w") as writeFile:
    for s in tpr:
        writeFile.write(f"{s}\t")

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig("/bioProjectIds/auroc1.png")
plt.show()

y_pred = rf.predict(x_test_fold)

# Compute CONFUSION MATRIX#
cm = confusion_matrix(y_test_fold, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])

# Plot confusion matrix
plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.savefig('/bioProjectIds/confusion_matrix.png')
plt.show()

# find the most imporant ngrams
feature_importances = rf.feature_importances_

# Get the names of the features
feature_names = np.array(ngrams)

# Sort features based on importance
sorted_indices = np.argsort(feature_importances)[::-1]

# Select the top n-grams
numTop = 12000
top_ngrams = feature_names[sorted_indices][:numTop]
top_importances = feature_importances[sorted_indices][:numTop]

# Plot the top feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(top_importances)), top_importances)
plt.xticks(range(len(top_importances)), top_ngrams, rotation=45, ha="right")
plt.xlabel('N-gram')
plt.ylabel('Feature Importance')
plt.title(f'Top {numTop} Feature Importances in Random Forest')
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

######REMOVING THE TOP X FEATURES WHAT WOULD HAPPEN?????##################

# Remove the top X n-grams 
top_ngrams_to_remove = sorted_indices[:numTop]
xRandomSample_reduced = np.delete(xRandomSample, top_ngrams_to_remove, axis=1)

# Re-create the StratifiedKFold object
skf = StratifiedKFold(n_splits=5, shuffle=True)

# Initialize the list for accuracy scores
lst_accu_stratified = []
false_positives = []  # Store columns with false positives
false_negatives = []  # Store columns with false negatives
try:
    for train_index, test_index in skf.split(xRandomSample_reduced, yTruthList):
        x_train_fold, x_test_fold = xRandomSample_reduced[train_index], xRandomSample_reduced[test_index]
        y_train_fold, y_test_fold = yTruthList[train_index], yTruthList[test_index]
        rf.fit(x_train_fold, y_train_fold)
        lst_accu_stratified.append(rf.score(x_test_fold, y_test_fold))
        for i in range(len(y_test_fold)):
            true_label = y_test_fold[i]
            predicted_label = y_pred[i]

            # Check for false positives (true negative but predicted positive)
            if true_label == 0 and predicted_label == 1:
                false_positives.append(bioProjectList[i])

            # Check for false negatives (true positive but predicted negative)
            if true_label == 1 and predicted_label == 0:
                false_negatives.append(bioProjectList[i])

    # Print the output.
    print(f'List of possible accuracy without top {numTop} n-grams:', lst_accu_stratified)
    print(f'\nMaximum Accuracy That can be obtained without top {numTop} n-grams is:', max(lst_accu_stratified) * 100, '%')
    print(f'\nMinimum Accuracy without top {numTop} n-grams:', min(lst_accu_stratified) * 100, '%')
    print(f'\nOverall Accuracy without top {numTop} n-grams:', mean(lst_accu_stratified) * 100, '%')
    print(f'\nStandard Deviation without top {numTop} n-grams is:', stdev(lst_accu_stratified))
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
plt.title(f'Receiver Operating Characteristic (ROC) Curve (after removing top {numTop} n-grams)')
plt.legend(loc='lower right')
plt.savefig(f"/bioProjectIds/auroc_removed_top{numTop}.png")
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
plt.savefig(f'/bioProjectIds/confusion_matrix_removed_top{numTop}.png')
plt.show()
