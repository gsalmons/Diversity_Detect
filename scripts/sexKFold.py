# Code modified from https://www.geeksforgeeks.org/stratified-k-fold-cross-validation/
import sys
# Import Required Modules.
from statistics import mean, stdev
from sklearn import preprocessing
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier  # Import Random Forest Classifier
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import random

random.seed(1)

# FEATCHING FEATURES AND TARGET VARIABLES IN ARRAY FORMAT.
yTruthDict = dict()
with open("/bioProjectIds/sexLabeled.tsv", "r") as readFile:
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
with open("/bioProjectIds/masterInputOracle.tsv", "r") as readFile:
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
all_y_scores_0 = []
all_y_scores_1 = []
# Initialize empty lists to store probabilities for different cases
prob_0_when_true_0 = []  # Probability of predicting 0 when true label is 0
prob_0_when_true_1 = []  # Probability of predicting 0 when true label is 1
prob_1_when_true_1 = []  # Probability of predicting 1 when true label is 1
prob_1_when_true_0 = []  # Probability of predicting 1 when true label is 0
counter = 0
allyscores = []
allytestfold = []
whichFold = []
whichColumns = []
try:
    for train_index, test_index in skf.split(xRandomSample, yTruthList):
        x_train_fold, x_test_fold = xRandomSample[train_index], xRandomSample[test_index]
        y_train_fold, y_test_fold = yTruthList[train_index], yTruthList[test_index]
        rf.fit(x_train_fold, y_train_fold)
        y_scores = rf.predict_proba(x_test_fold)
        counter += 1
        # for i in range(len(y_scores)):
        #     if y_test_fold[i] == 0:
        #         if y_scores[i][0] > y_scores[i][1]:
        #             prob_0_when_true_0.append(y_scores[i][0])  # Predicted 0 when true label is 0
        #         else:
        #             prob_1_when_true_0.append(y_scores[i][1])  # Predicted 1 when true label is 0
        #             print("Confused on", bioProjectList[i], counter)
        #     else:
        #         if y_scores[i][0] > y_scores[i][1]:
        #             prob_0_when_true_1.append(y_scores[i][0])  # Predicted 0 when true label is 1
        #             print("Confused on ", bioProjectList[i], counter)
        #         else:
        #             prob_1_when_true_1.append(y_scores[i][1])  # Predicted 1 when true label is 1
        y_scores = rf.predict_proba(x_test_fold)[:, 1]  #TODO: use pos class for boxplot probs. 
        precision, recall, _ = precision_recall_curve(y_test_fold, y_scores)
        auc_pr = auc(recall, precision)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {auc_pr:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.show()
        plt.savefig(f'/bioProjectIds/sex/precision_recall_curve_allsub_{counter}.png')
        for i in range(len(y_scores)):
            allyscores.append(y_scores[i])
        for i in range(len(y_test_fold)):
            allytestfold.append(y_test_fold[i])
            whichFold.append(counter)
            whichColumns.append(bioProjectList[test_index[i]])
        # for i in range(len(test_index)):
        # plt.figure(figsize=(8, 6))
        # boxplot = plt.boxplot([prob_0_when_true_0, prob_0_when_true_1, prob_1_when_true_1, prob_1_when_true_0],
        # patch_artist = True,
        # labels=['Predict 0 when True 0', 'Predict 0 when True 1', 'Predict 1 when True 1', 'Predict 1 when True 0'])
        # colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightpink']
        # for box, color in zip(boxplot['boxes'], colors):
        #     box.set_facecolor(color)
        # plt.xticks([0, 1, 2, 3], ['0 when True 0', '0 when True 1', '1 when True 1', 'Predict 1 when True 0'])
        # plt.ylabel("Probability")
        # plt.title("Probability Distribution")
        # plt.savefig(f"/bioProjectIds/probabilityDistribution_allsub_{counter}.png")
        # plt.show()

except:
    print(train_index, test_index)
    # Create boxplots for the different cases

#Precision recall
# y_scores = rf.predict_proba(x_test_fold)[:, 1]  #TODO: use pos class for boxplot probs. 
precision, recall, _ = precision_recall_curve(allytestfold, allyscores)
auc_pr = auc(recall, precision)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {auc_pr:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()
plt.savefig('/bioProjectIds/sex/precision_recall_curve.png')

with open("/bioProjectIds/sex/confidencesallsub.tsv", "w") as writeFile:
    writeFile.write(f"Fold\tPrediction\tTruth\tProj&Col\n")
    for i in range(len(allytestfold)):
        writeFile.write(f"{whichFold[i]}\t{allyscores[i]}\t{allytestfold[i]}\t{whichColumns[i]}\n")
        if allytestfold[i] == "1" and float(allyscores[i]) < 0.5:
            print(whichColumns[i])
        elif allytestfold[i] == "0" and float(allyscores[i]) > 0.5:
            print(whichColumns[i])        

# Calculate the AUC-ROC score
roc_auc = roc_auc_score(allytestfold, allyscores)

# Print or save the AUC-ROC score
print(f'AUC-ROC Score: {roc_auc:.2f}')

# with open("/bioProjectIds/kFoldTsvs/y_scores.tsv", "w") as writeFile:
#     for s in y_scores:
#         writeFile.write(f"{s}\t")

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(allytestfold, allyscores)

y_pred = rf.predict(x_test_fold)

# Compute confusion matrix
cm = confusion_matrix(y_test_fold, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])

# Plot confusion matrix
plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.savefig('/bioProjectIds/sex/confusion_matrix_allsub.png')
plt.show()

###We are attempting to find the most imporant ngrams
feature_importances = rf.feature_importances_

# Get the names of the features
feature_names = np.array(ngrams)

# Sort features based on importance
sorted_indices = np.argsort(feature_importances)[::-1]

# Select the top n-grams
numTop = 50
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
plt.savefig('/bioProjectIds/sex/mostRelevantNgrams_allsub.png')
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

with open("/bioProjectIds/sex/ngramFrequencyByCategory.tsv", "w") as writeFile:
    writeFile.write("Importance\tNgram\tFrequency in Sex Columns\tFrequency in Non Sex Columns\n")
    for i, index in enumerate(sorted_indices):
        writeFile.write(f"{i+1}\t{ngrams[index]}\t{raceAverages[index]}\t{nonraceAverages[index]}\n")

#############################################################################
######REMOVING THE TOP X FEATURES WHAT WOULD HAPPEN?????####################
#############################################################################

# Sort features based on importance
# sorted_indices = np.argsort(feature_importances)

# Remove the top X n-grams. Tweak this. It could be the top 50, 100, 150, till it possibly break
#already did 50. So try 100 or so. 
top_ngrams_to_remove = sorted_indices[:numTop]
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

y_scores = rf.predict_proba(x_test_fold)[:, 1]  
precision, recall, _ = precision_recall_curve(y_test_fold, y_scores)
auc_pr = auc(recall, precision)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {auc_pr:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()
plt.savefig(f'/bioProjectIds/sex/precision_recall_curve_{numTop}_removed.png')

y_pred = rf.predict(x_test_fold)

# Compute confusion matrix
cm = confusion_matrix(y_test_fold, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])

# Plot confusion matrix
plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.savefig(f'/bioProjectIds/sex/confusion_matrix_removed_top{numTop}.png')
plt.show()