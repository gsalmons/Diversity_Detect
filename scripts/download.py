import os
import re

ids = set()
# Do with /bioSamples/list_biosamples.txt if for all data
# Do with /bioSamples/list_randomInit_biosamples.txt if for labeled data
filePath = "/bioSamples/list_biosamples.txt"
with open(filePath, "r") as readFile: 
    for line in readFile:
        line = line.rstrip()
        ids.add(line)
print(len(ids))

alreadyGot = set()
with open("/bioSamples/list_randomInit_biosamples.txt", "r") as labeledFile:
    for line in labeledFile:
        line = line.rstrip()
        alreadyGot.add(line)
        
for current_file in os.listdir('/bioSamples/allJsons'):
    Idnumber = current_file.split("/")[-1]
    Idnumber = re.sub(".json", "", Idnumber)
    alreadyGot.add(Idnumber)
ids = ids - alreadyGot
print(len(ids))
with open("/bioSamples/keepLoading.txt", "w") as writeFile:
    for id in ids:
        writeFile.write(id + "\n")