"""Objective: Load metadata for the BioSample of BioProjects not selected for manual curation in a separate folder. 
Inputs: Argument 1: the directory "allJsons" or "jsons" required to evaluate which projects have already been downloaded. 
list_biosamples.txt. list_randomInit_biosamples.txt. 
Outputs: keepLoading.txt which contains the BioSample IDs that will not be hand labeled. 
"""
import os
import re
import sys

directory = sys.argv[1]
ids = set() 

if directory == "allJsons":
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
            
    for current_file in os.listdir(f'/bioSamples/{directory}'):
        Idnumber = current_file.split("/")[-1]
        Idnumber = re.sub(".json", "", Idnumber)
        alreadyGot.add(Idnumber)
    ids = ids - alreadyGot
    print(len(ids))
else:
    filePath = "/bioSamples/list_randomInit_biosamples.txt"
    with open(filePath, "r") as readFile: 
            for line in readFile:
                line = line.rstrip()
                ids.add(line)
    print(len(ids))

    alreadyGot = set()
    for current_file in os.listdir(f'/bioSamples/{directory}'):
        Idnumber = current_file.split("/")[-1]
        Idnumber = re.sub(".json", "", Idnumber)
        alreadyGot.add(Idnumber)
    ids = ids - alreadyGot

with open("/bioSamples/keepLoading.txt", "w") as writeFile:
    for id in ids:
        writeFile.write(id + "\n")