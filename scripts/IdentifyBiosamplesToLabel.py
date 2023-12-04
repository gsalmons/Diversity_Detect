"""Objective: Identify which bioSamples should be downloaded to be manually curated. 
Inputs: No args. bioProjectToBioSample.json. initialRandomSample.tsv
Outputs: list_randomInit_biosamples.txt
"""
import json

with open('/bioProjectIds/bioProjectToBioSample.json', "r") as jFile:
    allProj = json.loads(jFile.read())
biosamples = []

#This file has the bioProject IDs for our 2000 randomly sampled. 
with open ("/bioProjectIds/initialRandomSample.tsv", "r") as readFile:
    for line in readFile:
        line = line.rstrip()
        for sample in allProj[line]:
            biosamples.append(sample)

with open("/bioSamples/list_randomInit_biosamples.txt", "w") as writeFile:
    for sample in biosamples:
        writeFile.write(sample + "\n")