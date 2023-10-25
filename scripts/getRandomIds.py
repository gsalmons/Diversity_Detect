import random
import json
#load in all the bioProject IDs
idsWithSamples = []
with open("/bioProjectIds/bioProjectToBioSample.json", "r") as readFile:
    allProjects = json.loads(readFile.read())
    for project, samples in allProjects.items():
        if samples != []:
            idsWithSamples.append(project)

#Pick 100 of them at random
random.seed(0)
random_values = random.sample(idsWithSamples, 2000)
print(random_values)
#Save randomly generated IDs to a file
with open("/bioProjectIds/initialRandomSample.tsv", "w") as writeFile:
    for id in random_values:
        writeFile.write(id + "\n")

unlabeled = set(idsWithSamples) - set(random_values)
with open("/bioProjectIds/unlabeledProjects.tsv", "w") as writeFile:
    for id in unlabeled:
        writeFile.write(id + "\n")