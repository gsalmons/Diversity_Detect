import json
with open("/bioProjectIds/bioProjectToBioSample.json", "r") as jFile:
    allProj = json.loads(jFile.read())

samplesSet = set()
for proj, samples in allProj.items():
    if samples == []:
        continue
    for sample in samples:
        samplesSet.add(sample)

with open("/bioSamples/list_biosamples.txt", "w") as writeFile:
    for sam in samplesSet:
        writeFile.write(sam + "\n")