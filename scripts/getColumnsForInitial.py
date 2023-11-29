import json

bioProjects = []
allProj = dict()
cellLineTypes = list()
columnsToIgnore = ["sra", "geo", "accession", "id", "biosample"]
unknownVariants = {"", "Not applicable", "not applicable", "Not Determined", "none", "None", "Not available", "Not Available", "Not determined", "not determined", "Not Applicable", "not collected", "Not Collected", "Not collected", "missing", "not_applicable", ".", "--", "-", "unknown", "Unknown", "not_available", "not available", "UNKNOWN", "n/a", "N/A", "n.a.", "N.A.", "na.", "N.a", "Missing", "na", "NA", "NA."}

with open("/bioProjectIds/bioProjectToBioSample.json", "r") as jFile:
    allProj = json.loads(jFile.read())
print("loaded in alright!")

with open("/bioProjectIds/initialRandomSample.tsv", "r") as readFile:
    for line in readFile:
        line = line.rstrip()
        bioProjects.append(line)
print("Loaded the second in alright!")
numSampleMasterList = []
raceEthnicitySize = []
raceproject = []
for projectId in bioProjects:
    with open(f"/bioProjectIds/oracleColumns/{projectId}.tsv", "w") as writeFile:
        sampleIds = allProj[projectId]
        numSamples = len(sampleIds)
        numSampleMasterList.append(numSamples)
        projectInfo = dict()
        counter = 0
        for id in sampleIds:
            try:
                with open(f"/bioSamples/jsons/{id}.json", "r") as readFile:
                    colAndVal = json.loads(readFile.read())
                if counter == 0:
                    for column, values in colAndVal.items():
                        if column == "race" or column == "ethnicity":
                            raceproject.append(projectId)
                            raceEthnicitySize.append(numSamples)
                            print(f"Race info in {id}, {projectId}")
                        projectInfo[column] = {values}
                    counter += 1
                else:
                    for column, values in colAndVal.items():
                        if column.lower() == "cell_line":
                            cellLineTypes.append(values)
                        if column not in projectInfo:
                            projectInfo[column] = {values}
                        else:
                            projectInfo[column].add(values)
            except Exception as e:
                print(f"Problem reading in {id}, {projectId}: {e}")
        firstTime = True
        for info in sorted(list(projectInfo.keys())):
            #Get rid of columns that we know do not contain useful info for the model
            if info in columnsToIgnore:
                continue
            #Get rid of columns that have unique values for each biosample if there is more than one biosample
            #UNLESS that column is description
            if numSamples > 1 and len(projectInfo[info]) == numSamples and info != "description" and info != "title":
                if info == "race" or info == "ethnicity":
                    print("problemo!", projectId)
                # print(f"not including, {info}, {projectInfo[info]}")
                continue
            #Get rid of columns that have float numbers
            if type(next(iter(projectInfo[info]))) == float: 
                # print(f"not including, {info}, {projectInfo[info]}")
                continue
            #Get rid of columns that have just one "unknown" value
            if len(projectInfo[info]) == 1 and str(projectInfo[info]) in unknownVariants:
                # print(f"not including, {info}, {projectInfo[info]}")
                continue
            #Get rid of columns that multiple "unknown"-esque values
            allUnknownVar = True
            for v in projectInfo[info]:
                if allUnknownVar == False:
                    continue
                if v not in unknownVariants:
                    allUnknownVar = False
            if allUnknownVar:
                continue
            if info == "title":
                remove = True
                for sampleValue in projectInfo[info]:
                    if not sampleValue.startswith("GEO accession"):
                        remove = False
                if remove:
                    print(projectInfo[info])
                    continue
            if firstTime:
                writeFile.write(info + "\t")
                firstTime = False
            else:
                writeFile.write("\n" + info + "\t")
            for value in sorted(list(projectInfo[info])):
                writeFile.write(value + "\t")
cellLineTypes = list(set(cellLineTypes))
with open("/bioProjectIds/cellLines.tsv", "w") as writeFile:
    for values in cellLineTypes:
        if type(values) == set:
            for value in values:
                writeFile.write(value + "\t")
            writeFile.write("\n")
        else:
            writeFile.write(values + "\n")
with open("/bioProjectIds/sizes.tsv", "w") as writeFile:
    numLargeEnough = 0
    numTooSmall = 0
    raceLarge = 0
    raceSmall = 0
    for size in numSampleMasterList:
        if size > 4:
            numLargeEnough += 1
        else:
            numTooSmall += 1
    for s in raceEthnicitySize:
        if s > 4:
            raceLarge += 1
        else:
            raceSmall += 1
    writeFile.write(f"numLarge: {numLargeEnough}\tnumTooSmall: {numTooSmall}\nraceLarge: {raceLarge}\traceSmall: {raceSmall}\n")
    for i, s in enumerate(raceEthnicitySize):
        writeFile.write(str(raceproject[i]) + "\t"+ str(s) + "\n")