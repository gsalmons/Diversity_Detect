import json

bioProjects = []
allProj = dict()
cellLineTypes = list()
columnsToIgnore = ["sra", "geo", "accession", "id", "biosample"]

with open("/bioProjectIds/bioProjectToBioSample.json", "r") as jFile:
    allProj = json.loads(jFile.read())
print("loaded in alright!")

with open("/bioProjectIds/unlabeledProjects.tsv", "r") as readFile:
    for line in readFile:
        line = line.rstrip()
        bioProjects.append(line)
print("Loaded the second in alright!")

for projectId in bioProjects:
    with open(f"/bioProjectIds/unlabeledColumns/{projectId}.tsv", "w") as writeFile:
        sampleIds = allProj[projectId]
        numSamples = len(sampleIds)
        projectInfo = dict()
        counter = 0
        for id in sampleIds:
            try:
                with open(f"/bioSamples/allJsons/{id}.json", "r") as readFile:
                    colAndVal = json.loads(readFile.read())
                if counter == 0:
                    for column, values in colAndVal.items():
                        if column == "race":
                            print(f"Race info in {id}, {projectId}")
                        projectInfo[column] = {values}
                    counter += 1
                else:
                    for column, values in colAndVal.items():
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
                # print(f"not including, {info}, {projectInfo[info]}")
                if info == "race" or info == "ethnicity":
                    print("problemo!", projectId)
                continue
            #Get rid of columns that have float numbers
            if type(next(iter(projectInfo[info]))) == float:
                # print(f"not including, {info}, {projectInfo[info]}")
                continue
            #Get rid of columns that have "" values
            if projectInfo[info] == {""} or projectInfo[info] == {"missing"} or projectInfo[info] == {"--"} or projectInfo[info] == {"unknown"} or projectInfo[info] == {"-"} or projectInfo[info] == {"UNKNOWN"} or projectInfo[info] == {"Unknown"} or projectInfo[info] == {"n/a"} or projectInfo[info] == {"N/A"} or projectInfo[info] == {"n.a."} or projectInfo[info] == {"N.A."} or projectInfo[info] == {"n.a"} or projectInfo[info] == {"N.a"} or projectInfo[info] == {"Missing"} or projectInfo[info] == {"na"} or projectInfo[info] == {"NA"} or projectInfo[info] == {"NA."} or projectInfo[info] == {"na."} or projectInfo[info] == {"not available"} or projectInfo[info] == {"not_available"} or  projectInfo[info] == {"not_applicable"} or projectInfo[info] == {"not collected"} or projectInfo[info] == {"Not Collected"} or projectInfo[info] == {"Not collected"} or projectInfo[info] == {"Not Applicable"} or projectInfo[info] == {"Not applicable"} or projectInfo[info] == {"Not Determined"} or projectInfo[info] == {"Not determined"} or projectInfo[info] == {"not determined"} or projectInfo[info] == {"Not available"} or projectInfo[info] == {"Not Available"} or projectInfo[info] == {"none"} or projectInfo[info] == {"None"}:
                # print(f"not including, {info}, {projectInfo[info]}")
                continue
            if firstTime:
                writeFile.write(info + "\t")
                firstTime = False
            else:
                writeFile.write("\n" + info + "\t")
            for value in sorted(list(projectInfo[info])):
                writeFile.write(value + "\t")