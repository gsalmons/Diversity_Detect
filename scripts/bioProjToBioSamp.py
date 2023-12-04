"""Objective: Iteratively create a map of bioProjects to their associated bioSamples
Inputs: gets bioProject Ids from bioproject_result.txt. Will also read in the
bioProjectToBioSample.json if it has been started so no work is repeated and if interrupted it will
continue where it was prior to termination. 
Output: bioProjectToBioSample.json"""


import requests
from xml.etree import ElementTree
import time
import json

bioproject_to_biosample = {}
errorMessages = {}
try:
    with open("/bioProjectIds/errors.json", "r") as errorFile:
        errorMessages = json.load(errorFile.read())
except:
    errorMessages = {}

#Load BioProject IDs from a file
with open("/bioProjectIds/bioproject_result.txt", "r") as idFile:
    bio_project_ids = [line.strip() for line in idFile]

numWithSamples = 0
alreadyFound = {}

try:
    with open("/bioProjectIds/bioProjectToBioSample.json", "r") as jsonFile:
        bioproject_to_biosample = json.loads(jsonFile.read())
except:
    bioproject_to_biosample = {}

foundSet = bioproject_to_biosample.keys()
toFind = set(bio_project_ids) - foundSet - set(errorMessages.keys())

#Set up rate limiting parameters
requests_per_second = 2  #Adjust as per NCBI's rate limits
delay_between_requests = 1 / requests_per_second
numTotal = len(foundSet)
print(len(toFind))
for bioproject_id in toFind:
    numTotal += 1
    print(numTotal)
    #Now, retrieve associated BioSamples for this BioProject
    biosample_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=bioproject&db=biosample&id={bioproject_id}"

    try:
        biosample_response = requests.get(biosample_url)

        if biosample_response.status_code == 200:
            biosample_root = ElementTree.fromstring(biosample_response.text)
            biosample_elements = biosample_root.findall(".//Link/Id")
            biosamples = [biosample_element.text for biosample_element in biosample_elements]

            #Store the mapping in the dictionary
            if biosamples:
                numWithSamples += 1
                bioproject_to_biosample[bioproject_id] = list(set(biosamples))
            else:
                bioproject_to_biosample[bioproject_id] = list()

            #Implement rate limiting
            time.sleep(delay_between_requests)
        
            if (numTotal % 100 == 0):
                with open("/bioProjectIds/bioProjectToBioSample.json", "w") as jfile:
                    jfile.write(json.dumps(bioproject_to_biosample))
                with open("/bioProjectIds/errors.json", "w") as errorFile:
                    errorFile.write(json.dumps(errorMessages))
                print(numTotal)
                break
        else:
            print(f"Error for BioProject ID {bioproject_id}: HTTP {biosample_response.status_code}")
            errorMessages[bioproject_id] = biosample_response.status_code

    except Exception as e:
        print(f"Error for BioProject ID {bioproject_id}: {str(e)}")
        errorMessages[bioproject_id] = biosample_response.status_code

#Now, you have a dictionary where BioProject is the key and BioSample is the value for "Homo sapiens"
print("BioProject to BioSample Mapping for Homo sapiens:")
print(f"Total BioProjects processed: {len(bioproject_to_biosample)}")

with open("/bioProjectIds/errors.json", "w") as errorFile:
    errorFile.write(json.dumps(errorMessages))

with open("/bioProjectIds/bioProjectToBioSample.json", "w") as jfile:
    jfile.write(json.dumps(bioproject_to_biosample))