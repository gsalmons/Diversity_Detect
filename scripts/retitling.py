"""Objective: Standardize the naming of files downloaded. Most are downloaded with the
BioSample as the name but some are not. This ensures that the name for each file will be the ID. 
Inputs: No args. Contents of bioSamples/allJsons directory
Outputs: Renaming of files in bioSamples/allJsons directory
"""


import os
import json

# Change this directory as needed
json_directory = "/bioSamples/allJsons/"

# Iterate over each JSON file in the directory
for filename in os.listdir(json_directory):
    if filename.endswith("_attributes.json"):
        continue
    if filename.endswith(".json"):
        file_path = os.path.join(json_directory, filename)

        # Read the JSON content
        with open(file_path, "r") as json_file:
            data = json.load(json_file)

        # Extract the value of the "id" field
        if "id" in data:
            new_name = f"{data['id']}.json"
            
            # Rename the file
            new_path = os.path.join(json_directory, new_name)
            os.rename(file_path, new_path)

            print(f"Renamed '{filename}' to '{new_name}'")
        else:
            print(f"Skipping '{filename}' as it does not contain 'id' field.")