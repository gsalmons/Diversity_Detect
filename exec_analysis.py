#!/usr/bin/env python

import subprocess

def callFunction(script_path, commandType="python"):
    command = [commandType, script_path]
    # Run the external Python script
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    output, error = process.communicate()
    return_code = process.returncode
    if return_code == 0:
        print(f"{script_path} executed successfully.")
        print("Output:")
        print(output)
    else:
        print(f"{script_path} failed with return code {return_code}.")
        print("Error:")
        print(error)

    with open("/bioProjectIds/test.txt", "w") as wFile:
        wFile.write("It worked!")
    return

###Step 1
# callFunction("scripts/bioProjToBioSamp.py")

# ###Step 2
# callFunction("scripts/getBioSampleFromJson.py")

# ###Step 3
# callFunction("scripts/getRandomIds.py")

# ###Step 4
# callFunction("scripts/IdentifyBiosamplesToLabel.py")

###Step 5 
#Download the biosamples for our randomly picked BioProjects
# import os
# for current_file in os.listdir('/bioSamples'):
#     if current_file.startswith('keep'): # change to list_random if first time, all other times do this one
#         print(f" - Downloading RunInfos -- biosamples present in {current_file}")
#         os.system(f"metatools_download biosamples -l /bioSamples/{current_file} /bioSamples/jsons/")

#Download all biosamples
# callFunction("scripts/retitling.py")
# callFunction("scripts/download.py")
###Step 7
###Step 6 
# callFunction("scripts/getColumnsForInitial.py")

# import os
# for current_file in os.listdir('/bioSamples'):
#     if current_file.startswith('keepL'):
#         print(f" - Downloading RunInfos -- biosamples present in {current_file}")
#         os.system(f"metatools_download biosamples -l /bioSamples/{current_file} /bioSamples/allJsons/")

# # # # ### Step ? 
# callFunction("scripts/retitling.py")
# callFunction("scripts/download.py")
# callFunction("scripts/getColumnsForOther.py") # Makes tsv for each bioproject
# callFunction("scripts/uniqueTabDictionary.py")
# callFunction("scripts/createMasterInputFile.py")
# callFunction("scripts/createbettermasterfile.py")
callFunction("scripts/ourkfold.py")