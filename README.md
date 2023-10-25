# Race Metadata Project README

This README provides a step-by-step guide on how to download data from the NCBI Bioprojects database and use Active Learning to evaluate whether a BioSample contains race information. 

## Usage

1. **Download Bioproject Data**

   Download `bioproject_result.txt` from the [NCBI Bioprojects website](https://www.ncbi.nlm.nih.gov/bioproject/) using the Transcriptome and Homo sapiens filters. (Send to --> file options)
2. **Run the docker**

    ```bash
    sh run_docker.sh
    ```

## Within exec_analysis.py (run by Docker)

1. **Generate a map of bioProjects with associated bioSamples**

   Run `bioProjToBioSamp2.py` to generate `bioProjectToBioSample.json`, which maps Bioprojects to Biosample data.

2. **Generate the list of BioSamples to download**

   Use `getBiosampleIdsFromJson.py` to extract Biosample IDs and create `bioSamples/list_biosamples.txt`.

3. **Split Biosample List**
    
    Split list_biosamples.txt into smaller files with a maximum of 1000 entries each. This creates the to_download folder.
    ```bash
    split -l 1000 -d list_biosamples.txt to_download/to_download
    ```
    (Note: The code for this step comes from the metatools_ncbi repository.)

4. **Download Biosamples**

    Run the downloadBiosample.sh script to initiate the download process.
    ```bash
    sh downloadBiosample.sh
    ```
5. **Return to Main Directory**

    Return to the main project directory:

    ```bash
    cd ..
    ```
6. Generate a Sampled List

    Run the following command to generate a new list of Bioprojects with samples that can be randomly sampled from:

    ```bash
    python3 Active_Learning/bioprojectsWithSamples.py
    ```