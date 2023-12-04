# Race Metadata Project README

This README provides a step-by-step guide on how to download data from the NCBI Bioprojects database and use Random Forest and Active Learning to evaluate whether a BioSample contains race information. 

## Usage

1. **Download Bioproject Data**

   Download `bioproject_result.txt` from the [NCBI Bioprojects website](https://www.ncbi.nlm.nih.gov/bioproject/) using the Transcriptome and Homo sapiens filters. (Send to --> file options)
2. **Run the docker**

    ```bash
    sh run_docker.sh
    ```

## Within exec_analysis.py (run by Docker)

1. **Generate a map of bioProjects with associated bioSamples**

   Run `bioProjToBioSamp.py` to generate `bioProjectToBioSample.json`, which maps Bioprojects to Biosample data.

2. **Generate a list of bioSample IDs that need to be downloaded**

   Run `getBioSampleFromJson.py` to generate `list_biosamples.txt`, which contains all of the BioSample IDs that will have metadata downloaded.

3. **Randomly select which BioProjects will be annotated manually**

   Run `getRandomIds.py` to generate `intialRandomSample.tsv` and `unlabeledProject.tsv`, where the first file has the 2000 bioprojects that was randomly selected and the second file contains project IDs that were not selected. 

4. **Identify which Biosamples would be downloaded for labeling**

   Run `IdentifyBiosamplesToLabel.py` to generate `list_randomInit_biosamples.txt` which contains all of the BioSample IDs for our randomly selected bioprojects. 

5. **Download the biosamples and standardize their file names**

   Start with the metatools download to get the information for the biosamples that were identified. After running this, run `download.py` to get the biosample IDs into `allJsons`. If there are still biosamples left to download, we keep running and those biosample IDs are put in the `keepLoading.txt`. Run `retitling.py` to standardize the naming of the downloaded files in the `allJsons` directory. 

6. **Get the unique values for each column of a BioProject across all of its Biosamples**

   Run `getColumnsForInitial.py` and `getColumnsForOther.py`to get the unique values for each column of bioprojects. The first file does this for the randomly selected 2000 bioprojects while the second one works with the non-selected bioprojects. `oracleColumns` directory has individual tsv files for each bioproject. 

7. **Generate a file that contains every unique tri-grams present in labeled and unlabeled BioProject metadata**

   Run `uniqueTabDictionary.py` to generate `uniquePhrases.tsv` which contains all of the unique tri-grams across all bioproject metadata. Run `createMasterInputFile.py`to create a file that could be used as an input for machine learning. This works with the bioprojects that were randomly selected and hand labeled, while `createBetterMasterFile.py` takes out the tri-grams that occur less than 2 times. `masterInputOracle.tsv` contains all of the tri-grams from both labeled and unlabeled bioprojects. The columns are the tri-grams and the rows are the column names from the labeled bioprojects. 

8. **Uses Random Forest on individual columns of bioprojects to predict outcomes**

   Run `ourkfold.py` to generate `AUC-ROC Score`, `Confusion Matrix`, `Precision recall curves` and a graph showing the `top feature importances` for our race n-grams. Run `sexKfold.py` to generate the graphs and visualizations for our sex/gender n-grams. Run `tumor_stage_kfold.py` to generate the graphs and visualizations for our tumor stage n-grams. 