# Final project for the 'Algorithms in Structural Bioinformatics'

## Scope
This project was a two person assignment. We chose to train a classical Machine Learning model to classify kinase structures to active or inactive conformations, using no information other than the structure itself, unlike previous methods. To that end we generated geometric and topological features from the kinase structures and used those for training and predicting.

## Recreation Pipeline
### Python environment
- `python3 -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`

### The Data
- Run the first part of the [1_download_clean_pdbs.ipynb](notebooks/1_download_clean_pdbs.ipynb) notebook (__1. Download pdbs__).
- Run `./scripts/batch_download.sh -f metadata/pdb_ids.txt -o data_raw -c 2>&1 | tee logs/cif_download.log`
- Run `grep "Failed" logs/cif_download.log` to inspect for issues.
- Download manually any files that failed for some reason.
- Run the second part of the [1_download_clean_pdbs.ipynb](notebooks/1_download_clean_pdbs.ipynb) notebook (__2. Clean pdbs__).
- Run `grep "ERROR" logs/create_pdbs.log` to inspect for issues. If only a few files failed we can ignore them, we will have enough data.

### Features
- Run the [2_prep_ml_data](notebooks/2_prep_ml_data.ipynb) notebook to generate the topological and geometric features.
- Then run the [3_eda](notebooks/3_eda.ipynb) notebook to perform a rudimentary Exploratory Data Analysis.

### Repeated Nested Cross Validation
- Run the [4_nestedCV.ipynb](notebooks/4_nestedCV.ipynb) notebook to run the repeated nested cross validation pipeline.