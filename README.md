# Final project for the 'Algorithms in Structural Bioinformatics'

## Scope
This project was a two person assignment. We chose to train a classical Machine Learning model to classify kinase structures to active or inactive conformations, using no information other than the structure itself, unlike previous methods. To that end we generated geometric and topological features from the kinase structures and used those for training and predicting.

## Pipeline
### Python environment
- `python3 -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`

### The Data
- Run the notebook [1_download_pdbs.ipynb](notebooks/1_download_pdbs.ipynb)
- Run `./scripts/batch_download.sh -f metadata/pdb_ids.txt -o data_raw -c 2>&1 | tee cif_download.log`
- Run `grep "Failed" logs/cif_download.log` to inspect for issues.
- Download manually any files that failed for some reason.
- Run the notebook [2_clean_pdbs.ipynb](notebooks/2_clean_pdbs.ipynb)
- Run `grep "ERROR" logs/create_pdbs.log` to inspect for issues. If only a few files failed we can ignore them, we will have enough data.