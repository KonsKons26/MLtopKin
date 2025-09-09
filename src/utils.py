import os
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.kin_class import Kinase


def prep_kinase_data_list(
        in_path,
        df,
        gene_col="Gene",
        group_col="Gene_Group",
        pdb_col="PDB",
        validate_files=True
    ):
    """
    Prepares a list of kinase data dictionaries from directory structure and
    metadata DataFrame.

    Expected directory structure:
    in_path/
    ├── gene1/
    │   ├── active/
    │   │   ├── 1ABC_chainA.pdb
    │   │   └── 2DEF_chainB.pdb
    │   └── inactive/
    │       └── 3GHI_chainA.pdb
    └── gene2/
        ├── active/
        └── inactive/

    Parameters:
    -----------
    in_path : str
        Root directory containing gene/activity subdirectories with PDB files
    df : pandas.DataFrame
        DataFrame containing metadata with columns for genes, gene
        groups/families, and PDB IDs
    gene_col : str, default "Gene"
        Column name in df containing individual gene names
    group_col : str, default "Gene_Group"
        Column name in df containing gene group/family information
    pdb_col : str, default "PDB"
        Column name in df containing PDB IDs (should match filename patterns)
    validate_files : bool, default True
        Whether to validate that PDB files exist and are readable

    Returns:
    --------
    list of dict
        Each dict contains: pdb_file, name, activity, pdb_id, chain, gene, group
    """

    kinase_data_list = []

    # Validate input path
    if not os.path.isdir(in_path):
        raise FileNotFoundError(f"Input path not found: {in_path}")

    # Convert to Path object for easier manipulation
    base_path = Path(in_path)

    print(f"Scanning directory: {in_path}")
    print(f"DataFrame shape: {df.shape}")

    # Statistics tracking
    stats = {
        'total_files': 0,
        'processed': 0,
        'skipped_no_gene': 0,
        'skipped_no_activity': 0,
        'skipped_parse_error': 0,
        'skipped_not_in_df': 0,
        'skipped_file_error': 0
    }

    # Walk through directory structure
    for pdb_file in base_path.rglob("*.pdb"):
        stats['total_files'] += 1

        try:
            # Parse directory structure: gene/activity/file.pdb
            parts = pdb_file.parts
            base_idx = parts.index(base_path.name)

            # Extract gene and activity from path
            if len(parts) < base_idx + 3:
                print(
                    f"Warning: Insufficient path depth for {pdb_file}."
                    f" Expected: gene/activity/file.pdb"
                )
                stats['skipped_no_gene'] += 1
                continue

            gene = parts[base_idx + 1]
            activity = parts[base_idx + 2]

            # Validate activity label
            if activity.lower() not in ['active', 'inactive']:
                print(
                    f"Warning: Unrecognized activity '{activity}' in "
                    f"{pdb_file}. Expected 'active' or 'inactive'"
                )
                stats['skipped_no_activity'] += 1
                continue

            # Parse filename to extract PDB ID and chain
            filename = pdb_file.stem  # filename without extension

            # Handle different filename patterns
            if "_chain" in filename:
                pdb_id_base = filename.split("_")[0]
                chain_part = filename.split("_chain")[-1]
                chain = chain_part if chain_part else "A"
            elif "_" in filename and len(filename.split("_")) >= 2:
                parts_name = filename.split("_")
                pdb_id_base = parts_name[0]
                # Try to extract chain from second part
                chain = parts_name[1] if len(parts_name[1]) == 1 else "A"
            else:
                pdb_id_base = filename[:4] if len(filename) >= 4 else filename
                chain = "A"  # Default chain

            # Create lookup key for DataFrame
            lookup_key = f"{pdb_id_base}{chain}" if chain != "A" else pdb_id_base

            # Alternative lookup strategies
            lookup_keys = [
                lookup_key,
                pdb_id_base,
                f"{pdb_id_base}{chain}",
                f"{pdb_id_base}_{chain}",
                pdb_id_base.upper(),
                f"{pdb_id_base.upper()}{chain}"
            ]

            # Find matching row in DataFrame
            gene_name = None
            group = None
            for key in lookup_keys:
                matching_rows = df[df[pdb_col] == key]
                if not matching_rows.empty:
                    gene_name = matching_rows[gene_col].iloc[0]
                    group = matching_rows[group_col].iloc[0]
                    break

            if gene_name is None or group is None:
                print(
                    f"Warning: PDB '{lookup_key}' (alternatives: "
                    f"{lookup_keys[:3]}) not found in DataFrame."
                    f" Skipping {pdb_file}"
                )
                stats['skipped_not_in_df'] += 1
                continue

            # Validate file if requested
            if validate_files:
                try:
                    if not pdb_file.exists() or pdb_file.stat().st_size == 0:
                        print(
                            f"Warning: File {pdb_file} is empty or unreadable"
                        )
                        stats['skipped_file_error'] += 1
                        continue
                except OSError as e:
                    print(f"Warning: Cannot access file {pdb_file}: {e}")
                    stats['skipped_file_error'] += 1
                    continue

            # Create kinase data entry
            kinase_entry = {
                'pdb_file': str(pdb_file),
                'name': f"{gene}_{pdb_id_base}_{chain}",
                'activity': activity.lower(),
                'pdb_id': pdb_id_base,
                'chain': chain,
                'gene': gene_name,    # Individual gene name from DataFrame
                'group': group        # Gene group/family from DataFrame
            }

            kinase_data_list.append(kinase_entry)
            stats['processed'] += 1

        except Exception as e:
            print(f"Error processing {pdb_file}: {e}")
            stats['skipped_parse_error'] += 1
            continue

    # Print summary statistics
    print(f"\n=== Processing Summary ===")
    print(f"Total PDB files found: {stats['total_files']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Skipped - insufficient path depth: {stats['skipped_no_gene']}")
    print(f"Skipped - invalid activity label: {stats['skipped_no_activity']}")
    print(f"Skipped - parsing errors: {stats['skipped_parse_error']}")
    print(f"Skipped - not found in DataFrame: {stats['skipped_not_in_df']}")
    print(f"Skipped - file access errors: {stats['skipped_file_error']}")

    # Activity distribution
    if kinase_data_list:
        activity_counts = {}
        for entry in kinase_data_list:
            activity = entry['activity']
            activity_counts[activity] = activity_counts.get(activity, 0) + 1

        print(f"\n=== Activity Distribution ===")
        for activity, count in activity_counts.items():
            print(f"{activity.capitalize()}: {count}")

    return kinase_data_list


def create_kinase_dataset(kinase_data_list, max_workers=None, **kwargs):
    """
    Creates a dataset of Kinase objects from a list of kinase data using
    parallel processing.

    Parameters:
    -----------
    kinase_data_list : list of dict
        Each dict should contain: pdb_file, name, activity, pdb_id, chain
    max_workers : int, optional
        Maximum number of worker threads. If None, uses default from
        ThreadPoolExecutor
    **kwargs : additional parameters to pass to Kinase constructor

    Returns:
    --------
    list of Kinase objects
    """

    def create_single_kinase(data):
        """Helper function to create a single kinase with error handling."""
        try:
            return Kinase(**data, **kwargs)
        except Exception as e:
            print(
                f"Failed to create kinase for {data.get('name', 'unknown')}: {e}"
            )
            return None

    kinases = []

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_data = {
            executor.submit(create_single_kinase, data): data 
            for data in kinase_data_list
        }

        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_data), 
                          total=len(kinase_data_list), 
                          desc="Processing kinases"):
            kinase = future.result()
            if kinase is not None:
                kinases.append(kinase)

    return kinases


def extract_features_for_ml(kinases, include_dims=[0, 1]):
    """
    Extracts features and labels for machine learning from a list of Kinase
    objects.

    Parameters:
    -----------
    kinases : list of Kinase objects
    include_dims : list of homological dimensions to include

    Returns:
    --------
    X : np.array of shape (n_samples, n_features)
    y : np.array of shape (n_samples,) - binary labels (1 for active, 0 for
    inactive)
    metadata : list of dict containing kinase metadata
    """
    X = []
    y = []
    metadata = []

    for kinase in kinases:
        # Get flattened feature vector
        features = kinase.get_feature_vector(
            flatten=True, include_dims=include_dims
        )
        X.append(features)

        # Convert activity to binary label
        label = 1 if kinase.activity.lower() == 'active' else 0
        y.append(label)

        # Store metadata
        metadata.append(kinase.get_metadata())

    return np.array(X), np.array(y), metadata
