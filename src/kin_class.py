import numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils import molecular_weight
import freesasa as fs
import gudhi as gd
import persim


class Kinase:
    def __init__(
            self,
            pdb_file: str,
            name: str,
            activity: str,
            pdb_id: str,
            chain: str,
            group: str = None,
            gene: str = None,
            C_alpha_only: bool = False,
            remove_HETATM: bool = True,
            remove_hydrogens: bool = True,
            alpha_square: int = 20,
            max_dim: int = 2,
            pixel_size: float = 1.0,
            birth_range: tuple = (0, 10),
            pers_range: tuple = (0, 10)
        ):
        self.pdb_file = pdb_file
        self.name = name
        self.activity = activity
        self.pdb_id = pdb_id
        self.chain = chain
        self.group = group
        self.gene = gene
        self.C_alpha_only = C_alpha_only
        self.remove_HETATM = remove_HETATM
        self.remove_hydrogens = remove_hydrogens
        self.alpha_square = alpha_square
        self.max_dim = max_dim
        self.pixel_size = pixel_size
        self.birth_range = birth_range
        self.pers_range = pers_range

        # Initialize attributes
        self.persistence_diagrams = None
        self.persistence_images = {}

        self._parse_pdb()
        self._compute_sasa()
        self._compute_persistence()
        self._compute_persistence_images()

    def _parse_pdb(self):
        """
        Parses the PDB file to extract coordinates and calculate molecular
        weight. If C_alpha_only is True, only C_alpha atoms are selected. If
        remove_HETATM is True, HETATM residues are skipped. If remove_hydrogens
        is True, hydrogen atoms are skipped (not applicable if C_alpha_only is
        True).
        """
        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure("PDB", self.pdb_file)
        except Exception as e:
            print(f"File {self.pdb_file} could not be parsed with exception:")
            print(e)
            print("Exiting...")
            return None
        coordinates = []
        seq = ""

        for model in structure:
            for chain in model:
                if chain.id != self.chain:
                    continue

                for residue in chain:
                    # Skip HETATM
                    if self.remove_HETATM and residue.id[0] != " ":
                        continue

                    # Select C_alpha only
                    if self.C_alpha_only and residue.has_id("CA"):
                        coordinates.append(residue["CA"].get_coord())
                        seq += "C"

                    # Select all atoms
                    elif not self.C_alpha_only:
                        # Skip hydrogens
                        if self.remove_hydrogens:
                            for atom in residue:
                                if atom.element != "H":
                                    coordinates.append(atom.get_coord())
                                    seq += residue.get_resname()

                        else:
                            for atom in residue:
                                coordinates.append(atom.get_coord())
                                seq += residue.get_resname()

        if len(coordinates) == 0:
            raise ValueError(
                f"No valid atoms found for {self.pdb_file}"
            )

        # Calculate molecular weight
        if not self.C_alpha_only:
            self.seq = seq 
            self.mw = molecular_weight(seq, seq_type="protein")
        else:
            self.seq = None
            self.mw = len(seq) * 12.0107
        self.coordinates = np.array(coordinates)

    def _compute_sasa(self):
        """
        Computes the Solvent Accessible Surface Area (SASA) of the kinase
        structure.
        """
        structure = fs.Structure(self.pdb_file)
        sasa = fs.calc(structure).totalArea()
        normalized_sasa = sasa / self.mw
        self.sasa = sasa
        self.normalized_sasa = normalized_sasa

    def _compute_persistence(self):
        """
        Computes the persistent homology using Alpha complex from GUDHI.
        """
        try:
            # Create Alpha complex
            alpha_complex = gd.AlphaComplex(points=self.coordinates)

            # Create simplex tree
            simplex_tree = alpha_complex.create_simplex_tree(
                max_alpha_square=self.alpha_square
            )

            # Compute persistence
            persistence = simplex_tree.persistence(
                homology_coeff_field=2,  # Z/2Z coefficients
                min_persistence=0
            )

            # Store persistence diagrams by dimension
            self.persistence_diagrams = {}
            for dim in range(self.max_dim + 1):
                dim_persistence = simplex_tree.persistence_intervals_in_dimension(dim)
                if len(dim_persistence) > 0:
                    self.persistence_diagrams[dim] = dim_persistence
                else:
                    # Create empty diagram if no features in this dimension
                    self.persistence_diagrams[dim] = np.empty((0, 2))

        except Exception as e:
            print(f"Error computing persistence for {self.name}: {e}")
            # Initialize empty diagrams
            self.persistence_diagrams = {}
            for dim in range(self.max_dim + 1):
                self.persistence_diagrams[dim] = np.empty((0, 2))

    def _compute_persistence_images(self):
        """
        Computes persistence images for each homological dimension.
        """
        try:
            for dim, diagram in self.persistence_diagrams.items():
                if len(diagram) > 0:
                    # Filter out infinite persistence points and convert to
                    # birth-persistence format
                    finite_diagram = []
                    for point in diagram:
                        birth, death = point[0], point[1]
                        # Skip infinite persistence points
                        if not np.isinf(death):
                            persistence = death - birth
                            # Only include points with positive persistence
                            if persistence > 0:
                                finite_diagram.append([birth, persistence])

                    if len(finite_diagram) > 0:
                        finite_diagram = np.array(finite_diagram)

                        # Create persistence image using the new API
                        pim = persim.PersistenceImager(
                            pixel_size=self.pixel_size,
                            birth_range=self.birth_range,
                            pers_range=self.pers_range
                        )

                        # Transform diagram to persistence image
                        # Note: expects list of diagrams
                        img = pim.transform([finite_diagram])
                        # Get first (and only) image
                        self.persistence_images[f"H{dim}"] = img[0]
                    else:
                        # Create zero image if no finite features
                        pixels_x = int(
                            (self.birth_range[1] - self.birth_range[0])
                            / self.pixel_size
                        )
                        pixels_y = int(
                            (self.pers_range[1] - self.pers_range[0])
                            / self.pixel_size
                        )
                        self.persistence_images[f"H{dim}"] = np.zeros(
                            (pixels_y, pixels_x)
                        )
                else:
                    # Create zero image for empty diagrams
                    pixels_x = int(
                        (self.birth_range[1] - self.birth_range[0])
                        / self.pixel_size
                    )
                    pixels_y = int(
                        (self.pers_range[1] - self.pers_range[0])
                        / self.pixel_size
                    )
                    self.persistence_images[f"H{dim}"] = np.zeros(
                        (pixels_y, pixels_x)
                    )

        except Exception as e:
            print(f"Error computing persistence images for {self.name}: {e}")
            # Initialize empty images
            pixels_x = int(
                (self.birth_range[1] - self.birth_range[0])
                / self.pixel_size
            )
            pixels_y = int(
                (self.pers_range[1] - self.pers_range[0])
                / self.pixel_size
            )
            for dim in range(self.max_dim + 1):
                self.persistence_images[f"H{dim}"] = np.zeros(
                    (pixels_y, pixels_x)
                )

    def get_feature_vector(self, flatten=True, include_dims=None):
        """
        Returns the persistence images as feature vectors for machine learning.

        Parameters:
        -----------
        flatten : bool
            If True, flattens each persistence image to 1D
        include_dims : list
            List of homological dimensions to include (e.g., [0, 1])
            If None, includes all computed dimensions

        Returns:
        --------
        dict or np.array
            Dictionary of persistence images or concatenated feature vector
        """
        if include_dims is None:
            include_dims = list(range(self.max_dim + 1))

        features = {}
        for dim in include_dims:
            key = f"H{dim}"
            if key in self.persistence_images:
                img = self.persistence_images[key]
                if flatten:
                    features[key] = img.flatten()
                else:
                    features[key] = img

        if flatten and len(features) > 1:
            # Concatenate all flattened images
            return np.concatenate(list(features.values()))
        elif flatten and len(features) == 1:
            return list(features.values())[0]
        else:
            return features

    def get_metadata(self):
        """
        Returns metadata about the kinase for analysis.
        """
        return {
            'name': self.name,
            'activity': self.activity,
            'pdb_id': self.pdb_id,
            'chain': self.chain,
            'gene': self.gene,
            'group': self.group,
            'n_atoms': len(self.coordinates),
            'sasa': self.sasa,
            'normalized_sasa': self.normalized_sasa,
            'molecular_weight': self.mw
        }
