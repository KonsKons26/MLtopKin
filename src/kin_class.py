import numpy as np
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Polypeptide import is_aa
from Bio.Data.PDBData import protein_letters_3to1_extended
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
            domain: tuple = None,
            C_alpha_only: bool = False,
            remove_HETATM: bool = True,
            remove_hydrogens: bool = True,
            alpha_square: float = np.inf,
            max_dim: int = 2,
            pixel_size: float = 1.0,
            birth_range: tuple = (0, 10),
            pers_range: tuple = (0, 10),
            normalize: bool = True
        ):
        self.pdb_file = pdb_file
        self.name = name
        self.activity = activity
        self.pdb_id = pdb_id
        self.chain = chain
        self.group = group
        self.gene = gene
        self.domain = domain
        self.C_alpha_only = C_alpha_only
        self.remove_HETATM = remove_HETATM
        self.remove_hydrogens = remove_hydrogens
        self.alpha_square = alpha_square
        self.max_dim = max_dim
        self.pixel_size = pixel_size
        self.birth_range = birth_range
        self.pers_range = pers_range
        self.normalize = normalize

        # Initialize attributes
        self.persistence_diagrams = None
        self.persistence_images = {}

        self._parse_pdb()
        self._compute_sasa()
        self._compute_persistence()
        self._compute_persistence_images()

    def _parse_pdb(self):
        """
        Parses the PDB/mmCIF file to extract coordinates and calculate
        molecular weight. Only the first model is used. If C_alpha_only is
        True, only C_alpha atoms are selected. If remove_HETATM is True,
        HETATM residues are skipped, except modified amino acids. If domain is
        given, only residues numbered within (start, end) are kept. If
        remove_hydrogens is True, hydrogen atoms are skipped (not applicable
        if C_alpha_only is True).
        """
        if self.pdb_file.endswith(".cif"):
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)
        structure = parser.get_structure("PDB", self.pdb_file)
        coordinates = []
        sasa_atoms = []
        seq = ""
        n_residues = 0

        # Only the first model (NMR/EM files can hold many models)
        for chain in structure[0]:
            if chain.id != self.chain:
                continue

            for residue in chain:
                # Skip HETATM (ligands, waters, ions) but keep modified
                # amino acids such as phosphorylated residues
                if (
                    self.remove_HETATM
                    and residue.id[0] != " "
                    and not is_aa(residue)
                ):
                    continue

                # Keep only the kinase domain if its boundaries are given
                if self.domain is not None and not (
                    self.domain[0] <= residue.id[1] <= self.domain[1]
                ):
                    continue

                resname = residue.get_resname()
                n_residues += 1

                # Heavy atoms of the selected residues, used for the SASA
                for atom in residue:
                    if atom.element != "H":
                        # PDB style 4 character atom name, which FreeSASA
                        # needs to look up the atomic radius
                        name = atom.get_name()
                        if len(atom.element) == 1:
                            name = " " + name
                        sasa_atoms.append((
                            name.ljust(4), resname, str(residue.id[1]),
                            chain.id, *(float(c) for c in atom.get_coord())
                        ))

                # Select C_alpha only
                if self.C_alpha_only and residue.has_id("CA"):
                    coordinates.append(residue["CA"].get_coord())
                    seq += "C"

                # Select all atoms
                elif not self.C_alpha_only:
                    # One letter per residue (modified residues are mapped
                    # to their parent amino acid)
                    seq += protein_letters_3to1_extended.get(resname, "")
                    for atom in residue:
                        # Skip hydrogens
                        if self.remove_hydrogens and atom.element == "H":
                            continue
                        coordinates.append(atom.get_coord())

        if len(coordinates) == 0:
            raise ValueError(
                f"No valid atoms found for {self.pdb_file}"
            )

        # A domain that is mostly missing means that the residue numbers of
        # the file do not match the given domain boundaries
        if self.domain is not None:
            domain_length = self.domain[1] - self.domain[0] + 1
            if n_residues < 0.5 * domain_length:
                raise ValueError(
                    f"Only {n_residues} of the {domain_length} domain "
                    f"residues found for {self.pdb_file}"
                )

        # Calculate molecular weight
        if not self.C_alpha_only:
            self.seq = seq
            self.mw = molecular_weight(seq, seq_type="protein")
        else:
            self.seq = None
            self.mw = len(seq) * 12.0107
        self._sasa_atoms = sasa_atoms
        self.coordinates = np.array(coordinates)

    def _compute_sasa(self):
        """
        Computes the Solvent Accessible Surface Area (SASA) of the selected
        residues (all heavy atoms).
        """
        # Radii of atoms unknown to FreeSASA (modified residues) are guessed
        # from the element, do not print a warning for each of them
        fs.setVerbosity(fs.nowarnings)
        structure = fs.Structure()
        for atom in self._sasa_atoms:
            structure.addAtom(*atom)
        del self._sasa_atoms
        sasa = fs.calc(structure).totalArea()
        normalized_sasa = sasa / self.mw
        self.sasa = sasa
        self.normalized_sasa = normalized_sasa

    def _compute_persistence(self):
        """
        Computes the persistent homology using Alpha complex from GUDHI.
        Filtration values are squared radii.
        """
        # Create Alpha complex
        alpha_complex = gd.AlphaComplex(points=self.coordinates)

        # Create simplex tree
        simplex_tree = alpha_complex.create_simplex_tree(
            max_alpha_square=self.alpha_square
        )

        # Compute persistence
        simplex_tree.persistence(
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

    def _compute_persistence_images(self):
        """
        Computes persistence images for each homological dimension.
        """
        for dim, diagram in self.persistence_diagrams.items():
            # Drop infinite persistence points (with the full filtration this
            # is only the single connected component in H0)
            finite_diagram = diagram[np.isfinite(diagram[:, 1])]

            # All H0 classes of an alpha complex are born at 0, so a single
            # birth column is enough (1D profile of the deaths)
            if dim == 0:
                birth_range = (0, self.pixel_size)
            else:
                birth_range = self.birth_range

            pim = persim.PersistenceImager(
                pixel_size=self.pixel_size,
                birth_range=birth_range,
                pers_range=self.pers_range
            )

            # Note: expects a list of diagrams in birth-death coordinates
            # (skew=True converts them to birth-persistence)
            img = pim.transform([finite_diagram], skew=True)[0]

            # Images are sums over all points, so divide by the number of
            # points to remove the dependence on the size of the structure
            if self.normalize:
                img = img / len(self.coordinates)

            self.persistence_images[f"H{dim}"] = img

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
