from typing import Dict

embedding_map_fivebead = {
    "ALA": 1,
    "CYS": 2,
    "ASP": 3,
    "GLU": 4,
    "PHE": 5,
    "GLY": 6,
    "HIS": 7,
    "ILE": 8,
    "LYS": 9,
    "LEU": 10,
    "NLE": 10,  # Type Norleucine as Leucine
    "MET": 11,
    "ASN": 12,
    "PRO": 13,
    "GLN": 14,
    "ARG": 15,
    "SER": 16,
    "THR": 17,
    "VAL": 18,
    "TRP": 19,
    "TYR": 20,
    "N": 21,
    "CA": 22,
    "C": 23,
    "O": 24,
}


class CGEmbeddingMap(dict):
    """
    General class for defining embedding maps as Dict
    """

    def __init__(self, embedding_map_dict: Dict[str, int]):
        for k, v in embedding_map_dict.items():
            self[k] = v


class CGEmbeddingMapFiveBead(CGEmbeddingMap):
    """
    Five-bead embedding map defined by:
        - N : backbone nitrogen
        - CA : backbone alpha carbon (specialized for glycing)
        - C : backbone carbonyl carbon
        - O : backbone carbonyl oxygen
        - CB : residue-specific beta carbon
    """

    def __init__(self):
        super().__init__(embedding_map_fivebead)


class CGEmbeddingMapCA(CGEmbeddingMap):
    """
    One-bead embedding map defined by:
        - CA : backbone alpha carbon, carrying aminoacid identity
    """

    def __init__(self):
        ca_dict = {key: emb for key, emb in embedding_map_fivebead.items() if emb <= 20}
        super().__init__(ca_dict)


all_residues = [
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
]


def embedding_fivebead(atom_df):
    """
    Helper function for mapping high-resolution topology to
    5-bead embedding map.
    """
    name, res = atom_df["name"], atom_df["resName"]
    if name in ["N", "C", "O"]:
        atom_type = embedding_map_fivebead[name]
    elif name == "CA":
        if res == "GLY":
            atom_type = embedding_map_fivebead["GLY"]
        else:
            atom_type = embedding_map_fivebead[name]
    elif name == "CB":
        atom_type = embedding_map_fivebead[res]
    else:
        print(f"Unknown atom name given: {name}")
        atom_type = "NA"
    return atom_type


def embedding_ca(atom_df):
    """
    Helper function for mapping high-resolution topology to
    CA embedding map.
    """
    name, res = atom_df["name"], atom_df["resName"]
    if name == "CA":
        atom_type = embedding_map_fivebead[res]
    else:
        print(f"Unknown atom name given: {name}")
        atom_type = "NA"
    return atom_type


class CGEmbeddingMapCA_WRC(CGEmbeddingMap):
    """
    One-bead embedding map defined by:
        - CA : backbone alpha carbon, carrying aminoacid identity
        - PROa : Adding an extended embedding for the PROLINE in two specific bonds 
    """

    ext_embedding = {
        "PROa": 21,
    }

    def __init__(self):
        ca_dict = {key: emb for key, emb in embedding_map_fivebead.items() if emb <= 20}
        ca_dict.update(self.ext_embedding)
        super().__init__(ca_dict)



def embedding_ca_WRC(atom_df):
    """
    Helper function for mapping high-resolution topology to
    CA WRC modified embedding map.
    """

    inv_CGEmbeddingMapCA_WRC = {label: resi for resi, label in CGEmbeddingMapCA_WRC().items()}

    ext_embedding = {
       1349 : 21,
       2422 : 21
    }

    name, res, idxSeq, idx = atom_df["name"], atom_df["resName"], atom_df["resSeq"], atom_df['serial']

    if name == "CA":

        if idxSeq in ext_embedding.keys():
            
            new_res = inv_CGEmbeddingMapCA_WRC[ext_embedding[idxSeq]]

            if res in new_res:
                atom_type = CGEmbeddingMapCA_WRC()[new_res]
                print(f"Residue {res} with residue idx {idxSeq} and atom idx {idx} was mapped to {new_res} in WRC embedding with atom type {atom_type}.")
            else:
                raise ValueError(
                    f"Residue  {res} has no special mapping."
                )

        else:
            atom_type = embedding_map_fivebead[res]

    else:
        print(f"Unknown atom name given: {name}")
        atom_type = "NA"
    
    return atom_type