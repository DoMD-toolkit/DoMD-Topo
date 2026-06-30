import pickle
from typing import List, Union

import networkx as nx
import tqdm
from rdkit import Chem
from rdkit.Geometry import Point3D
from misc.parser import mols_to_nxgraphs, nxgraphs_to_mols
def write_mols_to_sdf(mols, output_path, force_v3000=True):
    """
    Writes a list of RDKit molecule objects into a single multi-molecule SDF file.
    Automatically preserves all atom properties and custom string metadata.

    Parameters:
        mols (list): List of RDKit Romol objects.
        output_path (str): Target path for the output .sdf file.
        force_v3000 (bool): If True, enforces V3000 format compliance (recommended for large systems).
    """
    # Initialize the SDWriter handler
    writer = Chem.SDWriter(output_path)

    if force_v3000:
        writer.SetForceV3000(True)

    for idx, mol in tqdm.tqdm(enumerate(mols), total=len(mols), desc='writing molecules to SDF', disable=False):
        if mol is None:
            continue
        # SDWriter automatically reads and writes all tags set via mol.SetProp()
        writer.write(mol)

    # Crucial: Always close the stream to flush buffer and finalize the file structure
    writer.close()

