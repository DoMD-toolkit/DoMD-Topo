import numpy as np
from rdkit import Chem
import logging
from domd_xyz.embed_molecule import embed_molecule
from domd_topology.reactor import Reactor
from domd_topology.functions import set_molecule_id_for_h
from misc.parser import mols_to_nxgraphs, nxgraphs_to_mols, read_cg_topology
from misc.io.xml import XmlParser
logger = logging.getLogger(__name__)


def build_aa_topology(mols_config, reaction_template, xml_path, reactions=None, rigid_meta=None, large=500, chunks_per_d=1):
    """
    Lightweight wrapper to reconstruct AA topology from CG system,
    generate 3D conformers, and inject essential metadata.

    Args:
        mols_config (dict): Monomer configuration dictionary, e.g., {'C': {'smiles': smiC, 'file': None}}
        reaction_template (dict): Reaction SMARTS patterns and topology rules.
        xml_path (str): Path to the GALAMOST CG XML configuration file.
        reactions (list/tuple, optional): Explicit sequence of reactions.
            If None, infers from the XML bond section.

    Returns:
        list[Chem.Mol]: A list of RDKit molecules with conformers and injected metadata.
    """
    logger.info('Parsing CG XML and extracting box dimensions...')
    xml = XmlParser(xml_path)

    # 1. Extract box info and convert to Angstroms
    box_coords = (xml.box.lx, xml.box.ly, xml.box.lz, xml.box.xy, xml.box.xz, xml.box.yz)
    box_tensor = np.array(tuple(map(float, box_coords)))[:3] * 10
    box_tensor_str = ' '.join([str(l) for l in box_tensor] + ['0']*6)

    # 3. Parse coarse-grained topology
    cg_sys, cg_mols = read_cg_topology(xml, mols_config)

    if rigid_meta is not None:
        for cg_mol in cg_mols:
            if cg_mol.graph['is_rigid']:
                mol_type = cg_mol.graph['type']
                cg_mol.graph['rigid_aidxs_map'] = rigid_meta[mol_type]['rigid_aidxs_map']

    # 4. Infer reaction tuples from XML if not explicitly provided
    if not reactions:
        reactions = []
        if 'bond' in xml.data:
            for bond in xml.data['bond']:
                reactions.append((bond[0], bond[1], bond[2]))

    # 5. Reconstruct all-atom (AA) connectivity via Reactor
    logger.info('Reconstructing AA connectivities via Reactor...')
    reactor = Reactor(mols_config, reaction_template)
    aa_mols, meta = reactor.process(cg_mols, reactions)

    # 6. Serial 3D coordinate embedding and metadata injection
    final_rdmols = []

    for i, aa_mol in enumerate(aa_mols):
        # Sanitize structure and add hydrogens
        Chem.SanitizeMol(aa_mol)
        aa_mol_h = Chem.AddHs(aa_mol)
        set_molecule_id_for_h(aa_mol_h)

        # Fragment embedding (using default safe thresholds for large systems)
        cg_mol = cg_mols[i]
        conf = embed_molecule(aa_mol_h, cg_mol, box=box_tensor, large=large, chunk_per_d=chunks_per_d)

        # Ensure the generated conformer is attached to the molecule
        if aa_mol_h.GetNumConformers() == 0 and conf is not None:
            aa_mol_h.AddConformer(conf, assignId=True)

        # 7. Extract and inject metadata: resname, res_id, box_tensor
        res_name = []
        res_id = []
        for a in aa_mol_h.GetAtoms():
            res_id.append(a.GetIntProp("global_res_id"))
            res_name.append(a.GetProp("res_name"))
        res_name_str = ' '.join(res_name)
        res_num_str = ' '.join(map(str, res_id))


        # Best for native RDKit compatibility (only supports strings/ints).
        aa_mol_h.SetProp("RES_NAMES", res_name_str)
        aa_mol_h.SetProp("RES_NUMS", res_num_str)
        aa_mol_h.SetProp("BOX_TENSOR", box_tensor_str)


        final_rdmols.append(aa_mol_h)
        logger.info(f'Successfully processed molecule {i+1} / {len(aa_mols)}')

    return final_rdmols

