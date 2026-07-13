from typing import Tuple

import networkx as nx
import numpy as np
from .domd_xyz.embedding_algorithms import embed_rigid, embed_hybrid, embed_by_etkdg, embed_by_fragment
from rdkit import Chem

from .domd_xyz.embed_with_cg_xyz import (
    analyze_topology
)
from .misc.logger import logger


def embed_molecule(
        molecule: Chem.Mol,
        cg_molecule: nx.Graph,
        molecule_graph: nx.Graph,
        box: np.ndarray = None,
        large: int = 500,
        chunk_per_d: int = 1
) -> Tuple[Chem.Mol, nx.Graph]:
    """
    The Ultimate Top-Level Orchestrator. Standardizes topology maps, routes the system
    to specialized solvers based on explicit rigidity modes ('FLEXIBLE', 'RIGID', 'HYBRID'),
    updates all properties, and returns both objects.
    """
    global2local, local2atoms = analyze_topology(molecule_graph, cg_molecule)

    if box is None:
        cg_coords = np.array([d['x'] for n, d in cg_molecule.nodes(data=True) if 'x' in d])
        min_coords = cg_coords.min(axis=0)
        max_coords = cg_coords.max(axis=0)
        span = max_coords - min_coords
        box = span + 100.0

        logger.warning(
            f"Simulation box bounds missing. Dynamically guessed bounding box from CG coordinates: "
            f"[{box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}] Å (with 50Å padding)."
        )

    rigidity_mode = cg_molecule.graph.get('rigidity', 'FLEXIBLE')

    if rigidity_mode == 'RIGID':
        logger.info("Embedding pure rigid system...")
        conf = embed_rigid(molecule, cg_molecule, box)

    elif rigidity_mode == 'HYBRID':
        logger.info("Embedding hybrid/grafting system with mixed rigid and flexible components...")
        conf = embed_hybrid(molecule, molecule_graph, cg_molecule, local2atoms, box, large, chunk_per_d)

    elif rigidity_mode == 'FLEXIBLE':
        logger.info("Embedding flexible system...")
        if molecule.GetNumAtoms() <= large:
            logger.info(
                f"System size ({molecule.GetNumAtoms()} atoms) <= threshold ({large}). Routing to embed_by_etkdg.")
            conf = embed_by_etkdg(molecule, cg_molecule, molecule_graph, local2atoms, box, chunk_per_d)
        else:
            logger.info(
                f"System size ({molecule.GetNumAtoms()} atoms) > threshold ({large}). Routing to embed_by_fragment.")
            conf = embed_by_fragment(molecule, molecule_graph, cg_molecule, local2atoms, box, chunk_per_d)
    else:
        raise ValueError(
            f"Unknown rigidity mode tag encountered in global config: '{rigidity_mode}', expected one of ['FLEXIBLE', 'RIGID', 'HYBRID'].")

    if conf is not None:
        if molecule.GetNumConformers() > 0:
            molecule.RemoveAllConformers()

        molecule.AddConformer(conf, assignId=True)

        for atom in molecule.GetAtoms():
            a_id = atom.GetIdx()
            pos = conf.GetAtomPosition(a_id)
            molecule_graph.nodes[a_id]['x'] = np.array([pos.x, pos.y, pos.z])

        logger.info("Successfully bound Conformer properties and mapped 'x' tensor coordinates into the graph.")
    else:
        logger.error("Failed to generate valid geometric parameters across all available workshops.")

    return molecule, molecule_graph
