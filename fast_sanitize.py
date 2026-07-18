import networkx as nx
import numpy as np
from rdkit import Chem
from tqdm import tqdm

from misc.logger import logger


# def fast_sanitize(mol: Chem.Mol):
#     if mol.GetNumAtoms() < 10000:
#         Chem.SanitizeMol(mol)
#     else:
#         logger.warning("SANITIZE IS TURNED OFF! BE SURE YOU KNOW WHAT YOU ARE DOING!")
#         mol.UpdatePropertyCache(strict=False)
#         fast_sanitize_ops = (
#                 Chem.SANITIZE_ALL ^
#                 Chem.SANITIZE_KEKULIZE ^
#                 Chem.SANITIZE_SETAROMATICITY ^
#                 Chem.SANITIZE_SYMMRINGS
#         )
#         state = Chem.SanitizeMol(mol, sanitizeOps=fast_sanitize_ops, catchErrors=True)
#         Chem.FastFindRings(mol)
#         if state != 0:
#             logger.warning("Fast sanitize failed, turn to full mode")
#             Chem.SanitizeMol(mol)


def _build_networkx_graph(mol: Chem.Mol) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(mol.GetNumAtoms()))

    edges_to_add = []
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        for bond in atom.GetBonds():
            neighbor_idx = bond.GetOtherAtomIdx(atom_idx)
            if neighbor_idx > atom_idx:
                edges_to_add.append((atom_idx, neighbor_idx, {'bond_idx': bond.GetIdx()}))

    G.add_edges_from(edges_to_add)
    return G


def fast_sanitize(mol: Chem.Mol, max_path: int = 12, buffer_size: int = 3) -> Chem.Mol:
    if mol.GetNumAtoms() < 2000:
        Chem.SanitizeMol(mol)
        return mol

    logger.warning(f"FAST SANITIZE MODE IS ENABLED! RINGS ARE IDENTIFIED WITHIN {max_path} BONDS.")

    # Global basic preprocessing
    mol.UpdatePropertyCache(strict=False)
    r = mol.GetRingInfo()

    num_atoms = mol.GetNumAtoms()
    sanitized_flags = np.zeros(num_atoms, dtype=bool)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            sanitized_flags[atom.GetIdx()] = True

    global_ring_atoms_set = set()
    total_radius = max_path + buffer_size

    core_radius = max_path // 2

    # Build the NetworkX graph for topological traversal
    G = _build_networkx_graph(mol)

    # Systematically sweep all atoms
    for center_idx in tqdm(range(num_atoms), disable=True):
        if sanitized_flags[center_idx]:
            continue

        # BFS using NetworkX to get all nodes within the total radius
        distances = nx.single_source_shortest_path_length(G, center_idx, cutoff=total_radius)

        if not distances:
            sanitized_flags[center_idx] = True
            continue

        buffer_atoms = set(distances.keys())

        trust_atoms = {node for node, dist in distances.items() if dist <= max_path}
        done_atoms = {node for node, dist in distances.items() if dist <= core_radius}

        buffer_bonds_set = set()
        for u in buffer_atoms:
            for v, data in G[u].items():
                if v in buffer_atoms:
                    buffer_bonds_set.add(data['bond_idx'])

        buffer_bonds = list(buffer_bonds_set)

        if not buffer_bonds:
            sanitized_flags[center_idx] = True
            continue

        # Extract sub-molecule and get the atom mapping
        amap = {}
        sub_mol = Chem.PathToSubmol(mol, buffer_bonds, atomMap=amap)
        reverse_amap = {sub_idx: global_idx for global_idx, sub_idx in amap.items()}

        # Boundary broken-ring cleanup
        Chem.FastFindRings(sub_mol)
        sub_ri = sub_mol.GetRingInfo()

        for sub_atom in sub_mol.GetAtoms():
            if sub_atom.GetIsAromatic() and not sub_ri.NumAtomRings(sub_atom.GetIdx()):
                sub_atom.SetIsAromatic(False)

        for sub_bond in sub_mol.GetBonds():
            if sub_bond.GetIsAromatic() or sub_bond.GetBondType() == Chem.BondType.AROMATIC:
                if not sub_ri.NumBondRings(sub_bond.GetIdx()):
                    sub_bond.SetIsAromatic(False)
                    if sub_bond.GetBondType() == Chem.BondType.AROMATIC:
                        sub_bond.SetBondType(Chem.BondType.SINGLE)

        # Local sanitize (Skip PROPERTIES to ignore valence errors on the very edge)
        Chem.SanitizeMol(
            sub_mol,
            sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES,
            catchErrors=True
        )

        for sub_idx, global_idx in reverse_amap.items():
            if global_idx in trust_atoms:
                sub_atom = sub_mol.GetAtomWithIdx(sub_idx)
                global_atom = mol.GetAtomWithIdx(global_idx)

                global_atom.SetIsAromatic(sub_atom.GetIsAromatic())
                global_atom.SetHybridization(sub_atom.GetHybridization())

                global_atom.SetFormalCharge(sub_atom.GetFormalCharge())
                global_atom.SetNumExplicitHs(sub_atom.GetNumExplicitHs())
                global_atom.SetNoImplicit(sub_atom.GetNoImplicit())

                global_atom.SetNumRadicalElectrons(sub_atom.GetNumRadicalElectrons())
                global_atom.SetIsotope(sub_atom.GetIsotope())

                global_atom.SetChiralTag(sub_atom.GetChiralTag())

                sanitized_flags[global_idx] = True

        for sub_bond in sub_mol.GetBonds():
            g_idx1 = reverse_amap[sub_bond.GetBeginAtomIdx()]
            g_idx2 = reverse_amap[sub_bond.GetEndAtomIdx()]

            if g_idx1 in trust_atoms and g_idx2 in trust_atoms:
                global_bond = mol.GetBondBetweenAtoms(g_idx1, g_idx2)
                if global_bond:
                    global_bond.SetBondType(sub_bond.GetBondType())
                    global_bond.SetIsAromatic(sub_bond.GetIsAromatic())

                    global_bond.SetIsConjugated(sub_bond.GetIsConjugated())

                    global_bond.SetStereo(sub_bond.GetStereo())
                    global_bond.SetBondDir(sub_bond.GetBondDir())

        # Filter and map back ring information
        sub_ring_info = sub_mol.GetRingInfo()
        for sub_ring_atoms, sub_ring_bonds in zip(sub_ring_info.AtomRings(), sub_ring_info.BondRings()):
            original_g_ring_atoms = tuple(reverse_amap[sa] for sa in sub_ring_atoms)

            # Only accept rings that are fully enclosed within the Trust region
            if all(ga in trust_atoms for ga in original_g_ring_atoms):
                sorted_g_ring = tuple(sorted(original_g_ring_atoms))

                # Global deduplication
                if sorted_g_ring not in global_ring_atoms_set:
                    global_ring_atoms_set.add(sorted_g_ring)

                    g_ring_bonds = []
                    for sb_idx in sub_ring_bonds:
                        sb = sub_mol.GetBondWithIdx(sb_idx)
                        gb1, gb2 = reverse_amap[sb.GetBeginAtomIdx()], reverse_amap[sb.GetEndAtomIdx()]
                        g_ring_bonds.append(mol.GetBondBetweenAtoms(gb1, gb2).GetIdx())

                    r.AddRing(original_g_ring_atoms, tuple(g_ring_bonds))

    if len(global_ring_atoms_set) == 0:
        r.AddRing((), ())
        logger.info("Molecule has 0 rings. df_init flag forcefully activated via empty tuple hack.")

    return mol


if __name__ == '__main__':
    from rdkit.Chem import AllChem

    mol = Chem.RWMol()
    for _ in range(6):
        mol.AddAtom(Chem.Atom(6))
    mol.AddAtom(Chem.Atom(6))
    mol.AddBond(0, 1)
    mol.AddBond(1, 2)
    mol.AddBond(2, 3)
    mol.AddBond(3, 4)
    mol.AddBond(4, 5)
    m = mol.GetMol()
    fast_sanitize(m)
    fp_bit = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
