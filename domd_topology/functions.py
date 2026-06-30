from typing import Union

from rdkit import Chem


def divide_into_molecules(aa_system):
    """Divides a molecular system into individual connected components.

        Splits a molecular object containing multiple disconnected fragments (e.g., a solvent box
        or a polymer mixture) into a list of separate, editable molecule objects.

        Args:
            aa_system (rdkit.Chem.rdchem.Mol): The input all-atom molecular system.

        Returns:
            list[rdkit.Chem.rdchem.RWMol]: A list of separated, editable molecule objects.
    """
    res = []
    for m in Chem.rdmolops.GetMolFrags(aa_system, asMols=True):
        n = Chem.RWMol()
        for atom in m.GetAtoms():
            n.AddAtom(atom)
        for bond in m.GetBonds():
            n.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType())
        res.append(n)
    return res


def set_molecule_id_for_h(molecule: Union[Chem.RWMol, Chem.Mol]) -> Union[Chem.RWMol, Chem.Mol]:
    """Assigns residue properties to hydrogen atoms based on their bonded heavy atoms.

        Iterates through all atoms in the molecule. If an atom is a heavy atom (atomic number != 1),
        it propagates its residue identifiers ('res_id', 'global_res_id') and residue name
        ('res_name') to all bonded hydrogen neighbors. This ensures hydrogens are correctly
        associated with their parent residues.

        Args:
            molecule (Union[Chem.RWMol, Chem.Mol]): The input molecule to modify.

        Returns:
            Union[Chem.RWMol, Chem.Mol]: The modified molecule with updated hydrogen properties.
    """
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() != 1:
            for nbr_atom in atom.GetNeighbors():
                if nbr_atom.GetAtomicNum() == 1:
                    nbr_atom.SetIntProp("res_id", atom.GetIntProp("res_id"))
                    nbr_atom.SetIntProp("global_res_id", atom.GetIntProp("global_res_id"))
                    nbr_atom.SetProp('res_name', atom.GetProp('res_name'))
    return molecule
