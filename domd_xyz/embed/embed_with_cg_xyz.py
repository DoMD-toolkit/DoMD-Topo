from typing import Union
from rdkit.Chem.rdForceFieldHelpers import MMFFHasAllMoleculeParams, MMFFGetMoleculeProperties, MMFFGetMoleculeForceField, MMFFSanitizeMolecule
import rdkit.ForceField 
import networkx as nx
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

from misc.logger import logger


class Position(object):
    """A simple container for 3D Cartesian coordinates.

        Attributes:
            x (float): The x-coordinate.
            y (float): The y-coordinate.
            z (float): The z-coordinate.
    """
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class FakeConf():
    """A mock class mimicking RDKit's Conformer API.

        This class is used to store atom positions generated from fragments before
        transferring them to a real RDKit Conformer. It provides a dictionary-based
        storage mechanism for non-contiguous updates.

        Attributes:
            x (dict): A dictionary mapping atom indices to Position objects.
    """
    def __init__(self, num_atoms):
        self.x = {}

    def set_pos(self, i, pos):
        """Sets the position for a specific atom index.

                Args:
                    i (int): The atom index.
                    pos (Position): The position object.
        """
        self.x[i] = pos

    def GetAtomPosition(self, idx):
        """Retrieves the position of a specific atom.

                Args:
                    idx (int): The atom index.

                Returns:
                    Position: The position object, or None if not set.
        """
        return self.x.get(idx)

    def GetPositions(self):
        """Returns all positions as a numpy array.

                Returns:
                    np.ndarray: A (N, 3) array of coordinates, where N is the number of stored atoms.
                        Rows are ordered by atom index.
        """
        ret = np.zeros((len(self.x), 3))
        for k in self.x:
            ret[k, 0] = self.x[k].x
            ret[k, 1] = self.x[k].y
            ret[k, 2] = self.x[k].z
        return ret

    def SetAtomPosition(self, idx, p):
        """Sets the position for an atom using either an array or Position object.

                Args:
                    idx (int): The atom index.
                    p (Union[np.ndarray, Position]): The coordinates [x, y, z] or Position object.
        """
        if isinstance(p, np.ndarray):
            self.x[idx] = Position(*p)
        else:
            self.x[idx] = p


def generate_pos_res(molecule: Union[Chem.Mol, Chem.RWMol], cg_mol: nx.Graph) -> FakeConf | None:
    r"""Generates 3D coordinates for a large molecule by embedding fragments (residues) sequentially.

    This function breaks the molecule into overlapping fragments based on the Coarse-Grained (CG)
    graph topology to handle large systems where global embedding might fail.

    **Algorithm Flow:**

    1.  **Mapping**: Map atoms to their global residue IDs ($R_{id}$).

    2.  **Fragment Construction**: For each node (residue) $v$ in the CG graph $G_{CG}$:

        * Define the fragment set $S_v = \{v\} \cup N(v)$, where $N(v)$ are the neighbors of $v$.
        * Construct a temporary RDKit molecule $M_{frag}$ containing atoms belonging to $S_v$.
        * This overlap ensures that the geometry at the connection points (linkers) is preserved.
    3.  **Sanitization & Handling**:

        * Handle broken aromatic rings by turning them non-aromatic in the fragment.
        * Preserve chirality: $\text{Tag}_{frag}(a) \leftarrow \text{Tag}_{global}(a)$.
    4.  **Embedding**:

        * Generate 3D coordinates $\mathbf{X}_{frag}$ for $M_{frag}$ using `AllChem.EmbedMolecule`.
        * **Chirality Correction**: If the generated fragment's chirality differs from the input,
            flip the chiral tag (CW $\leftrightarrow$ CCW) and re-embed.
    5.  **Assembly**:

        * Extract coordinates only for atoms belonging to the central residue $v$:
        .. math::

           \mathbf{X}_{global}(a) = \mathbf{X}_{frag}(a) \quad \forall a \in v

        * Store in `FakeConf`.

    Args:
        molecule (Union[Chem.Mol, Chem.RWMol]): The full all-atom molecule.
        cg_mol (nx.Graph): The coarse-grained graph describing the topology of residues.

    Returns:
        FakeConf | None: A mock conformer containing the assembled coordinates, or None if failed.
    """
    logger.info("Generating positions for molecule via fragments, note that the chirality may be broken.")
    conf = FakeConf(molecule.GetNumAtoms())
    atom_map = {}
    num_atoms = molecule.GetNumAtoms()
    for atom in molecule.GetAtoms():
    #for atom_id in range(num_atoms):
        #atom = molecule.GetAtomWithIdx(atom_id)
        res_id = atom.GetIntProp('global_res_id')
        if not atom_map.get(res_id):
            atom_map[res_id] = []
        atom_map[res_id].append(atom.GetIdx())
    adj_dict = dict(cg_mol.adjacency())
    for m_id in tqdm(cg_mol.nodes, total=len(cg_mol.nodes), desc='generating pos fragmently', disable=True):
        input_smiles = cg_mol.nodes[m_id]['smiles']
        input_mol = Chem.MolFromSmiles(input_smiles)
        input_mol = Chem.AddHs(input_mol)
        maxA = 1000
        input_mol_confid = -1
        while input_mol_confid == -1:
            input_mol_confid = AllChem.EmbedMolecule(input_mol,useRandomCoords=False,maxAttempts=maxA)
            maxA += 1000
        ChiralCenters = Chem.FindMolChiralCenters(input_mol, includeUnassigned=False)
        withChiralCenters = False
        if len(ChiralCenters) != 0:
            withChiralCenters = True
        # generate position of A by its neighbor monomers
        # to obtain better monomer-monomer connections
        n_ids = list(adj_dict[m_id].keys())
        #print(n_ids)
        fragment = Chem.RWMol()
        fragment_ids = [m_id] + n_ids
        bonds = set()
        local_map1 = {}
        local_map2 = {}
        count = 0
        broke = set()
        for res_id in fragment_ids:  # Get current monomer
            atoms = atom_map.get(res_id)
            if atoms is not None:
                for atom_id in atoms:
                    local_map1[atom_id] = count # global molecule id to fragment id
                    local_map2[count] = atom_id  # map fragment id to global molecule id
                    count += 1
        frg_aid_to_mol_aid = {}
        for i in range(count):
            atom_id = local_map2[i]
            atom = molecule.GetAtomWithIdx(atom_id)
            frg_aid = fragment.AddAtom(atom)
            frg_aid_to_mol_aid[frg_aid] = atom_id
            for bond in atom.GetBonds():
                a, b, t = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType()
                if molecule.GetAtomWithIdx(a).GetIntProp("global_res_id") not in fragment_ids:
                    # print(f"Atom {a} with residue idx {molecule.GetAtomWithIdx(a).GetIntProp('global_res_id')} not in fragment {fragment_ids}")
                    broke.add(atom_id)
                    continue
                if molecule.GetAtomWithIdx(b).GetIntProp("global_res_id") not in fragment_ids:
                    # print(f"Atom {b} with residue idx {molecule.GetAtomWithIdx(b).GetIntProp('global_res_id')} not in fragment {fragment_ids}")
                    broke.add(atom_id)
                    continue
                if a < b:
                    bonds.add((a, b, t))
                else:
                    bonds.add((b, a, t))
        # logger.warning(f'The broken atoms:{broke}')
        for bond in bonds:
            fragment.AddBond(local_map1[bond[0]], local_map1[bond[1]], bond[2])
        for bond in bonds:
            molbond = molecule.GetBondBetweenAtoms(bond[0],bond[1])
            stereo = molbond.GetStereo()
            satoms = list(molbond.GetStereoAtoms())
            #print(satoms,bond)
            #fragment.AddBond(local_map1[bond[0]], local_map1[bond[1]], bond[2])
            new_bond = fragment.GetBondBetweenAtoms(local_map1[bond[0]], local_map1[bond[1]])
            new_bond.SetBondDir(molbond.GetBondDir())
            if stereo in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE):
                new_bond.SetStereoAtoms(local_map1[satoms[0]], local_map1[satoms[1]])
            new_bond.SetStereo(stereo)
        # print(Chem.MolToSmiles(fragment))
        # _rma = []
        # for atom_ in fragment.GetAtoms():
        #    if local_map2.get(atom_.GetIdx()) is None:
        #        continue
        #    m_a_id = local_map2.get(atom_.GetIdx())
        #    atom__ = molecule.GetAtomWithIdx(m_a_id)
        #    if atom__.GetIntProp("global_res_id") != m_id and atom__.GetSymbol() == 'H':
        #        _rma.append(atom_.GetIdx())
        # for idx in sorted(_rma, reverse=True):
        #    fragment.RemoveAtom(idx)
        #if cg_mol.nodes[m_id]['type'] == 'GL1':
        #    print(Chem.MolToSmiles(fragment))

        chiral_tag_center = {}
        for atom in fragment.GetAtoms():  # avoiding breaking aromatic rings
            if atom.GetIntProp("global_res_id") == m_id:
                chiral_tag_center[atom.GetIdx()] = molecule.GetAtomWithIdx(frg_aid_to_mol_aid[atom.GetIdx()]).GetChiralTag()#atom.GetChiralTag()  # only care about the target monomer
                #if cg_mol.nodes[m_id]['type'] == 'GL1':
                #    print(molecule.GetAtomWithIdx(frg_aid_to_mol_aid[atom.GetIdx()]).GetChiralTag())
            if atom.GetIsAromatic():
                if not atom.IsInRing():
                    atom.SetIsAromatic(0)
            if local_map2.get(atom.GetIdx()) in broke:
                # if atom.GetIsAromatic() and atom.IsInRing():
                #    continue
                atom.SetIsAromatic(0)
                for btom in atom.GetNeighbors():
                    # logger.info(f'Neighbors of broken atom {atom.GetIdx()} : {btom.GetIdx()}')
                    if btom.GetIsAromatic() and btom.IsInRing():
                        continue
                    btom.SetIsAromatic(0)
                    _bond = fragment.GetBondBetweenAtoms(atom.GetIdx(), btom.GetIdx())
                    _bond.SetBondType(Chem.rdchem.BondType.SINGLE)
        #if cg_mol.nodes[m_id]['type'] == 'GL1':
        #    print(Chem.MolToSmiles(fragment,isomericSmiles=True))
        # print(Chem.MolToSmiles(fragment), '------')
        # fragment = AllChem.AddHs(fragment)
        Chem.SanitizeMol(fragment, Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
        res = Chem.SanitizeMol(fragment, catchErrors=True)

        if not res is Chem.rdmolops.SanitizeFlags.SANITIZE_NONE:
            logger.warning(f'Sanitize failed on: {Chem.MolToSmiles(fragment)}')
            # fragment = Chem.MolFromSmiles(Chem.MolToSmiles(fragment))
            # fragment = Chem.MolFromPDBBlock(Chem.MolToPDBBlock(fragment, flavor=4))
        # I don't know why yet
        # but not important, for the truncated monomers are not used
        # raise ValueError(f"{res}, {Chem.MolToSmiles(fragment)}")
        _mh = AllChem.AddHs(fragment)
        # print(Chem.MolToSmiles(_mh),'2222222')
        # print(Chem.MolToSmiles(_mh))
        for atom in _mh.GetAtoms():
            if not chiral_tag_center.get(atom.GetIdx()) is None:
                atom.SetChiralTag(chiral_tag_center[atom.GetIdx()])
            else:
                atom.SetChiralTag(rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
                # other monomers will be set to unspecified
        conf_id = AllChem.EmbedMolecule(_mh, maxAttempts=10000, useRandomCoords=False)
        maxA = 10000
        while conf_id == -1:
            conf_id = AllChem.EmbedMolecule(_mh, maxAttempts=maxA, useRandomCoords=False)
            maxA += 5000
            logger.info(f'Generating monomer fragment at maxAttempts {maxA}...')
        #conf_id = AllChem.EmbedMolecule(_mh, maxAttempts=500, useRandomCoords=True)
        if conf_id != -1 and res is not Chem.rdmolops.SanitizeFlags.SANITIZE_NONE:
            logger.warning(f'Sanitize failed on: {Chem.MolToSmiles(fragment)} with H: {Chem.MolToSmiles(_mh)}')
        # if not res is Chem.rdmolops.SanitizeFlags.SANITIZE_NONE:
        # Chem.MolToPDBFile(_mh, 'a0.pdb', flavor=4)
        # AllChem.UFFOptimizeMolecule(_mh)
        if conf_id == -1:
            _mh_metadata = {'atoms': {}, 'bonds': set()}
            for bond in _mh.GetBonds():
                a = bond.GetBeginAtom()
                b = bond.GetEndAtom()
                _mh_metadata['atoms'][a.GetIdx()] = a.GetSymbol()
                _mh_metadata['atoms'][a.GetIdx()] = a.GetSymbol()
                _mh_metadata['bonds'].add((a.GetIdx(), b.GetIdx()))
            # pickle.dump(_mh_metadata,open('/home/lmy/HTSP/FPSG/fuckmol_meta.pkl','wb'))
            # pickle.dump((_mh,fragment),open('/home/lmy/HTSP/FPSG/fuckmol.pkl','wb'))
            logger.error(
                f"Residue generation with conversation of chirality failed!\n{Chem.MolToSmiles(_mh, isomericSmiles=True)}")
            logger.error(f"The chirality of target monomer is {chiral_tag_center}, please make sure that the cut"
                         f"off method does not break chirality of molecule!")
            logger.error(
                f"This error would not stop the generation program but use a random initial coordinate for bead."
                f"This may contribute to wrong conformation breaking chirality.")
            conf_id = -1
            maxA = 1000
            while conf_id == -1:
                conf_id = AllChem.EmbedMolecule(_mh, maxAttempts=1000, useRandomCoords=False)
                maxA += 1000
                logger.info(f'Generating monomer with maxAttempts={maxA}.')
                if maxA >= 20000:
                    conf_id = 0
            if maxA >= 20000:
                conf_id = -1
            if conf_id == -1:
                logger.error(
                    f"Residue generation without conversation of chirality failed! Please check your reaction template, make sure"
                    f"get the right molecular in SMARTS")
                return
            #AllChem.UFFOptimizeMolecule(_mh,maxIters=2000)
            AllChem.MMFFOptimizeMolecule(_mh,maxIters=5000)
        if  withChiralCenters:
            ChiralCenters_mh = Chem.FindMolChiralCenters(_mh, includeUnassigned=False)
            # sort Chiral Centers Pattern
            ChiralCenters_sorted = sorted(ChiralCenters,key=lambda x: x[0])
            ChiralCenters_mh_sorted = sorted(ChiralCenters_mh,key=lambda x: x[0])
            if len(ChiralCenters_sorted) != len(ChiralCenters_mh_sorted):
                logger.error('Wrong Chiral Centers were found for the fragment, we recommend modify the input SMILES or SMARTS.')
                return
            ChiralChange = {}
            for (i0, Chiral0), (i1, Chiral1) in zip(ChiralCenters_sorted,ChiralCenters_mh_sorted):
                if Chiral0 != Chiral1:
                    ChiralChange[i1] = True
                else:
                    ChiralChange[i1] = False
            #print(ChiralChange)
            #print(Chem.MolToSmiles(_mh))
            for i1 in ChiralChange:
                if ChiralChange[i1]:
                    atom = _mh.GetAtomWithIdx(i1)
                    chiralTag = atom.GetChiralTag()
                    if chiralTag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
                        atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW)
                    elif chiralTag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
                        atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)
            #print(Chem.MolToSmiles(_mh))
            #print(Chem.MolToSmiles(input_mol))
            Changed_ChiralCenters_mh = Chem.FindMolChiralCenters(_mh, includeUnassigned=False)
            Changed_ChiralCenters_mh_sorted = sorted(Changed_ChiralCenters_mh,key=lambda x: x[0])
            #print('-'*100)
            #print(ChiralChange)
            #print(ChiralCenters_sorted)
            #print(ChiralCenters_mh_sorted)
            #print(Changed_ChiralCenters_mh_sorted)
            #print('-'*100)
            conf_id = AllChem.EmbedMolecule(_mh, maxAttempts=10000, useRandomCoords=False)
            maxA = 10000
            while conf_id == -1:
                conf_id = AllChem.EmbedMolecule(_mh, maxAttempts=maxA, useRandomCoords=False)
                maxA += 5000
                logger.info(f'Generating monomer fragment at maxAttempts {maxA}...')


        _conf = _mh.GetConformer(conf_id)
        for atom in _mh.GetAtoms():
            if local_map2.get(atom.GetIdx()) is None:
                continue
            m_a_id = local_map2.get(atom.GetIdx())
            p = _conf.GetAtomPosition(atom.GetIdx())
            if molecule.GetAtomWithIdx(m_a_id).GetIntProp("global_res_id") != m_id:
                continue
            logger.debug(f'{atom.GetSymbol()}, {molecule.GetAtomWithIdx(m_a_id).GetSymbol()}')
            conf.set_pos(m_a_id, Position(p.x, p.y, p.z))
    return conf


def embd(molecule: Union[Chem.Mol, Chem.RWMol], cg_molecule: nx.Graph, large: int = 500, custom_conf=None):
    """Main function to embed a molecule into 3D space.

        Decides between standard embedding for small molecules and fragment-based
        embedding for large molecules.

        Strategy:
            1. If `custom_conf` is provided, use it directly.
            2. If molecule size > `large`, use `generate_pos_res` (fragment-based).
            3. Else, use `AllChem.EmbedMolecule` (standard ETKDG).

        Args:
            molecule (Union[Chem.Mol, Chem.RWMol]): The RDKit molecule to embed.
            cg_molecule (nx.Graph): The coarse-grained graph template.
            large (int, optional): Threshold for switching to fragment-based embedding. Defaults to 500.
            custom_conf (rdkit.Chem.Conformer, optional): Pre-existing conformer to use. Defaults to None.

        Returns:
            rdkit.Chem.Conformer: The generated 3D conformer.
    """
    if custom_conf is not None:
        molecule.AddConformer(custom_conf)
        return molecule.GetConformer(0)
    if molecule.GetNumConformers() > 0:
        logger.warning(f"The molecule has {molecule.GetNumConformers()} conformers, return the 0th.")
        return molecule.GetConformer(0)
    if molecule.GetNumAtoms() > large:
        logger.info(f"Num of atoms {molecule.GetNumAtoms()} is greater than {large}, generating by residue.")
        conf = generate_pos_res(molecule, cg_molecule)
        logger.info(f"Generated position fragmently")
        #print(conf.GetPositions())
        if (conf is None) or (len(conf.x) != molecule.GetNumAtoms()):
            logger.error(f"Configuration generation error.")
            return
    else:
        withChiralCenters = False
        for m_id in cg_molecule.nodes:
            input_smiles = cg_molecule.nodes[m_id]['smiles']
            input_mol = Chem.MolFromSmiles(input_smiles)
            input_mol = Chem.AddHs(input_mol)
            maxA = 1000
            input_mol_confid = -1
            while input_mol_confid == -1:
                input_mol_confid = AllChem.EmbedMolecule(input_mol,useRandomCoords=False,maxAttempts=maxA)
                maxA += 1000
            ChiralCenters = Chem.FindMolChiralCenters(input_mol, includeUnassigned=False)
            if len(ChiralCenters) != 0:
                withChiralCenters = True
        if withChiralCenters:
            useR = False
        else:
            useR = True
        conf_id = -1
        i = 1
        maxAttempts = 10000
        seed = np.random.randint(0,high=10000)
        while conf_id == -1:
            conf_id = AllChem.EmbedMolecule(molecule, useRandomCoords=useR, maxAttempts=maxAttempts)
            logger.info(f'Configuration generation for molecule less than large. iteration {i:0>4d}!')
            if conf_id == -1:
                maxAttempts += 5000
                seed = np.random.randint(0,high=10000)
                logger.info(f'Configuration generation failed at iteration {i:0>4d}. Add maxAttempts to {maxAttempts}')
            i += 1
        conf = molecule.GetConformer(conf_id)
        #if conf_id == -1:
        #    conf = generate_pos_res(molecule, cg_molecule)
        #    if len(conf.x) != molecule.GetNumAtoms():
        #        logger.warning(f"Configuration generation error! {len(conf.x)} != {molecule.GetNumAtoms()}")
        #else:
        #    conf = molecule.GetConformer(conf_id)
    #print(conf.GetPositions())
    return conf
