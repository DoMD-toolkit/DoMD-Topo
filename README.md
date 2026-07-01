# DoMD-Topo

```text
░███████              ░███     ░███ ░███████          ░██████████                                   
░██   ░██             ░████   ░████ ░██   ░██             ░██                                       
░██    ░██  ░███████  ░██░██ ░██░██ ░██    ░██            ░██      ░███████  ░████████   ░███████  
░██    ░██ ░██    ░██ ░██ ░████ ░██ ░██    ░██ ░██████    ░██     ░██    ░██ ░██    ░██ ░██    ░██ 
░██    ░██ ░██    ░██ ░██  ░██  ░██ ░██    ░██            ░██     ░██    ░██ ░██    ░██ ░██    ░██ 
░██   ░██  ░██    ░██ ░██       ░██ ░██   ░██             ░██     ░██    ░██ ░███   ░██ ░██    ░██ 
░███████    ░███████  ░██       ░██ ░███████              ░██      ░███████  ░██░█████   ░███████  
                                                                             ░██                    
                                                                             ░██                    
```

A lightweight **"Chemical Compilation"** engine driven by the SMILES/SMARTS-driven Coarse-Graining/Fine-Graining (S-CGFG) framework to reconstruct All-Atom (AA) configurations from Coarse-Grained (CG) models.

---

## 1. System Specifications & Input Formats

The reconstruction pipeline is driven by three essential components, mapping localized chemical identifiers into structured, macro-topological spaces.

### 1.1 Coarse-Grained Graph (`cg_mol`)
A NetworkX graph representation of the coarse-grained layout. The structure must comply with the following attributes:

* **Graph Attributes**:
    * `is_rigid` *(bool)*: Designates whether the subsystem operates as a static, non-deformable rigid body.
* **Node Attributes (Per Bead)**:
    * `type` *(str)*: Coarse-grained bead classification mapping to the chemical templates.
    * `smiles` *(str)*: All-atom atomic composition string associated with the local bead region.
    * `x` *(np.ndarray)*: Target 3D spatial coordinate vector ($3 \times 1$) of the CG bead center.
    * `body` *(int)*: Structural rigid-body grouping tracker index.
* **Edge Attributes (Per Connection)**:
    * `bond_type` *(str/object)*: Local link parameters defining structural connectivity between beads.

### 1.2 Reactants Dictionary (`reactants_config`)
Establishes baseline molecular identities by linking CG bead classification keys directly to all-atom SMILES string paths.
```python
reactants_config = {
    'A': {'smiles': 'Nc1ccc(Oc2ccc(N)cc2)cc1', 'file': None},
    'B': {'smiles': 'O1C(=O)c2cc(Oc3cc4C(=O)OC(=O)c4cc3)ccc2C1=O', 'file': None},
}
```

### 1.3 Connection Templates (`reaction_template`)
Directs local transformation logic using reaction SMARTS for subgraph parsing and bonding modifications.

#### Core Paradigm: Atom Labeling
Reaction paths utilize distinct **numerical mapping tags** (e.g., `[#7H2:1]`) to track reaction boundaries:
* **Labeled Nodes**: Act as the **active coordination centers** where chemical bonds are broken or created.
* **Unlabeled Nodes**: Provide critical local **chemical environment context** to guarantee matching fidelity without altering underlying structures.

---

## 2. Core APIs

### `topology_builder`
Generates full-atom connectivity systems by stitching independent monomer subgraphs according to the specified reaction templates.
```python
aa_mol_h, aa_graph = topology_builder(reactants_config, reaction_template, cg_mol, mol_idx)
```
* **Outputs**: Returns the structural `Chem.Mol` topology object along with its tracking `molecule_graph` metadata network.

### `embed_molecule`
Orchestrates high-performance 3D spatial coordinate construction using structural context metadata. Divides calculations into distinct rigid alignment, ETKDG standard generation, or domain-decomposed fragment pipelines.
```python
molecule, molecule_graph = embed_molecule(aa_mol_h, cg_mol, aa_graph, box, large, chunk_per_d)
```
* **Outputs**: Appends the final conformer layout to the `molecule` item and injects explicit coordinate arrays into the `molecule_graph` nodes under the `'x'` attribute.

---

## 3. Workflow Usage Example

The following loop shows the standard implementation of the decoupled topology reconstruction and 3D embedding engine:

```python
from domd_xyz.embed_molecule import embed_molecule
from domd_topology.topology_builder import topology_builder
from misc.io.sdf import post_process_aa_mol

large_threshold = 500
chunks_per_dimension = 1
# assume that you have get the right cg_mol (nx.Graph)
# 1. Reconstruct the All-Atom topological connection graph
aa_mol_h, aa_graph = topology_builder(
        reactants_config, 
        reaction_template, 
        cg_mol
    )
    
# 2. Compute 3D coordinate conformers based on CG target layouts
aa_mol_h, aa_graph = embed_molecule(
        aa_mol_h, 
        cg_mol, 
        aa_graph, 
        box=box_tensor, 
        large=large_threshold, 
        chunk_per_d=chunks_per_dimension
    )
```
