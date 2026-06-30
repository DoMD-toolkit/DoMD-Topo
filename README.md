# DoMD-Topo
```text


░███████              ░███     ░███ ░███████           ░██████████                                 
░██   ░██             ░████   ░████ ░██   ░██              ░██                                     
░██    ░██  ░███████  ░██░██ ░██░██ ░██    ░██             ░██     ░███████  ░████████   ░███████  
░██    ░██ ░██    ░██ ░██ ░████ ░██ ░██    ░██ ░██████     ░██    ░██    ░██ ░██    ░██ ░██    ░██ 
░██    ░██ ░██    ░██ ░██  ░██  ░██ ░██    ░██             ░██    ░██    ░██ ░██    ░██ ░██    ░██ 
░██   ░██  ░██    ░██ ░██       ░██ ░██   ░██              ░██    ░██    ░██ ░███   ░██ ░██    ░██ 
░███████    ░███████  ░██       ░██ ░███████               ░██     ░███████  ░██░█████   ░███████  
                                                                             ░██                   
                                                                             ░██                   
                                                                                                   
```
A lightweight "Chemical Compilation" engine driven by the SMILES/SMARTS-driven Coarse-Graining/Fine-Graining (S-CGFG) framework to reconstruct All-Atom (AA) configurations from Coarse-Grained (CG) models.

---

## 1. API Usage

### Installation

```bash
conda create -n domd-toolkit -c conda-forge python==3.12 nomkl numpy rdkit=2025.03.6 openbabel numba networkx pandas scipy jupyter scikit-learn matplotlib MDAnalysis
conda activate domd-toolkit
```

### Core API Example

```python
from misc.pipeline import build_aa_topology
from misc.logger import logger
from misc.io.sdf import write_mols_to_sdf

logger.setLevel('INFO')

# Define input parameters
mols = {
    'A': {'smiles': 'Nc1ccc(Oc2ccc(N)cc2)cc1', 'file': None},
    'B': {'smiles': 'O1C(=O)c2cc(Oc3cc4C(=O)OC(=O)c4cc3)ccc2C1=O', 'file': None},
}

reaction_template = {
    'B-A': {
        'cg_reactant_list': [('A', 'B')],
        'smarts': '[#7H2:1].[#6:3](=[#8:4])[#8:2][#6:5]=[#8:6]>>[#6:3](=[#8:4])[#7:1][#6:5]=[#8:6].[#8:2]',
        'prod_idx': [0]
    }
}

# Execute topology reconstruction and 3D embedding
rdmols = build_aa_topology(mols, reaction_template, 'cg.xml', reactions=None, large=100)

# Export the generated conformers
write_mols_to_sdf(rdmols, 'polyimide.sdf')
```

### Arguments

* **`mols_config`** *(dict)*: Dictionary mapping CG bead types to their atomic SMILES definitions.
* **`reaction_template`** *(dict)*: Dictionary containing reaction SMARTS rules for fragment connectivity.
* **`xml_path`** *(str)*: Path to the PyGAMD CG `.xml` configuration file.
* **`reactions`** *(list, optional)*: Explicit ordered sequence of reactions. If `None`, automatically inferred from the XML `<bond>` section.
* **`large`** *(int, default 500)*: Molecule size threshold. Molecules smaller than this use ETKDG directly; larger systems trigger chunk-based embedding optimization.
* **`chunks_per_d`** *(int, default 1)*: Spatial grid subdivision factor for large polymer networks optimization.

---

## 2. Input Formats

### Monomer Definitions (SMILES Node Identity)
Defines the internal atomic structure of individual coarse-grained (CG) beads. This abstraction completely decouples the user from rigid coordinate/topology files (like `.itp` or `.gro`), allowing users to easily adjust mapping resolutions on the fly by changing the input SMILES strings.

### Reaction Templates (SMARTS Connectivity Logic)
Defines the graph modification logic using reaction SMARTS. Numerical atom maps (e.g., `[C:1]`) identify the active binding sites where bonds are broken or formed, while unmapped atoms define the required local chemical neighborhood without being altered.

---

## 3. Design Philosophy & Workflows

### S-CGFG Framework
Instead of treating backmapping as a purely geometric packing problem, DoMD-Topo approaches it as a **"Chemical Compilation"** process. It abstracts molecular systems into a generalized topological representation, providing resolution independence and complete chemical fidelity.

### High-Performance Fragmented Reactions
To prevent exponential performance degradation when assembling massive macromolecules, the algorithm uses a fragmented approach:
1. **Static Matching**: Subgraph matching is executed only on the small monomer templates rather than the expanding full polymer.
2. **Delta Recording**: Structural transformations are tracked as localized matrix differences.
3. **Unified Modification**: All tracked changes are batched and executed in a single, efficient graph assembly step.

### State Verification via Reacted Atom Set
To guarantee correct chemical valency and prevent impossible hyper-branched structures, an `allow_p` flag monitors active sites. Once a mapped atom reacts, it enters the `Reacted Atom Set`, dynamically blocking subsequent conflicting reactions at that specific node.

### Hierarchical 3D Coordinate Embedding
Reconstructing the full 3D coordinates utilizes an integrated optimization workflow:
1. **Fragment Generation**: High-fidelity 3D configurations are initiated via the ETKDG algorithm.
2. **Analytical Alignment**: Minimizes distance errors at connecting junctions using analytical rotation matrices.
3. **Domain Decomposition**: Large grids are divided into spatial chunks based on `chunks_per_d` for localized domain optimization.
4. **Soft-Core Relaxation**: Applies soft-core potential functions during stepwise energy minimization to resolve steric clashes (e.g., ring spearing) without altering the topolgy graph.
