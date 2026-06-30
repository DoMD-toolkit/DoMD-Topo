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

The DoMD-Topo pipeline processes three distinct input components to bridge the gap between coarse-grained configurations and full-atom topologies.

### 2.1 Coarse-Grained Configuration (XML File)
The `.xml` file (GALAMOST format) stores the baseline physical architecture of the coarse-grained (CG) system. It must explicitly define:
* **Positions**: The 3D coordinates ($x, y, z$) of each CG bead.
* **Types**: The identity/label of each bead type (matching the keys in `mols_config`).
* **Bonds**: The structural connectivity matrix defining which beads are linked together.

### 2.2 Monomer Mapping Matrix (`mols_config`)
This dictionary establishes the **Node Identity** by mapping each CG bead type name to its corresponding all-atom SMILES descriptor.
* *Example*: `{'A': {'smiles': 'Nc1ccc(Oc2ccc(N)cc2)cc1', 'file': None}}`
* *Significance*: This provides complete resolution independence. You can map a whole monomer to a single bead or split it into separate backbone and side-chain beads simply by adjusting the target SMILES strings, isolating the user from complex topology templates (e.g., `.itp`, `.rtp`).

### 2.3 Reaction Rules & Atom Labeling (`reaction_template`)
This section utilizes reaction SMARTS as a chemical language for subgraph matching and graph modification to define exactly how beads connect.

#### Critical Concept: Atom Labeling / Numerical Mapping
To correctly govern the connection logic, the reaction SMARTS must use explicit numerical mapping for reactive sites:

* **Syntax (Labeled vs. Unlabeled Atoms)**: 
  In a reaction SMARTS string like `[#7H2:1].[#6:3](=[#8:4])...`, the numerical identifiers (e.g., `:1`, `:3`) are critical. They define the **active atoms** that will explicitly undergo bond breaking or bond formation. 
  Any *unlabeled atoms* in the SMARTS string serve strictly as the required "chemical context" (e.g., specifying that a reactive carbon must be part of a carbonyl group), ensuring they are matched but left completely unmodified.
* **Function**: 
  This explicit labeling mathematically enforces proper **regioselectivity** (such as Head-to-Tail alignment), **stereochemistry** (cis/trans controls), and **branching logic** during both the macroscopic CG assembly and the full-atom backmapping phases.
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
