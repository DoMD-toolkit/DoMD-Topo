# DoMD-Conf

```text
██████╗  ██████╗ ███╗   ███╗██████╗        ██████╗ ██████╗ ███╗   ██╗███████╗
██╔══██╗██╔═══██╗████╗ ████║██╔══██╗      ██╔════╝██╔═══██╗████╗  ██║██╔════╝
██║  ██║██║   ██║██╔████╔██║██║  ██║█████╗██║     ██║   ██║██╔██╗ ██║█████╗
██║  ██║██║   ██║██║╚██╔╝██║██║  ██║╚════╝██║     ██║   ██║██║╚██╗██║██╔══╝
██████╔╝╚██████╔╝██║ ╚═╝ ██║██████╔╝      ╚██████╗╚██████╔╝██║ ╚████║██║
╚═════╝  ╚═════╝ ╚═╝     ╚═╝╚═════╝        ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝
```

DoMD-Conf reconstructs all-atom (AA) molecular topologies and coordinates from coarse-grained (CG) configurations. It supports flexible molecules, crosslinked networks, and rigid–flexible systems through user-defined reaction templates.

## 1. Online service

1. Prepare one ZIP archive containing `config.json` and all referenced input files.
2. Upload the ZIP archive to the DoMD Topology page.
3. Select **Topology Only** or **SDF Mode** and submit the task.
4. Use the returned Task ID to monitor the task and download the result ZIP.

The Task ID can also be supplied directly to DoMD-FF for subsequent force-field generation.

### 1.1 Input ZIP

All files must be placed together in the ZIP root directory. Do not place them in subdirectories.

```text
input.zip
├── config.json
├── cg_conf.xml              # or cg_conf.gsd
├── reactions.txt            # optional
└── rigid.pdb                # optional; rigid.sdf is also supported
```

#### `config.json`

The configuration contains five main entries:

```json
{
    "cg_topology_file": "cg_conf.xml",
    "reaction_file": "reactions.txt",
    "reactant_config": {},
    "reaction_template": {},
    "rigid_config": {}
}
```

| Entry | Required | Description |
|---|---------:|---|
| `cg_topology_file` |      Yes | CG configuration in XML or GSD format. Systems containing rigid bodies require XML. |
| `reaction_file` |       No | Text file containing the ordered reaction sequence. Users should provide this file whenever the reaction history is known. If omitted, reactions are inferred by BFS, which may produce an incorrect order for complex crosslinked or reaction-order-dependent systems. |
| `reactant_config` |      Yes | AA definitions of flexible CG bead types. |
| `reaction_template` |      Yes | Reaction SMARTS and allowed ordered CG reactant types. |
| `rigid_config` |       No | AA templates and CG-to-AA mappings for rigid bodies. Use `{}` when no rigid body is present. |

#### Flexible reactants

Each flexible CG type must provide at least one molecular source:

```json
{
    "A": {
        "smiles": "C=CC(=O)O",
        "file": null
    }
}
```

`file` may point to a PDB or SDF file. When both `file` and `smiles` are supplied, the file is used.

#### Reaction templates

```json
{
    "B-A": {
        "cg_reactant_list": [["A", "B"]],
        "smarts": "C(=O)[#8H1:1].[#6:2][#8H1:3]>>C(=O)[#8:3][#6:2].[#8:1]",
        "prod_idx": [0]
    }
}
```

| Entry | Description |
|---|---|
| `cg_reactant_list` | Allowed ordered combinations of CG reactant types. |
| `smarts` | RDKit reaction SMARTS. Atom-map labels define atoms retained or modified by the reaction. |
| `prod_idx` | Zero-based indices of products retained after the reaction. |
| `prob` | Optional reaction probability used by workflows that require it. |

#### Rigid bodies

```json
{
    "SiO": {
        "file": "POSS.pdb",
        "mapping": {
            "5": {
                "atom_idx": [0, 32, 33],
                "smarts": "[N:1]([H:2])[H:3]"
            }
        },
        "body_idx": [0]
    }
}
```

| Entry | Description |
|---|---|
| `file` | PDB or SDF file containing the complete AA rigid template. |
| `body_idx` | CG rigid-body IDs represented by this template. |
| `mapping` | Mapping from a CG local node index to an AA reaction fragment. JSON object keys are strings such as `"5"`. |
| `atom_idx` | Ordered zero-based atom indices in the rigid AA template. |
| `smarts` | SMARTS describing the mapped AA fragment. Single-atom and multi-atom fragments are supported. |

Rigid systems require XML CG input because rigid-body IDs and mapped node coordinates are read from the XML topology.

### 1.2 Reaction file

The reaction file contains one Python-style tuple per line:

```text
('B-A', 0, 1)
('A-A', 1, 8)
('A-A', 8, 13)
```

The first item is a reaction name defined in `reaction_template`. The remaining items are CG node indices.

Both forms of ordering are chemically significant:

- Tuple lines are executed from top to bottom.
- Node indices inside each tuple define the ordered reactants and must match an entry in `cg_reactant_list`.

Users must provide the correct reaction history when topology construction depends on reaction order. An incorrect line order or reactant-index order may prevent `topology_builder` from finding the required reaction site.

When `reaction_file` is omitted, DoMD-Conf applies a deterministic BFS traversal as a fallback. BFS is intended to protect against missing input; it may not recover the correct history of complex crosslinked or reaction-order-dependent systems.

### 1.3 Output

The online service returns a ZIP archive containing the selected result.

#### Topology Only

```text
system.pkl
```

`system.pkl` stores:

```python
list[rdkit.Chem.Mol]
```

Each RDKit molecule contains the reconstructed AA topology. Coordinate embedding is not performed in this mode.

#### SDF Mode

```text
system.sdf
```

The SDF contains the list of reconstructed RDKit molecules with AA conformers. Each molecule is stored as one SDF record, and consecutive records are separated by:

```text
$$$$
```

## 2. Core API

### 2.1 `parse_config`

```python
config = parse_config(user_config)
```

Parses the user configuration, reads the CG topology and molecular templates, separates the CG system into molecular graphs, and prepares the reaction lists.

**Input**

| Argument | Type | Description |
|---|---|---|
| `user_config` | `dict` | Dictionary loaded from `config.json`. |

**Output**

| Object | Type | Description |
|---|---|---|
| `config` | `Config` | Parsed configuration containing `reactant_config`, `reaction_template`, `rigid_config`, `cg_graphs`, `reaction_list`, `cg_sys`, and `box_tensor`. |

### 2.2 `topology_builder`

```python
rdmol, aa_graph = topology_builder(
    reactant_config,
    reaction_template,
    rigid_config,
    cg_graph,
    reactions
)
```

Builds the AA topology of one CG molecule by applying its ordered reaction list.

**Inputs**

| Argument | Type | Description |
|---|---|---|
| `reactant_config` | `dict` | Parsed flexible reactant templates. |
| `reaction_template` | `dict` | Parsed reaction SMARTS and CG reactant rules. |
| `rigid_config` | `dict` | Parsed rigid AA templates; use `{}` when no rigid body is present. |
| `cg_graph` | `networkx.Graph` | One molecular CG graph from `config.cg_graphs`. |
| `reactions` | `list[tuple]` | Ordered reactions corresponding to `cg_graph`. |

**Outputs**

| Object | Type | Description |
|---|---|---|
| `rdmol` | `rdkit.Chem.Mol` | Reconstructed AA molecular topology. |
| `aa_graph` | `networkx.Graph` | AA graph containing the atom-to-CG mapping and molecular metadata used during embedding. |

### 2.3 `embed_molecules`

```python
embedded_mols = embed_molecules(rdmols, aa_graphs, config, chunk_per_d=1)
```

Generates and aligns AA conformers for a list of reconstructed molecules using the corresponding CG coordinates.

**Inputs**

| Argument | Type | Description |
|---|---|---|
| `rdmols` | `list[Chem.Mol]` | AA molecules returned by `topology_builder`. |
| `aa_graphs` | `list[nx.Graph]` | AA graphs corresponding one-to-one with `rdmols`. |
| `config` | `Config` | Parsed configuration containing `cg_graphs`, `rigid_config`, and `box_tensor`. |
| `chunk_per_d` | `int` | Number of spatial subdivisions along each Cartesian dimension during chunk-based orientation optimization. Increasing it reduces the maximum local optimization size and memory demand for large connected systems, while adding decomposition overhead. Default: `1`. |

**Output**

| Object | Type | Description |
|---|---|---|
| `embedded_mols` | `list[Chem.Mol]` | Molecules with reconstructed AA conformers in the original list order. |

## 3. Examples

### 3.1 SPE network

This example contains a flexible crosslinked polymer electrolyte network and separate small molecular or ionic components.

```text
spe_network.zip
├── config.json
├── cg_conf.xml
├── reactions.txt
└── run.py
```

#### `config.json`

```json
{
    "cg_topology_file": "cg_conf.xml",
    "reaction_file": "reactions.txt",
    "reactant_config": {
        "A": {"smiles": "C=CC(=O)O", "file": null},
        "B": {"smiles": "OCCCC", "file": null},
        "C": {"smiles": "OCCOCCOCCOCCO", "file": null},
        "L": {"smiles": "[Li+]", "file": null},
        "T": {"smiles": "C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)(F)", "file": null},
        "S": {"smiles": "N#CCCC#N", "file": null}
    },
    "reaction_template": {
        "A-A": {
            "cg_reactant_list": [["A", "A"]],
            "smarts": "[C:1]=[C:2]C(=O)O.[C:3]=CC(=O)O>>[C:1][C:2](C(=O)O)[C:3]=CC(=O)O",
            "prod_idx": [0]
        },
        "B-A": {
            "cg_reactant_list": [["A", "B"]],
            "smarts": "C(=O)[#8H1:1].[#6:2][#8H1:3]>>C(=O)[#8:3][#6:2].[#8:1]",
            "prod_idx": [0]
        },
        "C-A": {
            "cg_reactant_list": [["A", "C"]],
            "smarts": "C(=O)[#8H1:1].[#6:2][#8H1:3]>>C(=O)[#8:1][#6:2].[#8:3]",
            "prod_idx": [0]
        },
        "S": {
            "cg_reactant_list": [["L"], ["T"], ["S"]],
            "smarts": "*>>*",
            "prod_idx": [0]
        }
    },
    "rigid_config": {}
}
```

#### `reactions.txt` excerpt

The full file preserves the reaction sequence recorded during CG construction. A shortened excerpt is shown below:

```text
('B-A', 0, 1)
('B-A', 2, 3)
('C-A', 200, 201)
('A-A', 676, 1896)
('A-A', 1896, 1320)
```

#### `run.py`

```python
import json

from embed_molecule import embed_molecules
from misc.io.sdf import write_mols_to_sdf
from misc.parser import parse_config
from pipeline import topology_builder

user_config = json.load(open('config.json', 'r'))
config = parse_config(user_config)
rdmols, aa_graphs = [], []

for cg_graph, reactions in zip(config.cg_graphs, config.reaction_list):
    rdmol, aa_graph = topology_builder(config.reactant_config, config.reaction_template,
                                       config.rigid_config, cg_graph, reactions)
    rdmols.append(rdmol)
    aa_graphs.append(aa_graph)

rdmols = embed_molecules(rdmols, aa_graphs, config, chunk_per_d=1)
write_mols_to_sdf(rdmols, 'spe_network.sdf', force_v3000=True)
```

### 3.2 POSS-grafted polymer

This example contains one POSS rigid body with four mapped amine reaction sites and four grafted PMMA chains. Its simple reaction tree is inferred by the BFS fallback, so no reaction file is required.

```text
poss_graft.zip
├── config.json
├── cg_conf.xml
├── POSS.pdb
└── run.py
```

#### `config.json`

```json
{
    "cg_topology_file": "cg_conf.xml",
    "reaction_template": {
        "P-P": {
            "cg_reactant_list": [["P", "P"]],
            "smarts": "[C:1][CH1:2].[CH3:3]>>[C:1][C:2][C:3]",
            "prod_idx": [0]
        },
        "CN-A": {
            "cg_reactant_list": [["CN", "A"]],
            "smarts": "[N:1][H:5].[OH1:2][C:3]=[O:4]>>[N:1][C:3]=[O:4].[O:2][H:5]",
            "prod_idx": [0]
        },
        "A-P": {
            "cg_reactant_list": [["A", "P"]],
            "smarts": "[CH2:1][CH1:2].[CH3:3]>>[C:1][C:2][C:3]",
            "prod_idx": [0]
        }
    },
    "reactant_config": {
        "P": {"smiles": "CC(C(=O)OC)C"},
        "A": {"smiles": "OC(=O)CC(C(=O)OC)C"},
        "CN": {"smiles": "[N:1]([H:2])[H:3]"}
    },
    "rigid_config": {
        "SiO": {
            "file": "POSS.pdb",
            "mapping": {
                "5": {"atom_idx": [0, 32, 33], "smarts": "[N:1]([H:2])[H:3]"},
                "6": {"atom_idx": [23, 38, 39], "smarts": "[N:1]([H:2])[H:3]"},
                "7": {"atom_idx": [28, 51, 52], "smarts": "[N:1]([H:2])[H:3]"},
                "8": {"atom_idx": [31, 58, 59], "smarts": "[N:1]([H:2])[H:3]"}
            },
            "body_idx": [0]
        }
    }
}
```

#### `run.py`

```python
import json

from embed_molecule import embed_molecules
from misc.io.sdf import write_mols_to_sdf
from misc.parser import parse_config
from pipeline import topology_builder

user_config = json.load(open('config.json', 'r'))
config = parse_config(user_config)
rdmols, aa_graphs = [], []

for cg_graph, reactions in zip(config.cg_graphs, config.reaction_list):
    rdmol, aa_graph = topology_builder(config.reactant_config, config.reaction_template,
                                       config.rigid_config, cg_graph, reactions)
    rdmols.append(rdmol)
    aa_graphs.append(aa_graph)

rdmols = embed_molecules(rdmols, aa_graphs, config, chunk_per_d=1)
write_mols_to_sdf(rdmols, 'POSS-g-PMMA.sdf', force_v3000=True)
```