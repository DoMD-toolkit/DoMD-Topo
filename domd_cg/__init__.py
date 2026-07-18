__doc__ = r"""
ChemFAST Coarse-Grained (CG) Modeling Module (`domd_cg`)
=======================================================

The ``domd_cg`` package provides automated pipelines to construct chemically
specific Coarse-Grained (CG) models directly from molecular descriptors (SMILES
strings or all-atom structure files). It automates topological assembly, assigns
force field parameters using machine learning, and generates initial 3D coordinates.

Methodology & Implementation
----------------------------

* **Bead Size (sigma):** Derived geometrically by generating an ensemble of low-energy
  3D conformers via the ETKDG algorithm. The effective diameter is calculated from
  the average maximum radius relative to the center of mass, inflated by a scaling
  factor (default 1.2):

  .. math:: \sigma = \lambda 2 \langle R_{\text{max}} \rangle

* **Interaction Strength (epsilon):** Predicted using three independent Graph Attention
  Networks (GAT) trained on the HSPiP dataset to output Hansen Solubility Parameter (HSP)
  components: dispersion, polar, and hydrogen-bonding. The Cohesive Energy Density (CED)
  is calculated as:

  .. math:: \text{CED} = \delta_{\text{tot}}^2 = \delta_D^{2} + \delta_P^{2} + \delta_H^{2}

* **Topological Assembly Pathways:**

  1. *In Situ Polymerization (Reactive):* For cross-linked/amorphous networks.
  2. *Algorithmic Construction:* Generates linear, block, or star architectures via SARW.
  3. *Post-hoc Reconstruction (BFS):* Infers connectivity matrices from bead snapshots.
  4. *Rigid-Body Protocol:* Discretizes high-resolution structures (PDB) into rigid point clouds.

Domain of Validity & Limitations
--------------------------------

+-------------------------------------+------------------------------------------+
| Supported Systems (High Validity)   | Restricted Systems (Physical Limitations)|
+=====================================+==========================================+
| * Synthetic homopolymers/copolymers | * Crystalline Carbohydrates              |
| * Amorphous polymer networks        | * Sequence-Specific Proteins/IDPs        |
| * Organic solvents and mixtures     | * Complex Coacervation (Electrostatics)  |
+-------------------------------------+------------------------------------------+

Core APIs 
-----------------
* :func:`HSP_predictor`: Predicts HSP components from SMILES strings.
* :func:`predict_cg_params`: Derives force field parameters for a reacting chemical pool.
* :func:`create_cg_system`: Orchestrates the complete CG system creation pipeline.
"""

from .misc.pipeline import create_cg_system, predict_cg_params, HSP_predictor

__all__ = [
    "create_cg_system",
    "predict_cg_params",
    "HSP_predictor"
]
