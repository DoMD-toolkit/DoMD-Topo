from __future__ import annotations

import copy
import json
import logging
import pickle
import zipfile
from pathlib import Path
from typing import Any

from domd.conf import topology_builder, write_mols_to_sdf, embed_molecules
from misc.parser import parse_config, post_process_aa_mol
from domd.ff.misc.logger import task_file_log_scope


# {'app_name': 'domd_topo', 'task_id': 'a80e59676d02471580ff204fdbd56407',
# 'task_ref': 'domd_topo:a80e59676d02471580ff204fdbd56407',
# 'workspace_base': 'D:\\DoMD-Toolkit\\workspaces',
# 'work_dir': 'D:\\DoMD-Toolkit\\workspaces\\domd_topo\\a80e59676d02471580ff204fdbd56407',
# 'file_paths': ['D:\\DoMD-Toolkit\\workspaces\\domd_topo\\a80e59676d02471580ff204fdbd56407\\poss.zip'],
# 'params': {'mode': 'sdf'}, 'created_at': 1784016959}


def parse_payload(payload: dict[str, Any]):
    work_dir = Path(payload["work_dir"])
    params = payload.get("params")
    mode = params.get("mode")
    assert mode in ['sdf', 'top_only', 'xyz_only'], ValueError("MODE MUSE BE `sdf`, `top_only` OR `xyz_only`!")
    path = Path(payload['file_paths'][0])
    if not path.is_file() or path.suffix != '.zip':
        raise FileNotFoundError("FILE NOT FOUND OR FILE IS NOT ZIP!")
    parent_dir = path.parent
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(parent_dir)
    config_file = work_dir / "config.json"
    if not config_file.is_file():
        raise FileNotFoundError("`config.json` FILE NOT FOUND IN ZIP FILE!")
    config = parse_config(json.loads(config_file.read_text()), work_dir)
    return config


def log_params(logger: logging.Logger, params: dict[str, Any]) -> None:
    logger.info(f"Run mode: `{params['mode']}`.")


def run_domd_topo(payload: dict[str, Any], logger: logging.Logger, results_dir: Path) -> str:
    """Generate OPLS AutoFF output files. Packaging belongs to redis_wkr."""
    task_id = str(payload["task_id"])
    params = payload["params"]

    with task_file_log_scope(
            task_name=f"domd_topo_{task_id}",
            log_dir=str(results_dir),
    ):
        logger.info("Starting DoMD-Topo task `%s`.", task_id)
        log_params(logger, params)
        config = parse_payload(payload)

        mol_list = []
        gra_list = []
        logger.info(f"Processing Total {len(config.cg_graphs):d} molecules.")
        for cg_mol, rxn_lst in zip(config.cg_graphs, config.reaction_list):
            aa_mol, aa_graph = topology_builder(config.reactant_config, config.reaction_template,
                                                rigid_config=config.rigid_config, cg_graph=cg_mol,
                                                reaction_list=rxn_lst, fast_sanitize_p=params['fastSanit'])

            aa_mol = post_process_aa_mol(aa_mol, aa_graph, config.box_tensor)
            mol_list.append(copy.deepcopy(aa_mol))
            gra_list.append(copy.deepcopy(aa_graph))

        if params['mode'] == 'sdf':
            logger.info(f"Embedding {len(config.cg_graphs):d} molecules.")
            mol_list = embed_molecules(mol_list, gra_list, config, chunk_per_d=params.get('chunks_per_d', 1))
            write_mols_to_sdf(mol_list, str(results_dir / 'system.sdf'), force_v3000=True)

        if params['mode'] == 'top_only':
            pickle.dump(mol_list, open(str(results_dir / 'system.pkl'), 'wb'))
