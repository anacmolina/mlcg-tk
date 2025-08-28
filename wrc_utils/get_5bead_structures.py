import os
import sys
from tqdm import tqdm
MLCG_TK_ROOT = "/local_scratch/am7078fu/packages/mlcg-tk"
sys.path.insert(0, (MLCG_TK_ROOT))

import mdtraj as md
from typing import List, Callable, Optional
from input_generator.embedding_maps import (
    CGEmbeddingMap,
)
from input_generator.raw_data_loader import DatasetLoader
from input_generator.raw_dataset import RawDataset

from jsonargparse import CLI
from time import ctime

from numpy.random import default_rng

# TODO: refactor to get structure from xtc 

def get_5bead_structure(
        dataset_name: str,
        names: List[str],
        pdb_template_fn: str,
        sample_loader: DatasetLoader,
        tag: str,
        save_dir: str,
        cg_atoms: List[str],
        embedding_map: CGEmbeddingMap,
        embedding_func: Callable,
        skip_residues: List[str],
        stride: int,
        mol_num_batches: int,
        save_md_pdbs: bool,
        nframes_per_batch: Optional[int],
        raw_data_dir: Optional[str]        
):
        if save_md_pdbs is False: 
               mol_num_batches=1
        else:
                if os.path.isdir(f"{save_dir}/structures"): 
                       pass
                else: 
                       os.mkdir(f"{save_dir}/structures")     
               
        dataset = RawDataset(dataset_name, names, tag, n_batches=mol_num_batches)

        for samples in tqdm(dataset, f"Processing CG data for {dataset_name} dataset..."):
    
                samples.input_traj, samples.top_dataframe = sample_loader.get_traj_top(
                        samples.mol_name, pdb_template_fn
                )
    
                samples.apply_cg_mapping(
                        cg_atoms=cg_atoms,
                        embedding_function=embedding_func,
                        embedding_dict=embedding_map,
                        skip_residues=skip_residues,
                )

                cg_xyz = samples.input_traj.atom_slice(samples.cg_atom_indices).xyz 
                cg_traj = md.Trajectory(cg_xyz, md.Topology.from_dataframe(samples.cg_dataframe))
                cg_traj.save_pdb(f"{save_dir}/{dataset_name}_5bead_structure.pdb")

                if save_md_pdbs:
                        aa_coords = sample_loader.load_coords_forces(
                                        raw_data_dir,
                                        samples.mol_name,
                                        stride=stride,
                                        batch=samples.batch,
                                        n_batches=samples.n_batches,
                                        load_forces=False,
                ) / 10.0
                        top = md.load(pdb_template_fn)[0].topology
                        total_frames = aa_coords.shape[0]
                        rng = default_rng()
                        sel_frames = rng.choice(total_frames, size=nframes_per_batch, replace=False)

                        for frame in sel_frames:

                                cg_xyz = aa_coords[frame, :, :]
                                cg_traj = md.Trajectory(cg_xyz, top)
                                cg_traj.save_pdb(f"{save_dir}/structures/{dataset_name}_traj_{samples.batch}_frame_{frame}.pdb") 

if __name__ == "__main__":
    print("Start gen_input_data.py: {}".format(ctime()))

    CLI([get_5bead_structure])

    print("Finish gen_input_data.py: {}".format(ctime()))
