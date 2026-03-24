import os
import sys
from tqdm import tqdm

import mdtraj as md
from typing import List, Callable, Optional
from input_generator.embedding_maps import (
    CGEmbeddingMap,
)
from mlcg_tk.input_generator.raw_data_loader import DatasetLoader
from mlcg_tk.input_generator.raw_dataset import RawDataset
from mlcg_tk.input_generator.utils import map_cg_topology

from jsonargparse import CLI
from time import ctime
from copy import deepcopy

from numpy.random import default_rng

# TODO: More general way, still work on progress
def get_pdb_from_xtc(
        trajectory_fn: str,
        pdb_template_fn: str,
        frames_list: List[int],
        stride: int,
        cg_atoms: List[str],
        embedding_function: CGEmbeddingMap,
        skip_residues: List[str],
        save_dir: str              
):
        """
        Description:
        
        """
        
        traj = md.load_xtc(filename=trajectory_fn, 
                           top=pdb_template_fn, 
                           stride=stride)
        
        pdb = md.load(pdb_template_fn)

        aa_traj = pdb.atom_slice(
            [a.index for a in pdb.topology.atoms if a.residue.is_protein]
        )

        top_dataframe = aa_traj.topology.to_dataframe()[0]
        top_dataframe = top_dataframe.apply(
            map_cg_topology,
            axis=1,
            cg_atoms=cg_atoms,
            embedding_function=embedding_function,
            skip_residues=skip_residues,
        )

        cg_df = deepcopy(top_dataframe.loc[top_dataframe["mapped"] == True])
        cg_atom_idx = cg_df.index.values.tolist()
        cg_traj = traj.atom_slice(cg_atom_idx)

        for frame in frames_list:
               cg_traj.save_pdb(f"{save_dir}/{trajectory_fn.split(".")[0]}_frame_{frame}.pdb") 


def get_cg_pdb_structures(
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
        embed_tag: str,
        nframes_per_batch: Optional[int],
        raw_data_dir: Optional[str]        
):
        if save_md_pdbs is False: 
               mol_num_batches=1
        else:
                if os.path.isdir(f"{save_dir}"): 
                       pass
                else: 
                       os.mkdir(f"{save_dir}")     
               
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
                cg_traj.save_pdb(f"{save_dir}/{dataset_name}_{embed_tag}_structure.pdb")

                if save_md_pdbs:
                        aa_coords, aa_forces = sample_loader.load_coords_forces(
                                        raw_data_dir,
                                        samples.mol_name,
                                        stride=stride,
                                        batch=samples.batch,
                                        n_batches=samples.n_batches                                        
                ) 
                        aa_coords = aa_coords / 10.0
                        top = md.load(pdb_template_fn)[0].topology
                        total_frames = aa_coords.shape[0]
                        rng = default_rng()
                        sel_frames = rng.choice(total_frames, size=nframes_per_batch, replace=False)

                        for frame in sel_frames:

                                cg_xyz = aa_coords[frame, :, :]
                                cg_traj = md.Trajectory(cg_xyz, top)
                                cg_traj.save_pdb(f"{save_dir}/{dataset_name}_batch_{samples.batch}_frame_{frame}.pdb") 

if __name__ == "__main__":
    print("Start get_cg_pdb_structures.py: {}".format(ctime()))

    CLI([get_cg_pdb_structures])

    print("Finish get_cg_pdb_structures.py: {}".format(ctime()))