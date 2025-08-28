import numpy as np
import mdtraj as md
from natsort import natsorted
from glob import glob
import os
from argparse import ArgumentParser

# TODO: fix for several trajectories, 
# TODO: select part for the trajectory change
# TODO: Output filename
def convert_npy_to_xtc(npy_files_path: str, 
                       topology_file: str, 
                       stride: int,
                       traj: int):
    """
    Convert a .npy file containing coordinates to an .xtc file.

    Parameters:
    npy_file (str): Path to the input .npy file.
    topology_file (str): Path to the topology file (e.g., .pdb or .gro).
    """
    print(f"Loading coordinates...")
    
    npy_files = natsorted(glob(f"{npy_files_path}/*_coords_*.npy"))

    coordinates = [np.load(npy_file)[traj].squeeze()/10 for npy_file in npy_files] # TODO: check this conversion!!! to nm
    coordinates = np.vstack(coordinates).squeeze()[::stride]
    print(coordinates.shape)
    topology = md.load(topology_file).topology
    trajectory = md.Trajectory(xyz=coordinates, topology=topology)
    xtc_file = f"sims_coords_traj{traj}.xtc"

    print(f"Saving trajectory to {xtc_file}...")
    trajectory.save_xtc(xtc_file)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--npy-files-path", "-npypath", type=str, help="display a square of a given number",
                        )
    parser.add_argument("--topology-file", "-topfn", type=str, help="display a square of a given number",
                        )
    parser.add_argument("--stride", "-stride", type=int, help="display a square of a given number",
                        )
    parser.add_argument("--traj", "-traj", type=int, help="trajectory number",
                        )
    
    args = parser.parse_args()
    #TODO: Fix for several copies
    print(args.npy_files_path, args.topology_file, args.stride, args.traj)
    convert_npy_to_xtc(args.npy_files_path, args.topology_file, args.stride, args.traj)
