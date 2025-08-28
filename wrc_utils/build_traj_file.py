import numpy as np
import mdtraj as md
from natsort import natsorted
from glob import glob
from typing import Optional
from time import ctime
from jsonargparse import CLI
from tqdm import tqdm

def convert_npy_to_xtc(npy_fns_path: str, 
                       topology_fn: str, 
                       stride: int,
                       save_fn: str,
                       ntraj: Optional[int] = None
                       ):
    """
    Convert a .npy file containing coordinates to an .xtc file.

    Parameters:
    npy_fns_path (str): Path to the input .npy files.
    topology_fn (str): Path to the topology file (e.g., .pdb or .gro).
    stride (int): Interval by which to stride loaded data.
    save_fn (str): Save output filename.
    ntraj (int, optional): Choose one trajectory if the .npy files has more than one.
    """
    print(f"Loading npy files...")
    
    npy_fns = natsorted(glob(f"{npy_fns_path}/*_coords_*.npy"))

    coords = []

    for npy_fn in tqdm(npy_fns):
        xyz = np.load(npy_fn)

        if xyz.shape[0] == 4:

            assert ntraj is not None, "Number of trajectory (ntraj) must be defined and cannot be None."

            xyz = xyz[ntraj].squeeze() / 10 # from angs to nm
            coords.append(xyz)

        else:

            xyz = xyz.squeeze() / 10 # from angs to nm
            coords.append(xyz)

    coords = np.vstack(coords).squeeze()
    coords = coords[::stride]

    topology = md.load(topology_fn).topology
    trajectory = md.Trajectory(xyz=coords, topology=topology)

    print(f"Saving trajectory to {save_fn}...")
    trajectory.save_xtc(save_fn)

if __name__ == "__main__":

    print("Start build_traj_file.py: {}".format(ctime()))

    CLI([convert_npy_to_xtc])

    print("Finish build_traj_file.py: {}".format(ctime()))


    
    
