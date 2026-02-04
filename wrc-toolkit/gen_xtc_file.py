from glob import glob
from typing import Optional
from natsort import natsorted
from time import ctime
from tqdm import tqdm
from jsonargparse import CLI

import numpy as np
import mdtraj as md

def create_xtc_file(npy_fns_path: str,
                    topology_fn: str, 
                    stride: int,
                    save_fn: str,
                    pattern_fn: Optional[str] = None,
                    itraj: Optional[int] = None
                    ):
    """
    Convert a .npy file containing coordinates to an .xtc file.

    Parameters:
    npy_fns_path (str): Path to the input .npy files.
    topology_fn (str): Path to the topology file (e.g., .pdb or .gro).
    stride (int): Interval by which to stride loaded data.
    save_fn (str): Save output filename.
    pattern_fn (str, optional): Add a pattern to identify the .npy files.
    itraj (int, optional): Choose one trajectory if the .npy files has more than one.
    """

    print(f"Looking npy files...")

    if pattern_fn is None:
        npy_fns = natsorted(glob(f"{npy_fns_path}/*_coords_*.npy"))
    else:
        npy_fns = natsorted(glob(f"{npy_fns_path}/{pattern_fn}*_coords_*.npy"))

    assert len(npy_fns) != 0, f"There are no files with: {npy_fns_path}/{pattern_fn}."

    print(f"The number of files found: {len(npy_fns)}")

    coords = []
        
    for npy_fn in tqdm(npy_fns):
        
        xyz = np.load(npy_fn)

        if xyz.shape[0] == 4:
            
            if itraj is None:
                raise ValueError("There is more than one trajectory in the file, therefore, itraj can not be None, please specify the i-th trajectory.")
            else:
                xyz = xyz[itraj].squeeze() / 10 # from angs to nm
                coords.append(xyz)
        else:
            
            xyz = xyz.squeeze() / 10 # from angs to nm
            coords.append(xyz)

    coords = np.vstack(coords).squeeze()
    coords = coords[::stride]

    topology = md.load(topology_fn).topology
    trajectory = md.Trajectory(xyz=coords, topology=topology)

    print(f"Saving trajectory to {save_fn}")
    trajectory.save_xtc(save_fn)

if __name__ == "__main__":

    print("Start gen_xtc_file.py: {}".format(ctime()))

    CLI([create_xtc_file])

    print("Finish gen_xtc_file.py: {}".format(ctime()))