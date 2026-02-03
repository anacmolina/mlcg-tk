import os as os
import numpy as np
import h5py as h5
from glob import glob
from jsonargparse import CLI
from time import ctime

def build_dummy_embeds(embeds_fn: str
        ):
    
    name_fn = embeds_fn.split('.')[0]
    real_fn = f"{name_fn}_real.npy"    

    real_embeds = np.load(embeds_fn)

    if os.path.isfile(real_fn) is False:
        print("Saving real embeds copy...")
        np.save(real_fn, real_embeds)
        print(f"Done, save as: {real_fn}")

    dummy_embeds = real_embeds[:400]

    print("Saving dummy embeds...")
    np.save(embeds_fn, dummy_embeds)
    print(f"Done, save as: {embeds_fn}")

    

def add_embeds_to_h5(dataset_name: str,
                    data_dir: str
                    ):

    h5_fn = glob(f"{data_dir}/{dataset_name}_*.h5")
    assert len(h5_fn) == 1, "File .h5 file not found"
    h5_fn = h5_fn[0]

    embeds_fn = glob(f"{data_dir}/{dataset_name}_*real.npy")
    assert len(embeds_fn) == 1, "More than one embeds files were found"
    embeds_fn = embeds_fn[0]

    print(h5_fn, embeds_fn)

    with h5.File(h5_fn,"a") as f:

        real_embeds = np.load(embeds_fn)
        del f[dataset_name][dataset_name].attrs['cg_embeds']

        f[dataset_name][dataset_name]['cg_embeds'] = real_embeds

def recover_real_embeds(embeds_fn: str
                        ):
    
    real_embeds = np.load(embeds_fn)
    embeds_new_fn = f"{embeds_fn.split("_real.npy")[0]}.npy"
    np.save(embeds_new_fn, real_embeds)
    print(f"Saving real embeds copy to: {embeds_new_fn}")

if __name__ == "__main__":
    print("Start use_dummy_embeds.py: {}".format(ctime()))

    CLI([build_dummy_embeds, add_embeds_to_h5, recover_real_embeds])

    print("Finish use_dummy_embeds.py: {}".format(ctime()))