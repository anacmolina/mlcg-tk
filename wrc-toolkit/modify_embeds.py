import numpy as np
from glob import glob
from natsort import natsorted
from jsonargparse import CLI
from time import ctime

def modify_embeds(
        embeds_fn: str,
        select_beads: list,
        new_embeds_beads: list,
    ):

    cg_embeds = np.load(embeds_fn)
    cg_embeds[select_beads] = new_embeds_beads

    np.save(embeds_fn, cg_embeds)
    print(f"Modified embeds saved in {embeds_fn}!")


def process_embeds_batch(
        save_dir: str,
        tag: str,
        select_beads: list,
        new_embeds_beads: list,
    ):

    fns = natsorted(glob(f"{save_dir}/*{tag}.npy"))

    for fn in fns:
        modify_embeds(
            embeds_fn=fn,
            select_beads=select_beads,
            new_embeds_beads=new_embeds_beads
        )

def main():
    print("Start gen_input_data.py: {}".format(ctime()))
    CLI([modify_embeds, process_embeds_batch])
    print("Finish gen_input_data.py: {}".format(ctime()))


if __name__ == "__main__":
    main()