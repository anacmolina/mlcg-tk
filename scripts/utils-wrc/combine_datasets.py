import os.path as osp
import sys

SCRIPT_DIR = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(SCRIPT_DIR, "../.."))

import h5py
import yaml
from time import ctime
from jsonargparse import CLI
from typing import List, Optional
from input_generator.utils import get_output_tag

def combine_datasets(
    dataset_names: List[str],
    save_dir: str,
    force_tag: Optional[str],
    save_h5: Optional[bool] = True,
    save_partition: Optional[bool] = True,
    new_name: Optional[str] = None,
    ):

    """
    Computes structural features and accumulates statistics on dataset samples

    Parameters
    ----------
    dataset_names : List[str]
        List of dataset name to combine
    save_dir : str
        Path to directory from which datasets will be loaded and to which output will be saved
    force_tag : str
        Label given to produced delta forces and saved packaged data
    save_h5 : bool
        Whether to save dataset h5 file(s)
    save_partition : bool
        Whether to save dataset partition file(s)
    new_name: str
        New name for the dataset and partition file
    """

    if new_name is None:
        datasets_label = "_".join(dataset_names)
        output_tag = get_output_tag([datasets_label, force_tag], placement="after")
    else:
        output_tag = get_output_tag([new_name, force_tag], placement="after")

    if save_h5:
        fnout_h5 = osp.join(save_dir, f"combined{output_tag}.h5")
        
        with h5py.File(fnout_h5, "w") as f:

            for dataset in dataset_names:
                f.create_group(dataset)
                for replica in ["replica1", "replica2", "replica3"]:
                    fn = f'{save_dir}/{dataset}_{replica}_cgschnet.h5'
                    f[dataset][f'{dataset}_{replica}'] = h5py.ExternalLink(fn, f'/{dataset}_{replica}/{dataset}_{replica}')

            f.close()

    if save_partition:
        fnout_part = osp.join(save_dir, f"partition{output_tag}.yaml")

        partition_opts = {"train": {}, "val": {}}
        partition_opts["train"]["metasets"] = {}
        partition_opts["train"]["batch_sizes"] = {}
        partition_opts["val"]["metasets"] = {}
        partition_opts["val"]["batch_sizes"] = {}

        for dataset in dataset_names:
            data_fn = osp.join(
                save_dir,
                f"partition_{dataset}{get_output_tag(force_tag, placement='after')}.yaml",
            )
            with open(data_fn, "r") as ifile:
                data_partition = yaml.safe_load(ifile)

            # make training data partition
            partition_opts["train"]["metasets"][dataset] = data_partition["train"][
                "metasets"
            ][dataset]
            partition_opts["train"]["batch_sizes"] = {
                dataset: data_partition["train"]["batch_sizes"]
            }

            # make validation data partition
            partition_opts["val"]["metasets"][dataset] = data_partition["val"][
                "metasets"
            ][dataset]
            partition_opts["val"]["batch_sizes"] = {
                dataset: data_partition["val"]["batch_sizes"]
            }

        with open(fnout_part, "w") as ofile:
            yaml.dump(partition_opts, ofile)

if __name__ == "__main__":
    print("Start combine_datasets.py: {}".format(ctime()))

    dataset_names = ['WT', 'A455P']
    save_dir = "/net/scratch-sheldon/am7078fu/projects/5beads-WRC/processed_data"
    force_tag = "cgschnet"
    #save_h5: Optional[bool] = True,
    #save_partition: Optional[bool] = True,
    new_name = "WRC"

    combine_datasets(
        dataset_names=dataset_names,
        save_dir=save_dir,
        force_tag=force_tag,
        new_name=new_name,
    )

    # CLI([combine_datasets], as_positional=False)

    print("Finish combine_datasets.py: {}".format(ctime()))
