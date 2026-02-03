from input_generator.prior_gen import PriorBuilder
from typing import List
from copy import deepcopy
from tqdm import tqdm
import pickle as pkl


def combine_prior_builders(
        list_prior_builders: List[PriorBuilder],
        save_fn: str
        ):
    
    combined_prior_builder = deepcopy(list_prior_builders[0])

    for i, new_prior_builder in enumerate(combined_prior_builder):
        
        name = new_prior_builder.name

        print(f"Merging histogram for {name}")

        for old_prior_builder in list_prior_builders[1:]:
    
            for key in tqdm(new_prior_builder.histograms[name].keys()):

                new_prior_builder.histograms[name][key] += old_prior_builder[i].histograms[name][key]

    with open(save_fn, "wb") as ofile:
        pkl.dump(combined_prior_builder, ofile)

    return combined_prior_builder