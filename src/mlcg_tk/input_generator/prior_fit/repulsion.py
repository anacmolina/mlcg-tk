import torch
from typing import Dict, Optional
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
import numpy as np

import pygad


def repulsion(x, sigma):
    """Method defining the repulsion interaction"""
    rr = (sigma / x) * (sigma / x)
    return rr * rr * rr


def fit_repulsion_from_potential_estimates(
    bin_centers_nz: torch.Tensor, **kwargs
) -> Dict:
    r"""Method for fitting interaction parameters from data

    Parameters
    ----------
    bin_centers:
        Bin centers from a discrete histgram used to estimate the energy
        through logarithmic inversion of the associated Boltzmann factor
    dG_nz:
        The value of the energy :math:`U` as a function of the bin
        centers, as retrieved via:

        :math:`U(x) = -\frac{1}{\beta}\log{ \left( p(x)\right)}`

        where :math:`\beta` is the inverse thermodynamic temperature and
        :math:`p(x)` is the normalized probability distribution of
        :math:`x`.


    Returns
    -------
    Dict:
        Dictionary of interaction parameters as retrived through
        `scipy.optimize.curve_fit`
    """

    delta = bin_centers_nz[1] - bin_centers_nz[0]
    sigma = bin_centers_nz[0] - 0.5 * delta
    stat = {"sigma": sigma}
    return stat


def fit_repulsion_from_values(
    bin_centers_nz: torch.Tensor,
    ncounts_nz: torch.Tensor,
    percentile: float,
    cutoff: Optional[float] = None,
    **kwargs,
) -> Dict:
    """Method for fitting interaction parameters directly from input features

    Parameters
    ----------
    values:
        Input features as a tensor of shape (n_frames)
    percentile:
        If specified, the sigma value is calculated using the specified
        distance percentile (eg, percentile = 1) sets the sigma value
        at the location of the 1th percentile of pairwise distances. This
        option is useful for estimating repulsions for distance distribtions
        with long lower tails or lower distance outliers. Must be a number from
        0 to 1
    cutoff:
        If specified, only those input values below this cutoff will be used in
        evaluating the percentile

    Returns
    -------
    Dict:
        Dictionary of interaction parameters as retrived through
        `scipy.optimize.curve_fit`
    """
    values = np.repeat(bin_centers_nz.numpy(), ncounts_nz.int().numpy())
    if cutoff != None:
        values = values[values < cutoff]
    sigma = torch.tensor(np.percentile(values, percentile))
    stat = {"sigma": sigma}
    return stat

def exp_repulsion(x, alpha, r_0):
    """Method defining the repulsion interaction"""
    rr = 1 - (x / r_0)
    return (6 / alpha) * np.exp( alpha * rr)

def interpolation_crossover(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        new_offspring = 0.5*(parent1+parent2)
        offspring.append(new_offspring)

        idx += 1

    return np.array(offspring)

def fit_exp_repulsion_using_genetic_algorithm(
    bin_centers_nz: torch.Tensor, 
    dG_nz: torch.Tensor,
    #TODO: Fix this parameter, it is not necessary
    ncounts_nz: torch.Tensor, 
    repulsion_function: callable=exp_repulsion,
    iters:int=500
) -> Dict:
    
    integral = torch.tensor(
        float(trapezoid(dG_nz.cpu().numpy(), bin_centers_nz.cpu().numpy()))
    )
    mask = torch.abs(dG_nz) > 1e-8 * torch.abs(integral)
    xs = bin_centers_nz[mask],
    ys = dG_nz[mask],

    def fitness_func(ga_instance, solution, solution_idx):
        new_ys = repulsion_function(xs,solution[0],solution[1])
        fitness = 1.0 / np.linalg.norm(ys-new_ys)
        return fitness
    
    num_generations = iters
    num_parents_mating = 20
    fitness_function = fitness_func
    sol_per_pop = 80
    num_genes = 2
    init_range_low = 1
    init_range_high = 10
    parent_selection_type = "sss"
    keep_parents = 0

    mutation_type = "random"
    mutation_percent_genes = 50
    ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=interpolation_crossover,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       parallel_processing=None)
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    stat = {"alpha": solution[0], "r_0": solution[1]}

    return stat