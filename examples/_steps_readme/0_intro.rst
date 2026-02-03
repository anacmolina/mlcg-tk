Input Training Data Generation Pipeline
=======================================

How To
------

To prepare simulation data for eventual use in training a transferable coarse grained (CG)
force field, these data must first be mapped to the CG resolution and then processed to
incorporate prior energy terms.

This is done as follows:

1. First, in ``mlcg-tk-gen_input_data`` the atomistic simulations are loaded and mapped to
   the lower resolution, followed by the construction of the neighbor lists specific to
   the chosen prior energy model.
2. Next, in the case that a prior model has not already been generated, this can be done
   by accumulating the statistics of the saved CG data and fitting these statistics to a
   set of predefined energy functions.
3. Finally, the outputs are fed to ``mlcg-tk-produce_delta_forces`` along with the prior
   model so that delta forces can be calculated.

The result is a set of CG coordinates, embeddings, and delta forces which can be used to
train a neural network as well as the neighbor lists associated with each molecule which
can be fitted to produce new prior models, if needed.

In the details below, input files are provided as an example.

Example: 1LY2 toy dataset
-------------------------

To exemplify the usage of this code, we provide an example on how to load a tiny dataset
of All-Atom (AA) simulations of a trpcage-variant, 1L2Y.


This example is just to demonstrate the usage of the package to transport from AA
simulations to the input for training an MLCG model. This dataset does not contain enough
points, nor is it in the right distribution, so that it would create a good CG model.

The dataset is provided under the folder ``./demo_raw_data/``. It contains a readme with
details of the simulation.

0) Create a directory to output all the intermediate files
==========================================================

The commands will output a lot of data and its better to save it in a separate directory
to ensure hygenic file management.

::

   mkdir ./demo_processed_data
