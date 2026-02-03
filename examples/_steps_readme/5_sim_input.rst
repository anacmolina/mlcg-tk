5) Generate simulation input
============================

Command::

   mlcg-tk-gen_sim_input process_sim_input --config configuration_files/trpcage_sim.yaml --config configuration_files/trpcage_priors.yaml


A trained MLCG model serves as a forcefield for conducting protein simulations. To run
simulations of a particular system, the command above will process each structure file
indicated by the ``pdb_fns`` option, map these to the specified CG resolution, generate
neighbor lists corresponding the the given ``prior_builders``, and save the specified
number of copies of ``AtomicData`` objects storing this information.

In contrast to traditional MD forcefields, machine-learned force fields are designed to
process data efficiently in batches, making it advantageous to run multiple simulations in
parallel in order to maximize resource utilization and minimize the cost per trajectory.
The ``copies`` option in the configuration file should be carefully selected based on the
size of the system to ensure efficient memory usage, and may require some testing to
achieve optimal simulation performance.

For generating a simulation input for the pretrained transferable model provided with the
manuscript ``Navigating protein landscapes with a machine-learned coarse-grained model``,
use the provided ``transferable_priors.yaml`` in ``configuration_files`` to build an input
configuration using consistent priors with the ones used in the manuscript. Adapt the
``trpcage_sim.yaml`` file to the protein sequence to be simulated.