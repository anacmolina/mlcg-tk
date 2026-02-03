1) Loading and processing all-atom simulation data
===================================================

Command:

::

   mlcg-tk-gen_input_data process_raw_dataset --config configuration_files/trpcage.yaml

   mlcg-tk-gen_input_data build_neighborlists --config configuration_files/trpcage.yaml --config configuration_files/trpcage_priors.yaml

This procedure will loop over all of the sample names specified by the ``names`` option.
For each instance, it will load the atomistic coordinates, forces, and structures and map
these to a lower resolution specified in the input file (this allows for various
resolutions and CG embeddings to be used). Then, using the ``PriorBuilders`` listed in
``prior_builders``, the script will generate a neighbor list for each molecule, so long as
the prior builders are implemented in ``prior_gen.py`` and their specific neighbor list
builders are implemented in ``mlcg_tk.input_generator.prior_nls``.


.. note:: 
    Keep in mind that the priors are assumed to be in ``[kcal/mol]`` at the fitting stage so
    raw forces should be transformed to ``[kcal/mol/angstrom]``.

Note if you are using a custom dataset:
---------------------------------------

If your program gets killed after the loading of the all-atom data succeeded (tqdm bar
finished) but before ``process_raw_dataset`` saved the CG output, try to set
``batch_size`` in your ``trpcage.yaml`` file. This will batch the matrix multiplication
between atomistic coordinates/forces, which is the most memory-consuming part of the
coarse-graining at this stage.

Batch processing for large molecules
------------------------------------

If the dataset loads into memory successfully (the tqdm bar completes), but the program
fails before saving the CG output, consider setting ``atoms_batch_size`` in your
``trpcage.yaml`` file.

This optional parameter specifies the batch size for processing atoms in large molecules.
When set, constraints among atoms for coordinate and force mappings will be computed in
batches of this size to reduce memory usage. To improve computational efficiency, it is
assumed that the molecular structures have ordered residues.

If ``atoms_batch_size`` is larger than the total number of atoms in the molecule, all
atoms will be processed at once (the default behavior).

Batch processing for large datasets
-----------------------------------

Should your dataset be too big to be loaded into memory at once (the tqdm bar doesn't
finish before it fails), you can set the ``mol_num_batches`` in your ``trpcage.yaml`` file as
well as your ``trpcage_stats.yaml``, ``trpcage_delta_forces.yaml`` and
``trpcage_packaging.yaml`` file. This will seperate the trajectories in your dataset into
``mol_num_batches`` chunks that will be treated as separate molecules for the
coarse-graining and statistics computing stages (see 2 below) and the statistics of the
different batches will be automatically accumulated to get only one prior object in the
end. Note that in this case, the force map will be only computed on the first batch and
re-used for all subsequent batches to ensure consistency in the case of optimized force
maps
