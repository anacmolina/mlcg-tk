3) Producing delta forces
=========================

Command::

   mlcg-tk-produce_delta_forces produce_delta_forces --config configuration_files/trpcage_delta_forces.yaml

This procedure will load the prior model specified by ``prior_fn`` and then once again
loop over all sample names provided. It will then calculate and remove the baseline forces
using the coordinates, forces, embeddings, and neighbourlists created in the previous
step. It will then save the delta forces which can then be used for training.

The following example script shows how delta forces can be computed on a computing cluster
using GPU acceleration:

.. code-block:: bash

   #!/bin/bash 
   #SBATCH --ntasks-per-node=1 
   #SBATCH --nodes=1 
   #SBATCH --mem=24G 
   #SBATCH --time=4-00:00:00 
   #SBATCH --gres=gpu:1 
   #SBATCH --partition=gpu 
   #SBATCH --output=trpcage_delta_forces_gpu.log 
   #SBATCH --job-name=test_job

   mlcg-tk-produce_delta_forces produce_delta_forces --config configuration_files/trpcage_delta_forces.yaml

Here, make sure to specify ``cuda`` for the ``device`` option in the configuration file.
Note that depending on the GPU being used and its available memory, it may be necessary to
adjust the ``batch_size``.
