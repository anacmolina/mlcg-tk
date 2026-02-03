4) Package Training Data
========================

Command::

   mlcg-tk-package_training_data package_training_data --config configuration_files/trpcage_packaging.yaml

Once all training data has been produced, these data must be packaged in a form that can
be passed to the MLCG library for model training. In this step, CG coordinates, delta
forces, and embeddings are loaded for all provided sample names in a raw dataset and saved
as an HDF5 file. In the same step, molecules are split into training and validation sets
and a saved in a partition file, which also stores information about the batch sizes to
use for training and any striding that should be applied.

In the case that multiple dataset are used to train a model, these can be merged into
combined HDF5 and partition files using the following:

Command::

   mlcg-tk-package_training_data combine_datasets --dataset_names '[dataset_1, dataset_2, etc]' --save_dir /path/to/saved/files/ --force_tag tag

The optional force tag specifies a label given to produced delta forces and saved packaged
data.

4.1) Add Decoys To Training Data
==================================

It is possible to add so-called decoys to the training data, distorted (unphysical)
configurations with a zero delta-force label, such that the network has examples of
configurations on which it should rely on the prior.

The script ``mlcg-tk-add_decoys`` enables the addition of decoys to a previously
constructed HDF5 dataset. It can be used the following way to append decoys to the
existing HDF5 dataset (or the config file can be modified for the script to copy the
original HDF5 dataset before adding the decoys):

Command::

   mlcg-tk-add_decoys add_decoy --config configuration_files/trpcage_decoys_dataset.yaml

The same script can also be used to add these decoys to an existing partition file in
order to incorporate them during training:

Command::

   mlcg-tk-add_decoys update_partition_file --config configuration_files/trpcage_decoys_partition.yaml

Note that an arbitrary number of decoys with different noise levels and strides can be
appended to a dataset; only the decoys present in the partition file will effectively be
taken into account for training.
