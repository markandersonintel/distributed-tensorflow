# tf-undistributed

# Usage

## Setting Up TFRecords Dataset
Use train_val_split.py to split folder of categorical subfolders into train and validation folders.  this will be used to get data into format for tf.inception's build_image_data.py script for TFRecord creation.

Use build_image_data.py to create TFRecord database with the following syntax:
```
python build_image_data.py --train_directory=<PATH_TO_THE_UNZIPPED_TRAIN_DIRECTORY> --validation_directory=<PATH_TO_THE_VALIDATION_DIRECTORY> --output_directory=<PATH_TO_SAVE_TF_RECORDS> --labels_file=<PATH_TO_THE_CREATED_LABELS_FILE>
```
E.g.
```
python build_image_data.py --train_directory=train/ --validation_directory=val/ --output_directory= --labels_file=labels_file.txt
```
trainer.py uses the training shards created as the input data.

The dataset used with this script can be found here:
* [Distracted Driver TFRecord Dataset](https://drive.google.com/open?id=1FYrVAszEFMNTUdObK8SrKOqM8bwVxSPl)


## Run Training
Run the following in a TensorFlow environment to start training:
```
python trainer.py
```
The following parameters may be set:
--omp			OMP_NUM_THREADS, default 32
--interop		inter_op_parallelism_threads, default 2
--intraop		intra_op_parallelism_threads, default 32
--stop_at_step	stop training at global step, default 20000


## Remote Tensorboard Setup
Checkpointing is done in the training script, and can be viewed through tensorboard.  To use tensorboard, qsub tensorboard.sh which will open a tensorboard instance using logdir=checkpoints.

Use qstat to find the tensorboard job, and run
```
qpeek -e job#
```
to get the error output.  This will have the hostname of the node which can be used to create a local tunnel.

Open a local terminal and run
```
ssh -L localhost:16006:c002-n001:6006 colfaxc002
```
where c002-n001 can be replaced with the hostname of the node allocated for the job. open a browser and enter localhost:16006.

## Running Inference and Validation
Inference and validation to be implemented.


