cd $PBS_O_WORKDIR
source env_vars.sh
source activate hvd
tensorboard --logdir=checkpoints
