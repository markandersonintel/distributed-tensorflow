# Distributed Tensorflow implementations

## tf-dist
Distibuted Tensorflow based on implementations outlined in the official TensorFlow documentation.  Creates parameter and worker servers which communicate through a monitored training session over an HPC cluster.

## tf-horovod
Distributed training using Uber's Horovod framework with MPI.

## tf-undistributed
Undistributed training with similar model topology as tf-horovod, for performance comparison.

## references
Distributed Tensorflow, Google (January, 2019) https://www.tensorflow.org/deploy/distributed

Sergeev, A., Del Balso, M. (2018) Horovod: fast and easy distributed deep learning in TensorFlow. arXiv:1802.05799 https://github.com/uber/horovod


