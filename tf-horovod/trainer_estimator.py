import argparse
import sys
import os
import errno
import shutil
import tensorflow as tf
layers = tf.layers
import numpy as np
import horovod.tensorflow as hvd
import socket

FLAGS = None

default_batchsize = 16
input_height = 480
input_width = 640
n_classes = 10
display_step = 1
filter_size = 3
depth_in = 3
depth_out1 = 16
depth_out2 = 32
depth_out3 = 64
dense_ct = 256

tf.logging.set_verbosity(tf.logging.INFO)
print(socket.gethostname())

#input function to read from TFRecord database (created using tf inception's build_image_data.py)
def dataset_input_fn(dir=os.getcwd(), prefix='train-', batch_size=default_batchsize):
    filenames = [dir+'/'+f for f in os.listdir(dir) if f.startswith(prefix)]
    print(f for f in filenames)
    if len(filenames) < 1:
        raise Exception("No files found with prefix "+prefix)
    dataset = tf.data.TFRecordDataset(filenames)

    # Use `tf.parse_single_example()` to extract data from a `tf.Example`
    # protocol buffer, and perform any additional per-record preprocessing.
    def parser(record):
        keys_to_features = {
            "image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
            "image/class/label": tf.FixedLenFeature((), tf.int64,
                                                    default_value=tf.zeros([], dtype=tf.int64)),
        }
        parsed = tf.parse_single_example(record, keys_to_features)

        # Perform additional preprocessing on the parsed data.
        image = tf.image.decode_jpeg(parsed["image/encoded"])
        image = tf.cast(tf.reshape(image, [480, 640, 3]), tf.float32)
        label = parsed["image/class/label"]
        #label = tf.one_hot(label,n_classes)
        return image, label

    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size=200)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()

    # `features` is a dictionary in which each value is a batch of values for
    # that feature; `labels` is a batch of labels.
    features, labels = iterator.get_next()
    return features, labels

def train_dataset_input_fn():
    return dataset_input_fn(prefix='train-', batch_size=FLAGS.batch_size)
def eval_dataset_input_fn():
    return dataset_input_fn(prefix='validation-',batch_size=1)

def conv_net(features, labels, mode):
    #convolution
    with tf.name_scope('pool_1'):
        features = tf.nn.max_pool(
            features, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    with tf.name_scope('conv_1'):
        h_conv1 = layers.conv2d(features, depth_out1, kernel_size=[filter_size+2, filter_size+2],
                                activation=tf.nn.relu, padding="SAME")
        h_pool1 = tf.nn.max_pool(
            h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope('conv_2'):
        h_conv2 = layers.conv2d(h_pool1, depth_out2, kernel_size=[filter_size, filter_size],
                                activation=tf.nn.relu, padding="SAME")
        h_pool2 = tf.nn.max_pool(
            h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope('conv_3'):
        h_conv3 = layers.conv2d(h_pool2, depth_out3, kernel_size=[filter_size, filter_size],
                                activation=tf.nn.relu, padding="SAME")
        h_pool3 = tf.nn.max_pool(
            h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # reshape tensor into a batch of vectors
        h_pool3_flat = tf.layers.flatten(h_pool3)

    # Densely connected layer with 1024 neurons.
    with tf.name_scope('fully_connected_1'):
        h_fc1 = layers.dropout(
            layers.dense(h_pool3_flat, dense_ct, activation=tf.nn.relu),
            rate=FLAGS.dropout, training=mode == tf.estimator.ModeKeys.TRAIN)

    with tf.name_scope('fully_connected_2'):
        h_fc2 = layers.dropout(
            layers.dense(h_fc1, dense_ct, activation=tf.nn.relu),
            rate=FLAGS.dropout, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Compute logits (1 per class) and compute loss.
    logits = layers.dense(h_fc2, n_classes, activation=None)

    #updated estimator functions
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    #logging
    #hostname for logging
    host = tf.constant(socket.gethostname(), name='host')
    hvd_rank = tf.constant(hvd.rank(), name='hvd_rank')
    hvd_size = tf.constant(hvd.size(), name='hvd_size')

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=tf.argmax(input=logits, axis=1), name='accuracy')
    tf.summary.scalar('accuracy', accuracy[1])
    log_hook = tf.train.LoggingTensorHook({'accuracy':accuracy[1]},
                               every_n_iter=FLAGS.log_n_iters)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # Horovod: scale learning rate by the number of workers.
        optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate * hvd.size())
        optimizer = hvd.DistributedOptimizer(optimizer)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          train_op=train_op,
                                          training_hooks=[log_hook])
    # mode == tf.estimator.ModeKeys.EVAL
    #accuracy calculated for EVAL
    eval_metric_ops = {'accuracy':accuracy}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    if FLAGS.omp:
        os.environ['OMP_NUM_THREADS']=str(FLAGS.omp)
    # Horovod: initialize Horovod.
    hvd.init()

    # GPU training, disabled
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    if False:
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(hvd.local_rank())

    # Horovod: save checkpoints only on worker 0 to prevent other workers from
    # corrupting them.
    model_dir = './' if hvd.rank() == 0 else None

    config_dict=dict()
    if FLAGS.inter_op:
        config_dict.update(inter_op_parallelism_threads=FLAGS.inter_op)
    if FLAGS.intra_op:
        config_dict.update(intra_op_parallelism_threads=FLAGS.intra_op)
    config = tf.ConfigProto(**config_dict)

    # Create the Estimator
    tfrecord_estimator = tf.estimator.Estimator(
        model_fn=conv_net, model_dir=model_dir,
        config=tf.estimator.RunConfig(session_config=config))

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    #add hvd hooks
    hooks = [
        # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states
        # from rank 0 to all other processes. This is necessary to ensure consistent
        # initialization of all workers when training is started with random weights
        # or restored from a checkpoint.
        hvd.BroadcastGlobalVariablesHook(0),
        tf.train.LoggingTensorHook(['host','hvd_rank','hvd_size'],
                                   every_n_iter=FLAGS.log_n_iters),
        ]
    if hvd.rank() == 0:
        hooks.append(tf.train.CheckpointSaverHook(checkpoint_dir='checkpoints/trainer_estimator/',save_steps=50))
    # Horovod: adjust number of steps based on number of GPUs.
    estimator_config={
        'input_fn':train_dataset_input_fn,
        'hooks':hooks,
    }
    if FLAGS.stop_at_step:
        estimator_config['steps'] = FLAGS.stop_at_step

    if FLAGS.validation == False:
        tfrecord_estimator.train(**estimator_config)
    else:
        ## If evaluating, run only on a single node.
        if hvd.rank() == 0:
            #Evaluate the model and print results
            eval_results = tfrecord_estimator.evaluate(input_fn=eval_dataset_input_fn,steps=1000)
            print(eval_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--inter_op",
        type=int,
        default=None,
        help="inter_op_parallism_threads"
    )
    parser.add_argument(
        "--intra_op",
        type=int,
        default=None,
        help="intra_op_parallism_threads"
    )
    parser.add_argument(
        "--omp",
        type=int,
        default=None,
        help="OMP_NUM_THREADS"
    )
    parser.add_argument(
        "--stop_at_step",
        type=int,
        default=200000,
        help="Stop training at step"
    )
    parser.add_argument(
        "--log_n_iters",
        type=int,
        default=50,
        help="Log information after n iterations"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./checkpoints",
        help="Directory for logging"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="learning rate, will scale with hvd.size()"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="dropout rate for fully connected layers"
    )
    parser.add_argument(
        "--validation",
        action='store_true',
        default=False,
        help="Use to run validation"
    )
    ##TODO: add epochs if required

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
