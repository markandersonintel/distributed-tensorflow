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
tf.logging.set_verbosity(tf.logging.INFO)
print(socket.gethostname())

#input function to read from TFRecord database (created using tf inception's build_image_data.py)
def dataset_input_fn(dir=os.getcwd(), prefix='train-'):
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
        label = parsed["image/class/label"] #tf.cast(parsed["image/class/label"], tf.float32)
        label = tf.one_hot(label,11)
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
    #features, labels = iterator.get_next()
    return iterator.get_next() #features, labels

def conv_net(feature, target, mode):
    #convolution
    if True:
        with tf.name_scope('pool_1'):
            feature = tf.nn.max_pool(
                feature, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    with tf.name_scope('conv_1'):
        h_conv1 = layers.conv2d(feature, depth_out1, kernel_size=[filter_size+2, filter_size+2],
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
            rate=dropout, training=mode == tf.estimator.ModeKeys.TRAIN)

    with tf.name_scope('fully_connected_2'):
        h_fc2 = layers.dropout(
            layers.dense(h_fc1, dense_ct, activation=tf.nn.relu),
            rate=dropout, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Compute logits (1 per class) and compute loss.
    logits = layers.dense(h_fc2, n_classes, activation=None)
    loss = tf.losses.softmax_cross_entropy(target, logits)

    return tf.argmax(logits, 1), loss

learning_rate = 0.0001
epochs = 200
batch_size = 16
num_batches = int(24000/batch_size) #replace with n/batch_size
input_height = 480
input_width = 640
n_classes = 11
dropout = 0.3
display_step = 1
filter_size = 3
depth_in = 3
depth_out1 = 16
depth_out2 = 32
depth_out3 = 64
dense_ct = 256

def main(_):
    # if os.path.isdir(FLAGS.log_dir):
    #     try:
    #         shutil.rmtree(FLAGS.log_dir)
    #     except OSError as ex:
    #         if ex.errno == errno.EEXIST and os.path.isdir(FLAGS.logdir):
    #             pass
    #         else:
    #             raise
    # os.mkdir(FLAGS.log_dir)
    os.environ["OMP_NUM_THREADS"] = str(FLAGS.omp)

    # Horovod: initialize Horovod.
    hvd.init()
    host = tf.constant(socket.gethostname())
    # Build model...
    # input placeholders
    with tf.name_scope('inputs'):
        # x = tf.placeholder(tf.float32, [None, input_height, input_width, depth_in], name='image')
        # y = tf.placeholder(tf.float32, [None, n_classes], name='label')
        # keep_prob = tf.placeholder(tf.float32)
        x, y = dataset_input_fn()
    # create cnn
    predict, loss = conv_net(x, y, tf.estimator.ModeKeys.TRAIN)
    tf.summary.scalar('loss',loss)

    #define optimizer, horovod wrapper
    # opt = tf.train.AdamOptimizer(learning_rate=learning_rate * hvd.size())
    # opt = hvd.DistributedOptimizer(opt)

    # Horovod: adjust learning rate based on number of GPUs.
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate * hvd.size())
    #opt = tf.train.RMSPropOptimizer(learning_rate * hvd.size())

    # Horovod: add Horovod Distributed Optimizer.
    opt = hvd.DistributedOptimizer(opt)

    global_step = tf.train.get_or_create_global_step()
    train_op = opt.minimize(loss, global_step=global_step)

    # evaluate model
    correct_pred = tf.equal(predict, tf.argmax(y, 1)) #tf.argmax(loss, 1)
    with tf.variable_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    #initialize varibales, merge summaries
    iter = dataset_input_fn()
    #init = tf.global_variables_initializer()
    #merged = tf.summary.merge_all()

    #add hvd hooks
    hooks = [
        # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states
        # from rank 0 to all other processes. This is necessary to ensure consistent
        # initialization of all workers when training is started with random weights
        # or restored from a checkpoint.
        hvd.BroadcastGlobalVariablesHook(0),

        # Horovod: adjust number of steps based on number of GPUs.
        tf.train.StopAtStepHook(last_step=200000 // hvd.size()),

        tf.train.LoggingTensorHook(tensors={'step': global_step, 'loss': loss, 'acc:': accuracy, 'host': host},
                                   every_n_iter=10)
    ]

    config = tf.ConfigProto(inter_op_parallelism_threads=FLAGS.interop, intra_op_parallelism_threads=FLAGS.intraop)
    # config.gpu_options.allow_growth = True
    # config.gpu_options.visible_device_list = str(hvd.local_rank())

    checkpoint_dir = FLAGS.log_dir if hvd.rank() == 0 else None

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                           save_checkpoint_steps=50,
                                           hooks=hooks,
                                           config=config) as mon_sess:
        print('starting monitored training session')

        #feed_dict= {x: np.zeros([1,input_height,input_width,depth_in], dtype=np.float32), y: np.zeros([1,n_classes], dtype=np.float32), keep_prob: 1.0}

        while not mon_sess.should_stop():
            #mon_sess.run(iter)
            mon_sess.run(train_op)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--interop",
        type=int,
        default=2,
        help="inter_op_parallism_threads"
    )
    parser.add_argument(
        "--intraop",
        type=int,
        default=32,
        help="intra_op_parallism_threads"
    )
    parser.add_argument(
        "--omp",
        type=int,
        default=32,
        help="OMP_NUM_THREADS"
    )
    parser.add_argument(
        "--stop_at_step",
        type=int,
        default=200000,
        help="Stop training at step"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./checkpoints",
        help="Directory for logging"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


