import os
import re
import tensorflow as tf
import time
import logging

from . import data
from . import catalonet0

# =============================== set up logging ==============================

LOGDEFAULT = logging.INFO
logger = logging.getLogger(__name__)

CHECKPOINT_FREQUENCY = 200


def purge_dir(directory, pattern):
    for f in os.listdir(directory):
        if re.search(pattern, f):
            os.remove(os.path.join(directory, f))


def run_training(dataset_dir, run_dir, run_name, continue_previous=False, batchsize=50, max_batches=0, repeat=1, progress_fun=None):
    dataconfig = data.read_dataconfig(os.path.join(dataset_dir, "datainfo.ini"))
    with open(os.path.join(dataset_dir, "labels.txt"), 'rt') as labelfile:
        trainset = data.DataSet(labelfile, os.path.join(dataset_dir, "boxes"), dataconfig)

    batches_in_trainset = int(trainset.N / batchsize + .5)
    if max_batches:
        total_batches = min(batches_in_trainset, max_batches) * repeat
    else:
        total_batches = batches_in_trainset * repeat

    # define input tensors
    input_placeholder = tf.placeholder(tf.float32, shape=[None, dataconfig.boxshape[0], dataconfig.boxshape[1],
                                                          dataconfig.boxshape[2], dataconfig.num_props])
    labels_placholder = tf.placeholder(tf.float32, shape=[None, dataconfig.num_classes])

    # global step variable
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # prediction, loss and training operations
    logits = catalonet0.inference(input_placeholder, dataconfig)
    loss = catalonet0.loss(logits, labels_placholder)
    train_op = catalonet0.train(loss, 1e-4, global_step)

    # log the training accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_placholder, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
    tf.scalar_summary("accuracy", accuracy)

    saver = tf.train.Saver()
    summary_op = tf.merge_all_summaries()

    # create output directories if they don't exist
    if not os.path.isdir(os.path.join(run_dir, "checkpoints")):
        os.makedirs(os.path.join(run_dir, "checkpoints"))

    if not os.path.isdir(os.path.join(run_dir, "logs", run_name)):
        os.makedirs(os.path.join(run_dir, "logs", run_name))

    checkpoint_path = os.path.join(run_dir, "checkpoints", run_name)
    with tf.Session() as sess:

        if os.path.exists(checkpoint_path) and continue_previous:
            saver.restore(sess, checkpoint_path)
        else:
            # purge log directory for run, before running it again
            purge_dir(os.path.join(run_dir, "logs", run_name), r'^events\.out\.tfevents\.\d+')

            sess.run(tf.initialize_all_variables())

        # Create summary writer
        summary_writer = tf.train.SummaryWriter(os.path.join(run_dir, "logs", run_name), sess.graph)

        for rep in range(repeat):
            trainset.rewind_batches()
            trainset.shuffle()

            start_time = time.time()

            batch_idx = 0
            while batch_idx < max_batches or max_batches == 0:
                # get training data
                labels, boxes = trainset.next_batch(batchsize)

                # abort training if we didn't get any more data
                if len(labels) == 0:
                    break

                # feed it
                feed_dict = {
                    input_placeholder: boxes,
                    labels_placholder: labels
                }

                # Do it!
                sess.run(train_op, feed_dict=feed_dict)

                # Log it
                summary_str, global_step_val, accuracy_val = \
                    sess.run([summary_op, global_step, accuracy], feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, global_step_val)
                summary_writer.flush()

                # print("i = {:>3}; accuracy: {:.0f}%".format(step, accuracy_val*100))
                # print(".", end="")

                # Save it
                if batch_idx % CHECKPOINT_FREQUENCY == 0:
                    saver.save(sess, checkpoint_path, global_step=global_step)

                batch_idx += 1

                if progress_fun:
                    progress_fun(batchsize * rep + batch_idx, total_batches)

            # Save it again, this time without appending the step number to the filename
            saver.save(sess, checkpoint_path)
            end_time = time.time()

            logger.info("Finished run {:d}. Total time: {:d} s. Time per batch: {:f} s"
                        .format(rep, int(end_time - start_time), (end_time - start_time) / batch_idx))
