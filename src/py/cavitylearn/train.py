

import os
import sys
import logging
import configparser

import re
import time
import math

import tensorflow as tf


from . import data
from . import catalonet0

# =============================== set up logging ==============================

LOGDEFAULT = logging.INFO
logger = logging.getLogger(__name__)

# =============================== set up config ===============================

THISCONF = 'cavitylearn-train'
config = configparser.ConfigParser(interpolation=None)

# default config values
config[THISCONF] = {
    "checkpoint_frequency": 200,
    "testing_frequency": 30
}

# Look for the config file
for p in sys.path:
    cfg_filepath = os.path.join(p, 'config.ini')
    if os.path.exists(cfg_filepath):
        logger.debug('Found config file in: ' + cfg_filepath)
        config.read(cfg_filepath)
        break
else:
    logger.debug("config.ini not found!")


def purge_dir(directory, pattern):
    """Purge all files in a directory that match a regular expression pattern.
    This does not recurse into subdirectories.

    :param directory: Directory to be purged
    :param pattern: The regular expression pattern as string
    """
    for f in os.listdir(directory):
        if re.search(pattern, f):
            os.remove(os.path.join(directory, f))


def run_training(dataset_dir, run_dir, run_name, continue_previous=False,
                 learnrate=1e-4, learnrate_decay=0.95, keep_prob_conv=0.75, keep_prob_hidden=0.50,
                 batchsize=50, max_batches=0, epochs=1, track_test_accuracy=False, progress_tracker=None):
    dataconfig = data.read_dataconfig(os.path.join(dataset_dir, "datainfo.ini"))
    testing_frequency = int(config[THISCONF]['testing_frequency'])

    # Get all datasets in the input directory
    datasets = data.DataSets(os.path.join(dataset_dir, "labels.txt"), os.path.join(dataset_dir, "boxes"), dataconfig)

    # get training dataset. If there isn't a dataset called "train", take all examples in the dataset
    if "train" in datasets:
        trainset = datasets["train"]
    else:
        trainset = datasets[""]

    # If we have a training set, validate against it. If not, ignore it.
    if track_test_accuracy:
        if "test" not in datasets:
            raise ValueError("Test set accuracy tracking requested, but test set not found")

        testset = datasets["test"]
    else:
        testset = None

    # calculate training batches to run
    batches_in_trainset = math.ceil(trainset.N / batchsize)
    if max_batches:
        total_train_batches = min(batches_in_trainset, max_batches) * epochs
    else:
        total_train_batches = batches_in_trainset * epochs

    # calculate our workload
    if testset:
        batches_in_testset = int(math.ceil(testset.N / batchsize))
        if max_batches:
            number_of_testset_evaluations = int(math.ceil(min(batches_in_trainset, max_batches) / testing_frequency) * epochs)
        else:
            number_of_testset_evaluations = int(math.ceil(batches_in_trainset / testing_frequency) * epochs)

        total_batches = total_train_batches + int(
            batches_in_testset * number_of_testset_evaluations)

        logger.debug("train batches %d ; batches_in_testset %d ; number_of_testset_evaluations %d ; "
                     "total_batches %d ; ",
                     total_train_batches, batches_in_testset, number_of_testset_evaluations, total_batches)
    else:
        total_batches = total_train_batches
        batches_in_testset = 0

    # initialize progress tracker
    if progress_tracker:
        progress_tracker.init(total_batches)

    # define input tensors
    with tf.variable_scope("input"):
        label_placeholder = tf.placeholder(tf.int32, shape=[None, dataconfig.num_classes], name="labels")
        input_placeholder = tf.placeholder(tf.float32, shape=[None, dataconfig.boxshape[0], dataconfig.boxshape[1],
                                                              dataconfig.boxshape[2], dataconfig.num_props]
                                           , name="boxes")

    # global step variable
    global_step = tf.Variable(0, name='global_step', trainable=False)

    p_keep_hidden_placeholder = tf.placeholder(tf.float32)
    p_keep_conv_placeholder = tf.placeholder(tf.float32)

    # prediction, loss and training operations
    logits = catalonet0.inference(input_placeholder, dataconfig, p_keep_conv_placeholder, p_keep_hidden_placeholder)
    loss = catalonet0.loss(logits, label_placeholder)
    train_op = catalonet0.train(loss, learnrate, learnrate_decay, global_step)

    # log the training accuracy
    num_correct = catalonet0.evaluation(logits, label_placeholder)
    accuracy = tf.truediv(num_correct, tf.shape(input_placeholder)[0], name="accuracy")
    tf.scalar_summary("accuracy", accuracy)
    train_summary_op = tf.merge_all_summaries()

    # log the test accuracy if required
    with tf.variable_scope("input"):
        test_accuracy_placeholder = tf.placeholder(tf.float32, name="test_accuracy")

    test_summary = tf.scalar_summary("accuracy", test_accuracy_placeholder)

    # create output directories if they don't exist

    # Create or purge checkpoint directory if not continuing
    if os.path.isdir(os.path.join(run_dir, "checkpoints")):
        if not continue_previous:
            purge_dir(os.path.join(run_dir, "checkpoints"), r'^{run_name}(-\d+)?(\.meta)?$'.format(run_name=run_name))
    else:
        os.makedirs(os.path.join(run_dir, "checkpoints"))

    # create purge log directory
    if os.path.isdir(os.path.join(run_dir, "logs", run_name)):
        if not continue_previous:
            purge_dir(os.path.join(run_dir, "logs", run_name), r'^events\.out\.tfevents\.\d+')
    else:
        os.makedirs(os.path.join(run_dir, "logs", run_name))

    if testset:
        # create or purge test log directory
        if os.path.isdir(os.path.join(run_dir, "logs", run_name, "test")):
            if not continue_previous:
                purge_dir(os.path.join(run_dir, "logs", run_name, "test"), r'^events\.out\.tfevents\.\d+')
        else:
            os.makedirs(os.path.join(run_dir, "logs", run_name, "test"))

    saver = tf.train.Saver()
    checkpoint_path = os.path.join(run_dir, "checkpoints", run_name)
    with tf.Session() as sess:

        if os.path.exists(checkpoint_path) and continue_previous:
            logger.info("Found training checkpoint file `{}` , continuing training. ".format(checkpoint_path))
            saver.restore(sess, checkpoint_path)
        else:
            sess.run(tf.initialize_all_variables())

        # Create summary writer
        train_writer = tf.train.SummaryWriter(os.path.join(run_dir, "logs", run_name), sess.graph)

        if testset:
            test_writer = tf.train.SummaryWriter(os.path.join(run_dir, "logs", run_name, "test"), sess.graph)

        logger.info(
            "Beginning training. You can watch the training progress by running `tensorboard --logdir {}`".format(
                os.path.join(run_dir, "logs")))

        timings = dict()
        batchcount = 0

        for epoch in range(epochs):
            start_time = time.time()

            trainset.shuffle(norestart=True)
            trainset.rewind_batches(norestart=False)

            batch_idx = 0
            while batch_idx < max_batches or max_batches == 0:

                tick = time.time()

                # get training data
                labels, boxes = trainset.next_batch(batchsize)

                timings["trainset_read"] = time.time() - tick

                # abort training if we didn't get any more data
                if len(labels) == 0:
                    break

                # feed it
                feed_dict = {
                    input_placeholder: boxes,
                    label_placeholder: labels,
                    p_keep_conv_placeholder: keep_prob_conv,
                    p_keep_hidden_placeholder: keep_prob_hidden
                }

                tick = time.time()

                # Do it!
                _, summary_str, global_step_val = sess.run([train_op, train_summary_op, global_step],
                                                           feed_dict=feed_dict)

                timings["trainset_calc"] = time.time() - tick

                tick = time.time()

                train_writer.add_summary(summary_str, global_step_val)
                train_writer.flush()

                timings["trainset_log"] = time.time() - tick

                # when we have a test set, evaluate the model accuracy on the test set
                if testset and batch_idx % testing_frequency == 0:

                    test_timings = {
                        "read_batch": list(),
                        "calc_batch": list(),
                        "eval_batch": list()
                    }
                    test_accuracies = []

                    for test_batch_idx in range(batches_in_testset):

                        tick = time.time()
                        labels, boxes = testset.next_batch(batchsize)

                        test_timings['read_batch'].append(time.time() - tick)

                        test_feed_dict = {
                            input_placeholder: boxes,
                            label_placeholder: labels,
                            p_keep_conv_placeholder: 1,
                            p_keep_hidden_placeholder: 1
                        }

                        tick = time.time()

                        test_accuracy_val = sess.run(accuracy, feed_dict=test_feed_dict)
                        test_accuracies.append(test_accuracy_val)

                        test_timings['calc_batch'].append(time.time() - tick)

                        if progress_tracker:
                            progress_tracker.update("Test batch ")

                        logger.debug("")
                        logger.debug("test: read_batch: %f ; calc_batch %f",
                                     test_timings['read_batch'][-1], test_timings['calc_batch'][-1])

                        batchcount += 1
                        pass

                    # rewind test batches after using them
                    testset.rewind_batches()

                    tick = time.time()

                    summary_str = sess.run(test_summary,
                                           feed_dict={
                                               test_accuracy_placeholder: sum(test_accuracies) / len(test_accuracies)})

                    test_timings['eval_batch'].append(time.time() - tick)

                    test_writer.add_summary(summary_str, global_step_val)
                    test_writer.flush()

                    timings['testset_read_avg'] = sum(test_timings['read_batch']) / len(test_timings['read_batch'])
                    timings['testset_calc_avg'] = sum(test_timings['calc_batch']) / len(test_timings['calc_batch'])
                    timings['testset_eval_avg'] = sum(test_timings['eval_batch']) / len(test_timings['eval_batch'])

                    logger.debug("")
                    logger.debug("testset_read_avg: %(testset_read_avg)f; testset_calc_avg: %(testset_calc_avg)f; "
                                 "testset_eval_avg: %(testset_eval_avg)f",
                                 timings)

                # Save it
                if batch_idx % int(config[THISCONF]['checkpoint_frequency']) == 0:
                    saver.save(sess, checkpoint_path, global_step=global_step)

                logger.debug("")
                logger.debug(
                    "trainset_read: %(trainset_read)f; trainset_calc: %(trainset_calc)f; "
                    "trainset_log %(trainset_log)f",
                    timings)

                batch_idx += 1
                batchcount += 1

                if progress_tracker:
                    progress_tracker.update(
                        "Train Batch {:>5d} ".format(batches_in_trainset * epoch + batch_idx)
                    )

            # Save it again, this time without appending the step number to the filename
            saver.save(sess, checkpoint_path)
            end_time = time.time()

            if batch_idx != 0:
                logger.info("")
                logger.info("Finished run {:d}. Total time: {:d} s. Time per batch: {:f} s"
                            .format(epoch+1, int(end_time - start_time), (end_time - start_time) / batch_idx))

        logger.debug("batchount: %d", batchcount)


