#!/usr/bin/env python3

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
TESTSET_EVAL_FREQUENCY = 30


def purge_dir(directory, pattern):
    for f in os.listdir(directory):
        if re.search(pattern, f):
            os.remove(os.path.join(directory, f))


def run_training(dataset_dir, run_dir, run_name, continue_previous=False, batchsize=50, max_batches=0, repeat=1,
                 track_test_accuracy=False, progress_tracker=None):
    dataconfig = data.read_dataconfig(os.path.join(dataset_dir, "datainfo.ini"))

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
    batches_in_trainset = int(trainset.N / batchsize + .5)
    if max_batches:
        total_batches = min(batches_in_trainset, max_batches) * repeat
    else:
        total_batches = batches_in_trainset * repeat

    # initialize progress tracker
    if progress_tracker:
        if not testset:
            progress_tracker.init(total_batches)
        else:
            batches_in_testset = int((testset.N / batchsize + .5))
            number_of_testset_evaluations = (total_batches / TESTSET_EVAL_FREQUENCY)
            total_batches_with_testset = total_batches + int(batches_in_testset * number_of_testset_evaluations)

            logger.debug("train batches %d ; batches_in_testset %d ; number_of_testset_evaluations %d ; "
                         "total_batches_with_testset %d ; ",
                         total_batches, batches_in_testset, number_of_testset_evaluations, total_batches_with_testset)

            progress_tracker.init(total_batches_with_testset)

    # define input tensors
    input_placeholder = tf.placeholder(tf.float32, shape=[None, dataconfig.boxshape[0], dataconfig.boxshape[1],
                                                          dataconfig.boxshape[2], dataconfig.num_props]
                                       , name="input_boxes")
    labels_placholder = tf.placeholder(tf.int32, shape=[None, dataconfig.num_classes], name="input_labels")

    # global step variable
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # prediction, loss and training operations
    logits = catalonet0.inference(input_placeholder, dataconfig)
    loss = catalonet0.loss(logits, labels_placholder)
    train_op = catalonet0.train(loss, 1e-5, global_step)

    # log the training accuracy
    accuracy = catalonet0.evaluation(logits, labels_placholder) / tf.shape(input_placeholder)[0]
    tf.scalar_summary("accuracy", accuracy)
    train_summary_op = tf.merge_all_summaries()

    # log the test accuracy if required
    if testset:
        test_accuracy_placeholder = tf.placeholder(tf.float32, name="input_test_accuracy")
        test_summary = tf.scalar_summary("accuracy", test_accuracy_placeholder)

    # create output directories if they don't exist
    if not os.path.isdir(os.path.join(run_dir, "checkpoints")):
        os.makedirs(os.path.join(run_dir, "checkpoints"))

    if not os.path.isdir(os.path.join(run_dir, "logs", run_name)):
        os.makedirs(os.path.join(run_dir, "logs", run_name))

    if testset and not os.path.isdir(os.path.join(run_dir, "logs", run_name, "test")):
        os.makedirs(os.path.join(run_dir, "logs", run_name, "test"))

    saver = tf.train.Saver()
    checkpoint_path = os.path.join(run_dir, "checkpoints", run_name)
    with tf.Session() as sess:

        if os.path.exists(checkpoint_path) and continue_previous:
            logger.info("Found training checkpoint file `{}` , continuing training. ".format(checkpoint_path))
            saver.restore(sess, checkpoint_path)
        else:
            # purge log directory for run, before running it again
            purge_dir(os.path.join(run_dir, "logs", run_name), r'^events\.out\.tfevents\.\d+')

            if testset:
                purge_dir(os.path.join(run_dir, "logs", run_name, "test"), r'^events\.out\.tfevents\.\d+')

            sess.run(tf.initialize_all_variables())

        # Create summary writer
        train_writer = tf.train.SummaryWriter(os.path.join(run_dir, "logs", run_name), sess.graph)

        if testset:
            test_writer = tf.train.SummaryWriter(os.path.join(run_dir, "logs", run_name, "test"), sess.graph)

        start_time = time.time()
        logger.info("Beginning training. You can watch the training progress by running `tensorboard --logdir {}` and "
                    "pointing your browser to `http://localhost:6006`".format(os.path.join(run_dir, "logs")))

        timings = dict()

        for rep in range(repeat):
            trainset.rewind_batches()
            trainset.shuffle()

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
                    labels_placholder: labels
                }

                tick = time.time()

                # Do it!
                sess.run(train_op, feed_dict=feed_dict)

                timings["trainset_calc"] = time.time() - tick


                # Log it
                tick = time.time()

                summary_str, global_step_val, accuracy_val = \
                    sess.run([train_summary_op, global_step, accuracy], feed_dict=feed_dict)
                train_writer.add_summary(summary_str, global_step_val)
                train_writer.flush()

                timings["trainset_log"] = time.time() - tick

                # when we have a test set, evaluate the model accuracy on the test set
                if testset and batch_idx % TESTSET_EVAL_FREQUENCY == 0:

                    test_timings = {
                        "read_batch": list(),
                        "calc_batch": list()
                    }
                    test_accuracies = []

                    testset.rewind_batches()
                    for test_batch_idx in range(int(testset.N / batchsize)):

                        tick = time.time()
                        labels, boxes = testset.next_batch(batchsize)

                        test_timings['read_batch'].append(time.time() - tick)

                        test_feed_dict = {
                            input_placeholder: boxes,
                            labels_placholder: labels
                        }

                        tick = time.time()

                        test_accuracy_val = sess.run(accuracy, feed_dict=test_feed_dict)
                        test_accuracies.append(test_accuracy_val)

                        test_timings['calc_batch'].append(time.time() - tick)

                        if progress_tracker:
                            progress_tracker.update(batchsize * rep + batch_idx)

                    summary_str = sess.run(test_summary,
                                           feed_dict={
                                               test_accuracy_placeholder: sum(test_accuracies) / len(test_accuracies)})
                    test_writer.add_summary(summary_str, global_step_val)
                    test_writer.flush()

                    timings['testset_read_avg'] = sum(test_timings['read_batch']) / len(test_timings['read_batch'])
                    timings['testset_calc_avg'] = sum(test_timings['calc_batch']) / len(test_timings['calc_batch'])

                    logger.debug("Testset: avg batch read time %f; avg batch calc time %f",
                                 timings['testset_read_avg'], timings['testset_calc_avg'])

                # Save it
                if batch_idx % CHECKPOINT_FREQUENCY == 0:
                    saver.save(sess, checkpoint_path, global_step=global_step)

                logger.debug("Trainset: avg batch read time %f; avg batch calc time %f; avg log time %f",
                             timings['trainset_read'], timings['trainset_calc'], timings['trainset_log'])

                batch_idx += 1

                if progress_tracker:
                    progress_tracker.update(batchsize * rep + batch_idx)

            # Save it again, this time without appending the step number to the filename
            saver.save(sess, checkpoint_path)
            end_time = time.time()

            if batch_idx != 0:
                batchtime = (end_time - start_time) / batch_idx
            else:
                batchtime = (end_time - start_time)

            logger.info("Finished run {:d}. Total time: {:d} s. Time per batch: {:f} s"
                        .format(rep, int(end_time - start_time), batchtime))


if __name__ == "__main__":
    import argparse
    import socket
    from time import strftime

    try:
        import pyprind

        class PyprindProgressTracker:
            def __init__(self):
                # self.current = 0
                self.bar = None

            def init(self, total):
                # self.current = 0
                self.bar = pyprind.ProgPercent(total, monitor=True, update_interval=2)

            def update(self, current=None):
                if self.bar and current:
                    self.bar.update(item_id="Batch {:>5d}".format(current))

            def finish(self):
                if self.bar:
                    print(self.bar)


        progress_tracker = PyprindProgressTracker()
    except ImportError:
        logger.warning("Failed to import pyprind module. Can't show you a pretty progress bar :'( ")
        progress_tracker = None


    def make_default_runname():
        return "{}.{}".format(socket.gethostname(), strftime("%Y%m%dT%H%M%S"))


    try:
        import pyprind
    except ImportError:
        logger.warning("Failed to import pyprind module. Can't show you a pretty progress bar :'( ")
        pyprind = None

    # ========================= Main argument parser ==========================
    parser_top = argparse.ArgumentParser(description='Catalophore neural network training')

    parser_top.add_argument('--log_level', action="store",
                            type=str.upper, dest='log_level',
                            metavar='LOG_LEVEL',
                            choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
                            default=LOGDEFAULT,
                            help='Set log level to be LOG_LEVEL. Can be one of: DEBUG,INFO,WARNING,ERROR,CRITICAL')

    parser_top.add_argument(action='store',
                            type=str, dest='dataset_dir',
                            metavar="DATADIR",
                            help="Dataset directory. This directory contains all the data and metadata files required "
                                 "for training.")

    parser_top.add_argument(action='store', nargs='?',
                            type=str, dest='run_dir',
                            metavar="RUNDIR",
                            help="Run directory. This directory will contain the output of the run.")

    parser_top.add_argument('--name', action='store',
                            type=str, dest='run_name',
                            default=make_default_runname(),
                            metavar="RUN_NAME",
                            help="Training run name")

    parser_top.add_argument('--batchsize', action='store',
                            type=int, dest='batchsize',
                            default=50,
                            metavar="BATCHSIZE",
                            help="Size of training batches.")

    parser_top.add_argument('--repeat', action='store',
                            type=int, dest='repeat',
                            default=1,
                            metavar="REPEATS",
                            help="Number of times to repeat the training")

    parser_top.add_argument('--max_batches', action='store',
                            type=int, dest='max_batches',
                            default=0,
                            metavar="MAX_BATCHES",
                            help="Stop training after at most MAX_BATCHES in each repeat.")

    parser_top.add_argument('--track_accuracy', action='store_true',
                            dest='track_accuracy',
                            help="Track the accuracy of the model on the test set")

    parser_top.add_argument('--continue', action='store_true',
                            dest='cont',
                            help="Pick up training from the last checkpoint, if one exists.")

    args = parser_top.parse_args()

    logging.basicConfig(level=args.log_level, format='%(levelname)1s:%(message)s')

    run_training(args.dataset_dir, args.run_dir, args.run_name,
                 continue_previous=args.cont, batchsize=args.batchsize, max_batches=args.max_batches,
                 repeat=args.repeat,
                 track_test_accuracy=args.track_accuracy,
                 progress_tracker=progress_tracker)

    if progress_tracker:
        progress_tracker.finish()
