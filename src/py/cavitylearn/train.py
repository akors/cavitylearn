import socket

import os
import sys
import logging
import configparser
import subprocess

import re
import time
import math
from collections import OrderedDict

import tensorflow as tf
from tensorflow.python.client import timeline

# I know I have to upgrade, but please leave me alone
from tensorflow.python.util import module_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

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
    "checkpoint_frequency_labelled": 1000,
    "checkpoint_frequency": 30,
    "checkpoint_max_to_keep": 4,
    "checkpoint_keep_every_n_hours": 2,
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


def get_git_revision_short_hash():
    wd = os.path.dirname(__file__)
    try:
        result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], stdout=subprocess.PIPE, cwd=wd)
    except FileNotFoundError:
        # Could not find git binary
        return None

    if result.returncode != 0:
        return None
    else:
        return result.stdout.decode('ascii').strip()


def purge_dir(directory, pattern):
    """Purge all files in a directory that match a regular expression pattern.
    This does not recurse into subdirectories.

    :param directory: Directory to be purged
    :param pattern: The regular expression pattern as string
    """
    for f in os.listdir(directory):
        if re.search(pattern, f):
            os.remove(os.path.join(directory, f))


def write_runinfo(runinfo_path, runinfo):
    with open(runinfo_path, "wt") as outfile:
        for k, v in runinfo.items():
            outfile.write("{}\t{}\n".format(k, v))


def pretty_print_runinfo(runinfo):
    max_keylength = max((len(k) for k in runinfo.keys()))

    runinfo_string = ""

    for k, v in runinfo.items():
        runinfo_string += '\t{:<{width}}: {}\n'.format(k, v, width=max_keylength)

    return runinfo_string


def run_training(dataset_dir, run_dir, run_name, continue_previous=False,
                 learnrate=1e-4, learnrate_decay=1.0, learnrate_decay_freq=0, keep_prob_conv=1,
                 keep_prob_hidden=0.75, l2reg_scale=0.0, l2reg_scale_conv=0.0, batchsize=50, epochs=1, batches=None,
                 track_test_accuracy=False, num_threads=None, track_timeline=False, progress_tracker=None):

    dataconfig = data.read_dataconfig(os.path.join(dataset_dir, "datainfo.ini"))
    testing_frequency = int(config[THISCONF]['testing_frequency'])

    config_proto_dict = {}
    config_proto_dict["gpu_options"] = tf.GPUOptions(allow_growth=True)
    if num_threads is not None:
        config_proto_dict["inter_op_parallelism_threads"] = num_threads
        config_proto_dict["intra_op_parallelism_threads"] = num_threads

    # define input tensors
    with tf.variable_scope("input"):
        label_placeholder = tf.placeholder(tf.int32, shape=[None], name="labels")
        boxes_placeholder = tf.placeholder(tf.float32, shape=[None, dataconfig.boxshape[0], dataconfig.boxshape[1],
                                                              dataconfig.boxshape[2], dataconfig.num_props],
                                           name="boxes")

        p_keep_conv_placeholder = tf.placeholder_with_default(1.0, shape=None, name="p_conv")
        p_keep_hidden_placeholder = tf.placeholder_with_default(1.0, shape=None, name="p_fc")

        tf.summary.scalar('dropout_keepprob_conv', p_keep_conv_placeholder)
        tf.summary.scalar('dropout_keepprob_fc', p_keep_hidden_placeholder)

    tf.add_to_collection("input", label_placeholder)
    tf.add_to_collection("input", boxes_placeholder)
    tf.add_to_collection("input", p_keep_conv_placeholder)
    tf.add_to_collection("input", p_keep_hidden_placeholder)

    # global step variable
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # prediction, loss and training operations
    logits = catalonet0.inference(boxes_placeholder, dataconfig,
                                  p_keep_hidden=p_keep_hidden_placeholder, p_keep_conv=p_keep_conv_placeholder,
                                  l2scale=l2reg_scale, l2scale_conv=l2reg_scale_conv)
    loss = catalonet0.loss(logits, label_placeholder)
    train_op = catalonet0.train(loss_op=loss, learning_rate=learnrate, learnrate_decay=learnrate_decay,
                                learnrate_decay_freq=learnrate_decay_freq, global_step=global_step)

    # required by the evaluate functions
    tf.add_to_collection('train_op', train_op)

    # log the training accuracy
    accuracy_op = catalonet0.accuracy(logits, label_placeholder)
    tf.summary.scalar("accuracy/accuracy_train", accuracy_op)
    train_summary_op = tf.summary.merge_all()

    logger.info("Loading datasets.")

    tick = time.time()

    # Get all datasets in the input directory
    datasets = data.load_datasets(
        os.path.join(dataset_dir, "labels.txt"),
        os.path.join(dataset_dir, "boxes"),
        dataconfig,
        datasets=("train", "test", ""),
        start_workers=False,
        verify=False)

    logger.debug("load_datasets: %f", time.time() - tick)

    # get training dataset. If there isn't a dataset called "train", take all examples in the dataset
    if "train" in datasets:
        trainset = datasets["train"]
    else:
        trainset = datasets[""]

    # If we have a test set, validate against it. If not, ignore it.
    if track_test_accuracy:
        if "test" not in datasets:
            raise ValueError("Test set accuracy tracking requested, but test set not found")

        testset = datasets["test"]
    else:
        testset = None

    if not batches:
        batches = math.ceil(trainset.N / batchsize) * epochs
    else:
        if epochs > 1:
            logger.warning("Both epochs and batches were specified. I will feed at most %d batches in total", batches)

    # calculate our workload
    batches_in_trainset = math.ceil(trainset.N / batchsize)

    if testset:
        batches_in_testset = math.ceil(testset.N / batchsize)
        number_of_testset_evaluations = math.ceil(batches / testing_frequency)

        total_batches = batches + batches_in_testset * number_of_testset_evaluations

        logger.debug("batches_in_trainset %d; train batches %d ; batches_in_testset %d ; "
                     "number_of_testset_evaluations %d ; total_batches %d ; ",
                     batches_in_trainset, batches, batches_in_testset, number_of_testset_evaluations, total_batches)

        with tf.name_scope("test_accuracy"):
            streaming_accuracy_result, streaming_accuracy_update_op = tf.metrics.mean(accuracy_op)
        test_summary = tf.summary.scalar("accuracy/accuracy_test", streaming_accuracy_result)
    else:
        total_batches = batches
        batches_in_testset = 0

    # initialize progress tracker
    if progress_tracker:
        progress_tracker.init(total_batches)

    # create run information
    runinfo_path = os.path.join(run_dir, "runinfo." + run_name + ".txt")
    runinfo = OrderedDict()

    rev = get_git_revision_short_hash()

    runinfo["name"] = run_name
    runinfo["hostname"] = socket.gethostname()
    if rev is not None:
        runinfo["revision"] = rev
    runinfo["tensorflow_version"] = tf.__version__
    if hasattr(tf, "__git_version__"):
        runinfo["tensorflow_git_version"] = tf.__git_version__
    runinfo["input_path"] = dataset_dir
    runinfo["output_path"] = run_dir
    runinfo["batchsize"] = batchsize
    runinfo["batches"] = batches
    runinfo["epochs"] = batches * batchsize / trainset.N
    runinfo["learnrate"] = learnrate
    runinfo["learnrate_decay"] = learnrate_decay
    runinfo["learnrate_decay_freq"] = learnrate_decay_freq
    runinfo["keepprob_conv"] = keep_prob_conv
    runinfo["keepprob_hidden"] = keep_prob_hidden
    runinfo["l2reg_scale"] = l2reg_scale
    runinfo["l2reg_scale_conv"] = l2reg_scale_conv

    # Create or purge checkpoint directory if not continuing
    if os.path.isdir(os.path.join(run_dir, "checkpoints")):
        if not continue_previous:
            purge_dir(os.path.join(run_dir, "checkpoints"),
                      r'^{run_name}(-\d+)?(\.meta|\.index|\.latest|\.data-\d+-of-\d+)?$'.format(run_name=run_name))
    else:
        os.makedirs(os.path.join(run_dir, "checkpoints"))

    # create or purge log directory
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

    saver = tf.train.Saver(max_to_keep=int(config[THISCONF]['checkpoint_max_to_keep']),
                           keep_checkpoint_every_n_hours=int(config[THISCONF]['checkpoint_keep_every_n_hours']))

    # launch input file read workers
    trainset.start_worker()
    if testset is not None:
        testset.start_worker()

    checkpoint_path = os.path.join(run_dir, "checkpoints", run_name)
    with tf.Session(config=tf.ConfigProto(**config_proto_dict)) as sess:
        if track_timeline:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        if continue_previous:
            try:
                saver.restore(sess, checkpoint_path)
                logger.info("Found training checkpoint file `{}`, continuing training. ".format(checkpoint_path))
            except tf.errors.NotFoundError:
                logger.warning("Training continuation was requested, but checkpoint file `{}` was not found. "
                               "Restarting training.".format(checkpoint_path))
                sess.run(tf.global_variables_initializer())
        else:
            sess.run(tf.global_variables_initializer())

        write_runinfo(runinfo_path, runinfo)

        # Create summary writer
        summary_writer = tf.summary.FileWriter(os.path.join(run_dir, "logs", run_name), sess.graph)

        if testset:
            # running mean has local variables. When testing, initialize those
            local_variables_initializer_op = tf.local_variables_initializer()
            sess.run(local_variables_initializer_op)

        logger.info(
            "Beginning training. You can watch the training progress by running `tensorboard --logdir {}`".format(
                os.path.join(run_dir, "logs")))

        logger.info("Run information: \n%s", pretty_print_runinfo(runinfo))

        # Finalize graph before training so we don't accidentally add more Ops
        sess.graph.finalize()

        # init loop variables
        batchcount = 0  # Actual number of batches evaluated (training + testing)
        epoch = 0  # Number of times the training set was fed into training
        train_accuracy_val = 0.0 # Current accuracy value of training
        test_accuracy_val = None  # Accuracy value of the test set when running with track_test_accuracy=True

        timings = dict()  # debug timings
        training_start_time = time.time()  # start time of the whole training
        epoch_start_time = training_start_time  # start time of the epoch

        batch_idx = 0
        for batch_idx in range(batches):

            # get training data
            tick = time.time()

            labels, boxes = trainset.next_batch(batchsize)

            timings["trainset_read"] = time.time() - tick

            # When we ran out of training examples, shuffle and rewind the training set
            if len(labels) == 0:
                epoch_end_time = time.time()

                # save model without global step number
                saver.save(sess, checkpoint_path, latest_filename="{:s}.latest".format(run_name))

                # rewind
                tick = time.time()

                trainset.shuffle(norestart=True)
                trainset.rewind_batches(norestart=False)

                timings["trainset_rewind"] = time.time() - epoch_end_time

                # read first data set

                # get training data
                labels, boxes = trainset.next_batch(batchsize)

                timings["trainset_read"] = time.time() - tick

                # check that we actually got data
                if len(labels) == 0 or len(boxes) == 0:
                    raise ValueError("Training set does not contain any examples")

                # report status
                logger.info("")
                logger.info("Finished epoch {:d}. Total time: {:d} s. Time per batch: {:f} s".format(
                    epoch + 1, int(epoch_end_time - epoch_start_time),
                    (epoch_end_time - epoch_start_time) / batches_in_trainset))

                epoch_start_time = time.time()

                epoch += 1

            # Feed it!
            feed_dict = {
                boxes_placeholder: boxes,
                label_placeholder: labels,
                p_keep_conv_placeholder: keep_prob_conv,
                p_keep_hidden_placeholder: keep_prob_hidden
            }

            tick = time.time()

            # Do it!
            _, summary_str, train_accuracy_val, global_step_val = sess.run([train_op, train_summary_op, accuracy_op, global_step],
                                                       feed_dict=feed_dict,
                                                       options=run_options, run_metadata=run_metadata)

            timings["trainset_calc"] = time.time() - tick

            tick = time.time()

            # Report it!
            summary_writer.add_summary(summary_str, global_step_val)
            summary_writer.flush()

            timings["trainset_log"] = time.time() - tick

            # when we have a test set, evaluate the model accuracy on the test set
            if testset and batch_idx % testing_frequency == 0:

                test_timings = {
                    "read_batch": list(),
                    "calc_batch": list(),
                    "eval_batch": list()
                }

                examples_so_far = 0

                # Reset 'total' and 'counts' variables of the streaming mean op
                sess.run(local_variables_initializer_op)

                for test_batch_idx in range(batches_in_testset):

                    tick = time.time()
                    labels, boxes = testset.next_batch(batchsize)

                    test_timings['read_batch'].append(time.time() - tick)

                    test_feed_dict = {
                        boxes_placeholder: boxes,
                        label_placeholder: labels
                    }

                    tick = time.time()

                    # calculate moving average
                    test_accuracy_val = \
                    sess.run(streaming_accuracy_update_op, feed_dict=test_feed_dict,
                             options=run_options, run_metadata=run_metadata)

                    # examples_so_far += len(labels)
                    # logger.debug("test_accuracy_val: %f; examples_so_far: %d; test_accuracy: %f; len(labels): %d",
                    #              test_accuracy_val, examples_so_far, test_accuracy, len(labels))

                    test_timings['calc_batch'].append(time.time() - tick)

                    if progress_tracker:
                        progress_tracker.update(f"Test batch {test_batch_idx+1}/{batches_in_testset}, Accuracy={test_accuracy_val:.3f}")

                    logger.debug("")
                    logger.debug("test: read_batch: %f ; calc_batch %f",
                                 test_timings['read_batch'][-1], test_timings['calc_batch'][-1])

                    batchcount += 1
                    pass

                # rewind test batches after using them
                testset.rewind_batches()

                tick = time.time()

                summary_str = sess.run(test_summary)

                test_timings['eval_batch'].append(time.time() - tick)

                summary_writer.add_summary(summary_str, global_step_val)
                summary_writer.flush()

                timings['testset_read_avg'] = sum(test_timings['read_batch']) / len(test_timings['read_batch'])
                timings['testset_calc_avg'] = sum(test_timings['calc_batch']) / len(test_timings['calc_batch'])
                timings['testset_eval_avg'] = sum(test_timings['eval_batch']) / len(test_timings['eval_batch'])

                logger.debug("")
                logger.debug("testset_read_avg: %(testset_read_avg)f; testset_calc_avg: %(testset_calc_avg)f; "
                             "testset_eval_avg: %(testset_eval_avg)f",
                             timings)

            # Save global_step-labelled checkpoint
            if batch_idx % int(config[THISCONF]['checkpoint_frequency_labelled']) == 0:
                saver.save(sess,
                           checkpoint_path, global_step=global_step, latest_filename="{:s}.latest".format(run_name))

            # Save running checkpoint
            if batch_idx % int(config[THISCONF]['checkpoint_frequency']) == 0:
                saver.save(sess, checkpoint_path, latest_filename="{:s}.latest".format(run_name))

            # write the timeline data information if available
            if track_timeline:
                # Create the Timeline object, and write it to a json
                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                with open(os.path.join(run_dir, 'timeline.' + run_name + '.json'), 'w') as f:
                    f.write(ctf)

            logger.debug("")
            logger.debug(
                "trainset_read: %(trainset_read)f; trainset_calc: %(trainset_calc)f; "
                "trainset_log %(trainset_log)f",
                timings)

            batchcount += 1

            if progress_tracker:
                progress_tracker.update(
                    f"Tr. Batch {batch_idx+1:>3d}/{batches}, Ep {epoch+1:>2d}. Accuracy={train_accuracy_val:.3f}"
                )

        logger.debug("batchount: %d", batchcount)

        # Save final model, this time without appending the step number to the filename
        saver.save(sess, checkpoint_path, latest_filename="{:s}.latest".format(run_name))

        end_time = time.time()

        logger.info("Training completed. Total time: {:d} s. Time per batch: {:f} s".format(
            int(end_time - training_start_time),
            (end_time - training_start_time) / (batch_idx + 1)))
