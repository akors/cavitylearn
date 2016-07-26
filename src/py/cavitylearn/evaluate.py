import configparser
import os
import logging
import math
import tensorflow as tf
import numpy as np
import time

from . import data


THISCONF = 'cavitylearn-train'
config = configparser.ConfigParser(interpolation=None)

# default config values
config[THISCONF] = {
    "polling_interval": 10
}

# =============================== set up logging ==============================

LOGDEFAULT = logging.INFO
logger = logging.getLogger(__name__)


def calc_metrics(dataset_dir, checkpoint_path, dataset_names=list(), batchsize=50, num_threads=None,
                 progress_tracker=None):

    datasets, saver = _prep_eval(checkpoint_path, dataset_dir, dataset_names)

    # initialize progress tracker
    if progress_tracker:
        progress_tracker.init(sum(math.ceil(ds.N / batchsize) for ds in datasets.values()))

    result_dict, global_step_val = _do_one_eval(checkpoint_path=checkpoint_path, datasets=datasets,
                                                saver=saver, batchsize=batchsize, num_threads=num_threads,
                                                progress_tracker=progress_tracker)

    return result_dict


def watch_training(dataset_dir, checkpoint_path, logdir, name=None, dataset_names=list(),
                   max_time=0, max_unchanged_time=1800, wait_for_checkpoint=False,
                   batchsize=50, num_threads=None):

    polling_interval = int(config[THISCONF]['polling_interval'])

    # if the run name was not specified, try to deduce it from the checkpoint file string
    if name is None:
        name = os.path.basename(checkpoint_path)

    start_time = time.time()

    # if requested, wait for the checkpoint file
    if not os.path.exists(checkpoint_path) and wait_for_checkpoint:
        logger.info("Checkpoint file does not exist yet. Waiting for it to come into existance")

        # wait for the file to start existing
        while True:
            if os.path.exists(checkpoint_path):
                logger.info("Checkpoint file was just created! Proceeding with evaluation")
                break

            wait_time = time.time() - start_time

            if (max_time != 0 and wait_time > max_time) or \
                (max_unchanged_time != 0 and wait_time > max_unchanged_time):
                logger.warning("I have waited for %d seconds for the checkpoint file, but now I don't want to anymore",
                               wait_time)
                return

            time.sleep(polling_interval)

    datasets, saver = _prep_eval(checkpoint_path, dataset_dir, dataset_names, logdir)

    # create loggind directories and writers
    summary_writers = dict()
    if logdir:
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        if not os.path.isdir(logdir):
            raise ValueError("log directory `%s` existst, but it is not a directory" % logdir)

        for ds_name in datasets.keys():
            d = os.path.join(logdir, name, ds_name)
            if not os.path.exists(d):
                os.makedirs(d)

            if not os.path.isdir(d):
                raise ValueError("log directory `%s` existst, but it is not a directory" % d)

            summary_writers[ds_name] = tf.train.SummaryWriter(d)

    checkpoint_last_modified = 0
    modified_time = checkpoint_last_modified
    while True:
        try:
            modified_time = os.path.getmtime(checkpoint_path)
        except FileNotFoundError:
            if not wait_for_checkpoint:
                raise
            else:
                logger.debug("Checkpoint file has disappeared! Proceeding as though there were no changes.")

        # if the checkpoint hasn't changed, evaluate termination conditions
        if checkpoint_last_modified == modified_time:
            if max_unchanged_time != 0 and time.time() - modified_time > max_unchanged_time:
                logger.info("Checkpoint file hasn't been updated in over %d seconds. I'm stopping the evaluation now.",
                            max_unchanged_time)
                return

            if max_time != 0 and (time.time() - start_time) > max_time:
                logger.info("Maximum runtime of %d seconds exceeding. I'm stopping the evaluation now.",
                            max_unchanged_time)
                return

            # If we're allowed to continue, wait and try again
            logger.debug("No changes to checkpoint file. Going back to sleep.")
            time.sleep(polling_interval)
            continue

        logger.info("Checkpoint was updated, recalculating accuracy")

        checkpoint_last_modified = modified_time

        result_dict, global_step_val = _do_one_eval(checkpoint_path=checkpoint_path, datasets=datasets,
                                   saver=saver, batchsize=batchsize, num_threads=num_threads)

        logger.info("Performance after %d steps:", global_step_val)
        for ds_name, res in result_dict.items():
            logger.info("Dataset `%s`: accuracy: %f", ds_name, res['accuracy'])

            summary = tf.Summary(value=[
                tf.Summary.Value(tag="accuracy", simple_value=res['accuracy'])
            ])

            summary_writers[ds_name].add_summary(summary, global_step=global_step_val)

        for ds in datasets.values():
            ds.rewind_batches()


def _prep_eval(checkpoint_path, dataset_dir, dataset_names, logdir=None):
    dataconfig = data.read_dataconfig(os.path.join(dataset_dir, "datainfo.ini"))

    # replace "root" dataset with empty string, if its there
    if dataset_names is not None:
        if "root" in dataset_names:
            dataset_names[dataset_names.index("root")] = ""

    logger.info("Loading datasets")
    tick = time.time()

    # Get all datasets in the input directory
    datasets = data.load_datasets(
        os.path.join(dataset_dir, "labels.txt"),
        os.path.join(dataset_dir, "boxes"),
        dataconfig,
        datasets=dataset_names,
        shuffle=False,
        verify=False)
    logger.debug("load_datasets: %f", time.time() - tick)

    # rename root dataset back
    if dataset_names is not None and "" in dataset_names:
        datasets["root"] = datasets.pop("")

    # Verify that all requested datasets are there
    if dataset_names:
        for name in dataset_names:
            if name not in datasets:
                raise KeyError("Dataset %s not found found in dataset directory." % name)

    saver = tf.train.import_meta_graph(checkpoint_path + '.meta')

    return datasets, saver


def _do_one_eval(checkpoint_path, datasets, saver, batchsize, num_threads=None, progress_tracker=None):

    config_proto_dict = {}
    if num_threads is not None:
        config_proto_dict["inter_op_parallelism_threads"] = num_threads
        config_proto_dict["intra_op_parallelism_threads"] = num_threads

    result_dict = dict()
    timings = dict()  # debug timings

    with tf.Session(config=tf.ConfigProto(**config_proto_dict)) as sess:
        logits = sess.graph.get_tensor_by_name('softmax_linear/softmax_linear:0')
        predicted = tf.arg_max(logits, 1)
        label_placeholder = sess.graph.get_tensor_by_name("input/labels:0")
        boxes_placeholder = sess.graph.get_tensor_by_name("input/boxes:0")

        global_step = sess.graph.get_tensor_by_name("global_step:0")

        saver.restore(sess, checkpoint_path)

        for ds_name, ds in datasets.items():

            ds_start_time = time.time()

            batches = math.ceil(ds.N / batchsize)

            confusion_matrix = np.zeros([ds.dataconfig.num_classes, ds.dataconfig.num_classes], dtype=np.int32)
            example_idx = 0
            for batch_idx in range(batches):

                tick = time.time()
                labels, boxes = ds.next_batch(batchsize)
                timings["batch_read"] = time.time() - tick

                if len(labels) == 0:
                    break

                tick = time.time()
                pred_slice = sess.run(predicted, feed_dict={
                    label_placeholder: labels,
                    boxes_placeholder: boxes
                })

                confm_batch = np.zeros([ds.dataconfig.num_classes, ds.dataconfig.num_classes], dtype=np.int32)
                for i in range(ds.dataconfig.num_classes):
                    true_idx = labels == i  # indices of examples where the true label was "i"
                    for j in range(ds.dataconfig.num_classes):
                        pred_idx = pred_slice == j  # indices of examples where the predicted label was "j"

                        # calculate confusion matrix entry
                        confm_batch[i, j] = np.sum(true_idx & pred_idx)

                # add batch confusion matrix to total confusion matrix
                confusion_matrix += confm_batch

                timings["batch_calc"] = time.time() - tick

                if progress_tracker:
                    progress_tracker.update()

                logger.debug("")
                logger.debug(
                    "batch_read: %(batch_read)f; batch_calc: %(batch_calc)f", timings)

                example_idx += len(labels)

            ds_end_time = time.time()

            logger.info("Finished evaluating dataset {:s}. Total time: {:d} s. Time per batch: {:f} s".format(
                ds_name, int(ds_end_time - ds_start_time),
                (ds_end_time - ds_start_time) / batches))

            tick = time.time()

            accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

            precision = np.zeros([ds.dataconfig.num_classes], dtype=np.float32)
            recall = np.zeros([ds.dataconfig.num_classes], dtype=np.float32)
            for i in range(ds.dataconfig.num_classes):
                precision[i] = confusion_matrix[i, i] / np.sum(confusion_matrix[:, i])
                recall[i] = confusion_matrix[i, i] / np.sum(confusion_matrix[i, :])

            f_score = (precision * recall) / (precision + recall)
            g_score = np.sqrt(precision * recall)

            logger.debug("calc_metrics: %f", time.time() - tick)

            result_dict[ds_name] = {
                "confusion_matrix": confusion_matrix,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f_score": f_score,
                "g_score": g_score
            }

        global_step_val = sess.run(global_step)

        return result_dict, global_step_val
