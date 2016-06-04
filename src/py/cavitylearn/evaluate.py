import os
import logging
import math
import tensorflow as tf
import numpy as np
import time

from . import data
from . import catalonet0


# =============================== set up logging ==============================

LOGDEFAULT = logging.INFO
logger = logging.getLogger(__name__)


def calc_metrics(dataset_dir, checkpoint_path, dataset_names=[], batchsize=50, progress_tracker=None):
    dataconfig = data.read_dataconfig(os.path.join(dataset_dir, "datainfo.ini"))

    logger.info("Loading datasets")

    tick = time.time()

    # Get all datasets in the input directory
    datasets = data.load_datasets(
        os.path.join(dataset_dir, "labels.txt"),
        os.path.join(dataset_dir, "boxes"),
        dataconfig,
        shuffle=False,
        verify=False)

    logger.debug("load_datasets: %f", time.time() - tick)

    # root dataset should always be in datasets
    assert "" in datasets

    # rename root dataset
    if len(datasets) == 1:
        datasets["root"] = datasets[""]
        del datasets[""]

    # Only evaluate on requested datasets
    if dataset_names:
        dss = dict()
        for name in dataset_names:
            if name not in datasets:
                raise KeyError("Dataset %s not found found in dataset directory." % name)

            dss[name] = datasets[name]

        datasets = dss
        del dss

    with tf.variable_scope("input"):
        label_placeholder = tf.placeholder(tf.int32, shape=[None], name="labels")
        input_placeholder = tf.placeholder(tf.float32, shape=[None, dataconfig.boxshape[0], dataconfig.boxshape[1],
                                                              dataconfig.boxshape[2], dataconfig.num_props]
                                           , name="boxes")

        p_keep_conv_placeholder = tf.constant(1.0, tf.float32, name="p_conv")
        p_keep_hidden_placeholder = tf.constant(1.0, tf.float32, name="p_fc")

    # initialize progress tracker
    if progress_tracker:
        progress_tracker.init(sum(math.ceil(ds.N / batchsize) for ds in datasets.values()))

    logits = catalonet0.inference(input_placeholder, dataconfig,
                                  p_keep_conv=p_keep_conv_placeholder, p_keep_hidden=p_keep_hidden_placeholder)
    predicted = tf.arg_max(logits, 1)

    with tf.variable_scope("metrics"):
        pred = tf.placeholder(tf.int32, shape=[None], name="predicted")
        targ = tf.placeholder(tf.int32, shape=[None], name="labels")

        #confusion_matrix_op = tf.contrib.metrics.confusion_matrix(pred, targ,
        #                                                          num_classes=dataconfig.num_classes)

    result_dict = dict()
    timings = dict()  # debug timings
    saver = tf.train.Saver()

    for ds_name, ds in datasets.items():

        ds_start_time = time.time()

        batches = math.ceil(ds.N / batchsize)

        with tf.Session() as sess:
            saver.restore(sess, checkpoint_path)

            all_predicted = np.full([ds.N], -1, dtype=np.int32)
            all_labels = np.full([ds.N], -1, dtype=np.int32)

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
                    input_placeholder: boxes
                })

                all_labels[example_idx:example_idx+len(labels)] = labels
                all_predicted[example_idx:example_idx+len(labels)] = pred_slice


                # confusion_matrix_sum += confm

                timings["batch_calc"] = time.time() - tick

                if progress_tracker:
                    progress_tracker.update()

                logger.debug(
                    "batch_read: %(batch_read)f; batch_calc: %(batch_calc)f", timings)

                example_idx += len(labels)

            #confusion_matrix = sess.run(confusion_matrix_op, feed_dict={
            #    pred: predicted,
            #    targ: all_labels
            #})
            #confusion_matrix = tf.contrib.metrics.confusion_matrix(predicted, all_labels,
            #                                                          num_classes=dataconfig.num_classes)

            #print(confusion_matrix)

        ds_end_time = time.time()

        logger.info("Finished evaluating dataset {:s}. Total time: {:d} s. Time per batch: {:f} s".format(
            ds_name, int(ds_end_time - ds_start_time),
            (ds_end_time - ds_start_time) / batches))

        accuracy = float(np.sum(all_labels == all_predicted) / len(all_labels))

        tick = time.time()

        confusion_matrix = np.zeros([dataconfig.num_classes, dataconfig.num_classes], dtype=np.float32)
        for i in range(dataconfig.num_classes):
            for j in range(dataconfig.num_classes):
                true_idx = all_labels == i
                pred_idx = all_predicted == j

                confusion_matrix[i, j] = np.sum(true_idx & pred_idx)

        logger.debug("calc_confusion_matrix: %f", time.time() - tick)

        print("Dataset", ds_name, ":")
        print("Accuracy: %.2f %%" % (accuracy*100))
        print("confusion_matrix:\n", confusion_matrix)

        result_dict[ds_name] = {
            "accuracy": accuracy,
            "confusion_matrix": confusion_matrix
        }

    return result_dict
