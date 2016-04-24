import os
import numpy as np
import lzma

import logging

from . import converter

logger = logging.getLogger(__name__)

DTYPE = np.float32


class DataConfig:
    def __init__(self, classes, num_props, boxshape):
        self.num_classes = len(classes)
        self.classes = list(classes)
        self.num_props = num_props
        self.boxshape = list(boxshape)


class DataSet:
    def __init__(self, labelfile, boxdir, dataconfig, shuffle=True):
        self._dataconfig = dataconfig

        labels_uuids = [row.strip().split('\t') for row in labelfile]

        missingfiles = np.zeros([len(labels_uuids)], dtype=bool)

        num_missing = 0
        boxfiles = list()
        labels = list()

        # loop through all box files, verify that they are there and an XZ file.
        for i, row in enumerate(labels_uuids):
            boxfile = os.path.join(boxdir, row[0] + ".xz")
            try:
                with lzma.open(boxfile):
                    pass

                boxfiles.append(boxfile)
                labels.append(row[1])

            except FileNotFoundError:
                num_missing += 1
                logger.warning("Box file not found: {}".format(boxfile))

        self.N = len(boxfiles)

        if num_missing:
            logger.warning("{:d} files missing".format(num_missing))

        self._labels = converter.labels_to_array(labels, dataconfig.classes)

        if shuffle:
            rand_order = np.random.permutation(self.N)
            self._labels = self._labels[rand_order]
            self._boxfiles = [boxfiles[i] for i in rand_order]
        else:
            self._boxfiles = boxfiles

        self._last_batch_index = 0
        pass

    @property
    def labels(self):
        return self._labels.copy()

    def rewind_batches(self, last_index=0):
        self._last_batch_index = last_index
        pass

    def next_batch(self, batch_size):
        next_index = self._last_batch_index + batch_size
        if next_index > self.N:
            batch_size = self.N - self._last_batch_index
            next_index = self.N

        label_slice = self._labels[self._last_batch_index:next_index, :]
        filenames_slice = self._boxfiles[self._last_batch_index:next_index]

        boxes_slice = np.zeros([batch_size,
                                self._dataconfig.boxshape[0],
                                self._dataconfig.boxshape[1],
                                self._dataconfig.boxshape[2],
                                self._dataconfig.num_props], dtype=DTYPE)

        for i, f in enumerate(filenames_slice):
            with lzma.open(f) as xzfile:
                xzfile_array = np.frombuffer(xzfile.read(), dtype=DTYPE)
                boxes_slice[i, :, :, :] = xzfile_array.reshape([
                    self._dataconfig.boxshape[0],
                    self._dataconfig.boxshape[1],
                    self._dataconfig.boxshape[2],
                    self._dataconfig.num_props])

        self._last_batch_index = next_index

        return label_slice, boxes_slice


def learning_datasets(labelfile, dataconfig, validation_part, test_part, shuffle=True):
    if not (isinstance(validation_part, np.float) and validation_part > 0) or \
            not (isinstance(test_part, np.float) and test_part > 0):
        raise ValueError("validation_part and test_part must be positive floating point numbers between 0 and 1")

    if validation_part + test_part >= 1.0:
        raise ValueError("Validation and Test partitions cannot make up more than 100% of the data set")

    all_labels_uuids = [row.strip().split("\t") for row in labelfile]



