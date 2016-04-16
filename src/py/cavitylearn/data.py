import numpy as np
import lzma

import logging

logger = logging.getLogger(__name__)

DTYPE = np.float32


class DataConfig:
    def __init__(self, num_classes, num_props, boxshape):
        self.num_classes = num_classes
        self.num_props = num_props
        self.boxshape = boxshape


class DataSet:
    def __init__(self, labelfile, boxfiles, dataconfig):
        self.N = len(boxfiles)
        self._dataconfig = dataconfig

        # open and load label file right now
        with lzma.open(labelfile) as label_xz:
            self._labels = np.frombuffer(label_xz.read(), dtype=bool).reshape([-1, dataconfig.num_classes])

        if self._labels.shape[0] != len(boxfiles):
            raise TypeError("Number of examples ({}) does not match number of labels ({})".format(len(boxfiles),
                                                                                                  self._labels.shape[
                                                                                                      0]))
        missingfiles = np.zeros([self.N], dtype=bool)

        # loop through all box files, verify that they are there and an XZ file.
        for i, boxfile in enumerate(boxfiles):
            try:
                with lzma.open(boxfile):
                    pass
            except FileNotFoundError:
                missingfiles[i] = True
                logger.warning("Box file not found: {}".format(boxfile))

        num_missing = sum(missingfiles)
        if num_missing:
            logger.warning("{:d} files missing".format(num_missing))

        self._labels = self._labels[missingfiles, :]

        self._boxfiles = [f for i, f in enumerate(boxfiles) if not missingfiles[i]]

        self._last_batch_index = 0

    def rewind_batches(self):
        self._last_batch_index = 0

    def next_batch(self, batch_size):
        next_index = self._last_batch_index + batch_size
        label_slice = self._labels[self._last_batch_index:next_index, :]
        filenames_slice = self._boxfiles[self._last_batch_index:next_index]

        boxes_slice = np.zeros([batch_size, self._dataconfig.num_props,
                                self._dataconfig.boxshape[0],
                                self._dataconfig.boxshape[1],
                                self._dataconfig.boxshape[2]], dtype=DTYPE)

        for i, f in enumerate(filenames_slice):
            with lzma.open(f) as xzfile:
                boxes_slice[i, :, :, :] = np.frombuffer(xzfile.read(), dtype=DTYPE).reshape([
                    self._dataconfig.num_props,
                    self._dataconfig.boxshape[0],
                    self._dataconfig.boxshape[1],
                    self._dataconfig.boxshape[2]])

        self._last_batch_index = next_index

        return label_slice, boxes_slice
