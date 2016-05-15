import os
import shutil
import re
import numpy as np
import lzma

import configparser
import logging

from collections import OrderedDict

from . import converter

logger = logging.getLogger(__name__)

DTYPE = np.float32


class DataConfig:
    def __init__(self, classes, num_props, boxshape, dtype):
        self.num_classes = len(classes)
        self.classes = list(classes)
        self.num_props = num_props
        self.boxshape = list(boxshape)
        self.dtype = dtype


DATACONFIG_SECTION = 'dataconfig'


def read_dataconfig(configfile):
    config = configparser.ConfigParser()

    if isinstance(configfile, str):
        configfile = open(configfile, "rt")

    try:
        config.read_file(configfile)

        classes = [c.strip() for c in config[DATACONFIG_SECTION]["classes"].split(',')]
        properties = [p.strip() for p in config[DATACONFIG_SECTION]["proplist"].split(',')]
        shape = [int(s) for s in config[DATACONFIG_SECTION]["shape"].split(',')]

        dtype_str = config[DATACONFIG_SECTION]["dtype"]
        if dtype_str == "float32":
            dtype = np.float32
        else:
            raise ValueError("Unkown data type `{}` in dataconfig file".format(dtype_str))

        return DataConfig(
            classes=classes,
            num_props=len(properties),
            boxshape=shape,
            dtype=dtype
        )

    finally:
        configfile.close()

BOX_SUFFIX = '.box.xz'
RE_BOXFILE = re.compile('\.box\.xz$')


class DataSet:
    def __init__(self, labelfile, boxfiles, dataconfig, shuffle=True):
        self._dataconfig = dataconfig

        if isinstance(labelfile, str):
            with open(labelfile, "rt") as labelfile:
                label_list = [row.strip().split('\t') for row in labelfile]
        else:
            label_list = [row.strip().split('\t') for row in labelfile]

        label_dict = {
            entry[0]: entry[1]
            for entry in label_list
        }

        boxfiles_labels = OrderedDict()

        # loop through all box files, verify that they are there, an XZ file and have an entry in the label file
        for boxfile in boxfiles:
            try:
                with lzma.open(boxfile):
                    pass

                # get the name of the box: get basename, delete box suffix and look it up in the label list
                boxfile_name = os.path.basename(boxfile)
                if boxfile_name.endswith(BOX_SUFFIX):
                    boxfile_name = boxfile_name[:-len(BOX_SUFFIX)]

                if boxfile_name not in label_dict:
                    logger.warning("Box file `{}` not found in label file.".format(boxfile))
                    continue

                boxfiles_labels[boxfile_name] = label_dict[boxfile_name]

            except FileNotFoundError:
                logger.warning("Box file not found: {}".format(boxfile))

        self.N = len(boxfiles_labels)

        logger.debug("{:d} box files found from {:d} labels in list".format(self.N, len(label_dict)))

        self._labels = converter.labels_to_array(list(boxfiles_labels.values()), dataconfig.classes)
        self._boxfiles = boxfiles

        if shuffle:
            self.shuffle()

        self._last_batch_index = 0
        pass

    @property
    def labels(self):
        return self._labels.copy()

    def shuffle(self):
        rand_order = np.random.permutation(self.N)
        self._labels = self._labels[rand_order]
        self._boxfiles = [self._boxfiles[i] for i in rand_order]

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


class DataSets:
    def __init__(self, labelfile, boxdir, dataconfig, shuffle=True):

        self.__datasets = {
        }

        rootfiles = list()

        # walk the box directory. Create dataset for each directory that contains '.box.xz' files.
        for root, dirs, files in os.walk(boxdir):
            # accumulate all boxfiles
            boxfiles = [os.path.join(root, boxfile) for boxfile in files if RE_BOXFILE.search(boxfile)]

            if not len(boxfiles):
                continue

            # add files to current dataset
            self.__datasets[os.path.basename(root)] = DataSet(labelfile, boxfiles, dataconfig, shuffle=shuffle)

            # add files to root dataset
            rootfiles.extend(boxfiles)

        self.__datasets[""] = DataSet(labelfile, rootfiles, dataconfig, shuffle=shuffle)

    @property
    def datasets(self):
        return self.__datasets


def split_datasets(labelfile, rootdir, dataconfig, test_part, validation_part=0, shuffle=True):
    if not (isinstance(validation_part, np.float) and validation_part > 0) or \
            not (isinstance(test_part, np.float) and test_part > 0):
        raise ValueError("validation_part and test_part must be positive floating point numbers between 0 and 1")

    if validation_part + test_part >= 1.0:
        raise ValueError("Validation and Test partitions cannot make up more than 100% of the data set")

    # Collect all box files in this directory recursively
    allfiles = list()
    for root, dirs, files in os.walk(rootdir):
        boxfiles = [os.path.join(root, boxfile) for boxfile in files if RE_BOXFILE.search(boxfile)]
        allfiles.extend(boxfiles)

    # Randomize order if requested
    if shuffle:
        order = np.random.permutation(len(allfiles))
        allfiles = [allfiles[i] for i in order]
        
    # calculate number of examples in test partition and cv-partition
    num_test = int(len(allfiles) * test_part)
    num_val = int(len(allfiles) * validation_part)

    # move training, test and cv files to their places
    ds = "train"
    if not os.path.isdir(os.path.join(rootdir, ds)):
        os.makedirs(os.path.join(rootdir, ds))
    for idx in range(0, len(allfiles) - num_test - num_val):
        shutil.move(allfiles[idx], os.path.join(rootdir, ds, os.path.basename(allfiles[idx])))

    ds = "test"
    if not os.path.isdir(os.path.join(rootdir, ds)):
        os.makedirs(os.path.join(rootdir, ds))
    for idx in range(len(allfiles) - num_test - num_val, len(allfiles) - num_val):

        shutil.move(allfiles[idx], os.path.join(rootdir, ds, os.path.basename(allfiles[idx])))

    ds = "cv"
    if not os.path.isdir(os.path.join(rootdir, ds)):
        os.makedirs(os.path.join(rootdir, ds))
    for idx in range(len(allfiles) - num_val, len(allfiles)):
        shutil.move(allfiles[idx], os.path.join(rootdir, ds, os.path.basename(allfiles[idx])))

    return DataSets(labelfile, rootdir, dataconfig, shuffle=shuffle)

