import threading

import os
import shutil
import re
import numpy as np
import lzma
import queue


import configparser
import logging

import sys
from collections import OrderedDict

from . import converter

# =============================== set up logging ==============================

logger = logging.getLogger(__name__)


# =============================== set up config ===============================

THISCONF='cavitylearn-data'
config = configparser.ConfigParser(interpolation=None)

# default config values
config[THISCONF] = {
    "queue_maxsize": 1000,
    "queue_timeout": 1
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

BOX_SUFFIX = ".box"
RE_BOXFILE = re.compile('^(.*?)(\.r\d\d)?\.box$')

RE_BOXXZFILE = re.compile('^(.*?)(\.r\d\d)?\.box\.xz$')


def load_boxfile(f, dataconfig):
    if RE_BOXXZFILE.match(f):
        with lzma.open(f) as xzfile:
            file_array = np.frombuffer(xzfile.read(), dtype=dataconfig.dtype)

    elif RE_BOXFILE.match(f):
        with open(f, "rb") as infile:
            file_array = np.frombuffer(infile.read(), dtype=dataconfig.dtype)

    else:
        logger.error("Unknown file suffix for box file `{}`".format(f))
        return None

    return file_array.reshape([
        dataconfig.boxshape[0],
        dataconfig.boxshape[1],
        dataconfig.boxshape[2],
        dataconfig.num_props])


class DataSet:
    def __init__(self, labelfile, boxfiles, dataconfig, shuffle=True, verify=True):
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

                if RE_BOXXZFILE.match(boxfile):
                    if verify:
                        with lzma.open(boxfile):
                            pass

                    # get the name of the box: get basename, delete box suffix and look it up in the label list
                    boxfile_name = os.path.basename(boxfile)
                    boxfile_name = RE_BOXXZFILE.match(boxfile_name).group(1)

                elif RE_BOXFILE.match(boxfile):
                    if verify:
                        with open(boxfile, "rb"):
                            pass

                    # get the name of the box: get basename, delete box suffix and look it up in the label list
                    boxfile_name = os.path.basename(boxfile)
                    boxfile_name = RE_BOXFILE.match(boxfile_name).group(1)
                else:
                    logger.warning("File %d does not end in .box or .box.xz. I'm not quite sure what to do with it.")
                    continue

                if boxfile_name not in label_dict:
                    logger.warning("Box file `{}` not found in label file.".format(boxfile))
                    continue

                boxfiles_labels[boxfile] = label_dict[boxfile_name]

            except FileNotFoundError:
                logger.warning("Box file not found: {}".format(boxfile))

        self.N = len(boxfiles_labels)

        logger.debug("{:d} box files found from {:d} labels in list".format(self.N, len(label_dict)))

        self._labels = converter.labels_to_classindex(list(boxfiles_labels.values()), dataconfig.classes)
        self._boxfiles = list(boxfiles_labels.keys())

        if shuffle:
            self.shuffle(norestart=True)

        self._last_batch_index = 0

        self._queue_shutdown_flag = False
        self._boxqueue = queue.Queue(maxsize=int(config[THISCONF]['queue_maxsize']))

        self._workthread = None
        self._restart_worker()
        pass

    def _boxfile_read_worker(self):
        start_index = self._last_batch_index

        # Iterate over all files in the file list
        for i in range(start_index, len(self._boxfiles)):
            file = self._boxfiles[i]

            # load a single file
            box = load_boxfile(file, self._dataconfig)
            if box is None:
                continue

            # repeatedly try to insert the file into the result queue
            while True:
                try:
                    # try to put the result into the queue, with timeout
                    self._boxqueue.put((i, box), timeout=int(config[THISCONF]['queue_timeout']))

                    # if no exception was raised, break inner loop and continue with loading files
                    break
                except queue.Full:
                    # no problem, that was just the timeout. Continue trying to insert the result into the queue
                    pass
                finally:
                    # shut down queuing operations immediately, if shutdown_queue is set.
                    if self._queue_shutdown_flag:
                        return
        pass

    @property
    def labels(self):
        return self._labels.copy()

    @property
    def files(self):
        return list(self._boxfiles)


    def _restart_worker(self):
        # Signal that we want to quit the loading business
        self._queue_shutdown_flag = True

        # Eat all remaining boxes in the queue and drop them
        try:
            while True:
                self._boxqueue.get_nowait()
        except queue.Empty:
            pass

        # join the worker thread
        if self._workthread:
            self._workthread.join()

        # restart the loading business
        self._queue_shutdown_flag = False

        self._workthread = threading.Thread(target=self._boxfile_read_worker, daemon=True)
        self._workthread.start()

    def shuffle(self, norestart=False):
        rand_order = np.random.permutation(self.N)
        self._labels = self._labels[rand_order]
        self._boxfiles = [self._boxfiles[i] for i in rand_order]

        if not norestart:
            self._restart_worker()

    def rewind_batches(self, last_index=0, norestart=False):
        self._last_batch_index = last_index

        if not norestart:
            self._restart_worker()

    def next_batch(self, batch_size):
        next_index = self._last_batch_index + batch_size
        if next_index > self.N:
            batch_size = self.N - self._last_batch_index
            next_index = self.N

        label_slice = self._labels[self._last_batch_index:next_index]

        boxes_slice = np.zeros([batch_size,
                                self._dataconfig.boxshape[0],
                                self._dataconfig.boxshape[1],
                                self._dataconfig.boxshape[2],
                                self._dataconfig.num_props], dtype=self._dataconfig.dtype)

        for i in range(batch_size):
            file_idx, box = self._boxqueue.get()
            boxes_slice[i, :, :, :] = box
            self._boxqueue.task_done()

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
            boxfiles = [os.path.join(root, boxfile) for boxfile in files if RE_BOXXZFILE.search(boxfile) or RE_BOXFILE.search(boxfile)]

            if not len(boxfiles):
                continue

            # add files to current dataset, but only if the current root dir is not the top level box directory
            if not os.path.abspath(root) == os.path.abspath(boxdir):
                self.__datasets[os.path.basename(root)] = DataSet(labelfile, boxfiles, dataconfig, shuffle=shuffle)

        for ds in self.__datasets.values():
            # add files to root dataset
            rootfiles.extend(ds.files)

        self.__datasets[""] = DataSet(labelfile, rootfiles, dataconfig, shuffle=shuffle, verify=False)

    def __getitem__(self, item):
        return self.__datasets.__getitem__(item)

    def __contains__(self, item):
        return self.__datasets.__contains__(item)


def unpack_datasets(sourcedir, outdir, progress_tracker=None):

    for root, dirs, files in os.walk(sourcedir):
        current_outdir = os.path.join(outdir, os.path.relpath(root, sourcedir))

        if not os.path.isdir(current_outdir):
            os.makedirs(current_outdir)

        for file in files:
            # copy already uncompressed files
            if RE_BOXFILE.search(file):
                shutil.copy(os.path.join(root, file), os.path.join(current_outdir, file))
            elif RE_BOXXZFILE.search(file):
                outfilename = RE_BOXXZFILE.match(file).group(1) + BOX_SUFFIX

                with lzma.open(os.path.join(root, file)) as infile, \
                        open(os.path.join(current_outdir, outfilename), 'wb') as outfile:
                    outfile.write(infile.read())

            if progress_tracker:
                progress_tracker.update()


def split_datasets(labelfile, rootdir, dataconfig, test_part, validation_part=0.0, shuffle=True):
    if not (isinstance(validation_part, np.float) and validation_part >= 0) or \
            not (isinstance(test_part, np.float) and test_part >= 0):
        raise ValueError("validation_part and test_part must be positive floating point numbers between 0 and 1")

    if validation_part + test_part >= 1.0:
        raise ValueError("Validation and Test partitions cannot make up more than 100% of the data set")

    # Collect all box files in this directory recursively
    allfiles = list()
    for root, dirs, files in os.walk(rootdir):
        boxfiles = [os.path.join(root, boxfile) for boxfile in files
                    if RE_BOXXZFILE.search(boxfile) or RE_BOXFILE.search(boxfile)]
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

