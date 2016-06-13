import sys
import os
import shutil
import io
import threading
import re
import lzma

import configparser
import logging

from collections import OrderedDict
import queue
import numpy as np


from . import converter

# =============================== set up logging ==============================

logger = logging.getLogger(__name__)


# =============================== set up config ===============================

THISCONF = 'cavitylearn-data'
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
    """Data configuration class

    :type classes: list[string]
    :type num_props: int
    :type boxshape: list[int]
    :type dtype: np.dtype
    """

    def __init__(self, classes: list, num_props: int, boxshape: list, dtype: np.dtype):
        """DataConfig object constructor.

        :param classes: List of classes in the dataset
        :param num_props: Number of properties or "colors" per box pixel.
        :param boxshape: List with 3 integers with the shape of the box.
        :param dtype: Datatype of the box pixels.
        """
        self.num_classes = len(classes)
        self.classes = list(classes)
        self.num_props = num_props
        self.boxshape = list(boxshape)
        self.dtype = dtype


DATACONFIG_SECTION = 'dataconfig'


def read_dataconfig(configfile):
    """Read dataconfig from .ini file.

    :param str configfile: File path or file object to the data configuration .ini file
    :return: A DataConfig object

    :rtype: DataConfig
    """
    conf = configparser.ConfigParser()

    if isinstance(configfile, str):
        configfile = open(configfile, "rt")

    try:
        conf.read_file(configfile)

        classes = [cl.strip() for cl in conf[DATACONFIG_SECTION]["classes"].split(',')]
        properties = [prop.strip() for prop in conf[DATACONFIG_SECTION]["proplist"].split(',')]
        shape = [int(s) for s in conf[DATACONFIG_SECTION]["shape"].split(',')]

        dtype_str = conf[DATACONFIG_SECTION]["dtype"]
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
RE_BOXFILE = re.compile(r'^(.*?)(\.r\d\d)?\.box$')

RE_BOXXZFILE = re.compile(r'^(.*?)(\.r\d\d)?\.box\.xz$')


def load_boxfile(f: str, dataconfig: DataConfig) -> np.array:
    """Load a box file.

    This reads the input file depending on its ending. If it ends in .box, the file is read as-is. If it ends in
    .box.xz, the data is first decompressed using the LZMA algorithm. The input file data is read as a numpy array and
    reshaped to match the info in the dataconfig.


    :param f: Filename of the box file. Has to end either in .box or .box.xz .
    :param dataconfig: Data configuration
    :return: data file as array, reshaped to match the data configuration
    """
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
    """Data set handle class.

    This class represents a set of input box arrays along with their labels. It is created from a list of files ending
    in .box (uncompressed) or .box.xz.

    The data files are read permanently in the background, and the resulting arrays and labels are buffered until they
    are retrieved via read_batch.
    """

    def __init__(self, labelfile: io.IOBase, boxfiles: list, dataconfig: DataConfig, shuffle=True, verify=True):
        """Create a new DataSet from a list of box files, a label file and data configuration.

         The label file is a tab separated file with two columns. The first column is the UUID of the box file
         (basename of the box file without .box or .box.xz extension), and the second column is the name of the class.

        :param labelfile: Filepath or file object of the label file.
        :param boxfiles: List of box file paths. All filenames have to end in .box or .box.xz .
        :param dataconfig: Data configuration object
        :param shuffle: If true, randomize the order upon construction
        :param verify: If true, opens each file and verifies that it is readable and an LZMA-compressed file
        (if it ends in .box.xz).
        """
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
        """Boxfile read worker.
        Sequentially reads all files currently in the files list, and pushes them into the box array queue.
        If the maximum queue size has been reached, the function blocks and waits until there is still space in the
        queue.

        To stop this worker, self._queue_shutdown_flag has to be set to true.


        This function should be started from a separate thread.

        :return: None
        """
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

    def _restart_worker(self):
        """Start or restart the boxfile read worker.

        If a boxfile rad worker is currently running, it is hut down.
        A new thread for the boxfile read worker is started.

        :return: None
        """
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
        """Shuffle the order of the dataset. Restarts the boxfile read worker unless norerstart is specified.

        :param norestart: Do not restart the boxfile read worker
        :return: None
        """
        rand_order = np.random.permutation(self.N)
        self._labels = self._labels[rand_order]
        self._boxfiles = [self._boxfiles[i] for i in rand_order]

        if not norestart:
            self._restart_worker()

    def rewind_batches(self, last_index=0, norestart=False):
        """Rewind the batch index pointer. This resets the DataSet to a fresh state.
        The next call to next_batch after invoking this functino will return the same data as the first call to
        next_batch.

        Unless norestart is specified, the boxfile read worker will be restarted.
        This function should be called when the dataset has been exhausted and new batches are still desired.

        :param last_index:
        :param norestart: Do not restart the boxfile read worker
        :return: None
        """
        self._last_batch_index = last_index

        if not norestart:
            self._restart_worker()

    def next_batch(self, batch_size: int) -> (np.array, np.array):
        """Retrieve the next batch of box arrays.



        :param batch_size: Number of labels/boxes to return at most.
        :return:
        """
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

    @property
    def labels(self) -> np.array:
        """labels property

        :return: A numpy array with the label indices, in the order in which they are returned by next_batch.
        """
        return self._labels.copy()

    @property
    def files(self) -> list:
        """files property

        :return: A list of all the boxfile paths, in the order in which they are returned by next_batch.
        """
        return list(self._boxfiles)


def load_datasets(labelfile: io.IOBase, boxdir: str, dataconfig: DataConfig, shuffle=True, verify=True):
    """Load datases from a dataset directory.

    Recursively traverses the given dataset directory, and creates a DataSet for each directory which directly contains
    .box or .box.xz files. Additionally, a DataSet is created for ALL .box or .box.xz files recursively found in
    top-level directory.

    :param labelfile: Filepath or file object of the label file.
    :param boxdir: Input directory that will be recursively searched for .box or .box.xz files.
    :param dataconfig: Data configuration object
    :param shuffle: If true, randomize the order upon construction
    :param verify: If true, opens each file and verifies that it is readable and an LZMA-compressed file
    (if it ends in .box.xz).
    :return: A dictionary with the names of the directories containing .box/.box.xz files directly and "" as keys, and
    the datasets for the respective directories and the root directory as values.
    """

    datasets = {
    }

    rootfiles = list()

    # walk the box directory. Create dataset for each directory that contains '.box.xz' files.
    for root, dirs, files in os.walk(boxdir):
        # accumulate all boxfiles
        boxfiles = [os.path.join(root, boxfile) for boxfile in files if
                    RE_BOXXZFILE.search(boxfile) or RE_BOXFILE.search(boxfile)]

        if not len(boxfiles):
            continue

        # add files to current dataset, but only if the current root dir is not the top level box directory
        if not os.path.abspath(root) == os.path.abspath(boxdir):
            datasets[os.path.basename(root)] = DataSet(labelfile, boxfiles, dataconfig, shuffle=shuffle, verify=verify)
            if isinstance(labelfile, io.IOBase):
                labelfile.seek(io.SEEK_SET)

    for ds in datasets.values():
        # add files to root dataset
        rootfiles.extend(ds.files)

    datasets[""] = DataSet(labelfile, rootfiles, dataconfig, shuffle=shuffle, verify=False)

    return datasets


def unpack_datasets(sourcedir: str, outdir: str, progress_tracker=None):
    """Uncompress compressed .box files.

    This traverses a directory recursively, unpacking each .box.xz file to a .box in the output directory with the same
    relative path.

    :param sourcedir: Source  directory containing .box.xz files
    :param outdir: Output directory for the .box files. This can be the source directory.
    :param progress_tracker: An object with an update() function, that will be called once for each file.
    :return:
    """

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


RE_BOXFILE_ROT = re.compile('^(.*?)(\.r\d\d)?\.box(\.xz)?$')


def split_datasets(rootdir: str, test_part: float, validation_part=0.0, shuffle=True):
    """Split dataset into train, test and cv partitions.
    Recursively collects .box and .box.xz files in the root directory, then distributes those files to test, train and
    cv subdirectories according to the fractions specified by test_part and validation_part.

    test_part + validation_part < 1

    :param rootdir: Root directory that contains the .box/.box.xz files
    :param test_part: Fraction of data that will be the test partition, must be between 0 and 1.
    :param validation_part: Fraction of data that will be the cv partition, must be between 0 and 1.
    :param shuffle: Shuffle original datasets.
    """

    if not (isinstance(validation_part, np.float) and validation_part >= 0) or \
            not (isinstance(test_part, np.float) and test_part >= 0):
        raise ValueError("validation_part and test_part must be positive floating point numbers between 0 and 1")

    if validation_part + test_part >= 1.0:
        raise ValueError("Validation and Test partitions cannot make up more than 100% of the data set")

    # Collect all box files in this directory recursively
    uuid_files_dict = dict()

    for root, dirs, files in os.walk(rootdir):
        for file in files:
            m = RE_BOXFILE_ROT.match(file)
            if not m:
                continue

            filepath = os.path.join(root, file)
            uuid = m.group(1)
            if uuid not in uuid_files_dict:
                uuid_files_dict[uuid] = [filepath]
            else:
                uuid_files_dict[uuid].append(filepath)

    number_of_uuids = len(uuid_files_dict)

    uuids = list(uuid_files_dict.keys())

    # Randomize order if requested, otherwise order lexicographically
    if shuffle:
        order = np.random.permutation(number_of_uuids)
        uuids = [uuids[idx] for idx in order]
    else:
        uuids.sort()

    # calculate number of examples in test partition and cv-partition
    num_test = int(number_of_uuids * test_part)
    num_val = int(number_of_uuids * validation_part)

    # move training, test and cv files to their places
    ds = "train"
    if not os.path.isdir(os.path.join(rootdir, ds)):
        os.makedirs(os.path.join(rootdir, ds))
    for idx in range(0, number_of_uuids - num_test - num_val):
        for file in uuid_files_dict[uuids[idx]]:
            shutil.move(file, os.path.join(rootdir, ds, os.path.basename(file)))

    ds = "test"
    if not os.path.isdir(os.path.join(rootdir, ds)):
        os.makedirs(os.path.join(rootdir, ds))
    for idx in range(number_of_uuids - num_test - num_val, number_of_uuids - num_val):
        for file in uuid_files_dict[uuids[idx]]:
            shutil.move(file, os.path.join(rootdir, ds, os.path.basename(file)))

    ds = "cv"
    if not os.path.isdir(os.path.join(rootdir, ds)):
        os.makedirs(os.path.join(rootdir, ds))
    for idx in range(number_of_uuids - num_val, number_of_uuids):
        for file in uuid_files_dict[uuids[idx]]:
            shutil.move(file, os.path.join(rootdir, ds, os.path.basename(file)))

    return
