#!/usr/bin/env python3


import argparse
import logging

import os

from cavitylearn import data

import cavitylearn.evaluate
import cavitylearn.data

# =============================== set up logging ==============================

LOGDEFAULT = logging.INFO
logger = logging.getLogger(__name__)

try:
    import pyprind
except ImportError:
    logger.warning("Failed to import pyprind module. Can't show you a pretty progress bar :'( ")
    pyprind = None


class PyprindProgressTracker:
    def __init__(self):
        # self.current = 0
        self.bar = None

    def init(self, total):
        # self.current = 0
        self.bar = pyprind.ProgPercent(total, monitor=True)

    def update(self, current=None):
        if self.bar:
            self.bar.update(item_id=current)

    def finish(self):
        if self.bar:
            print(self.bar)


def prettyprint_confusion_metrics(confm, classes):
    print("     " + "  ".join(("{: >6s}".format(c) for c in classes)))

    for i in range(len(classes)):
        print(classes[i][:3] + ": " + "  ".join(("{: >6d}".format(int(v)) for v in confm[i,:])))


def print_metrics(metrics, dataconfig):
    for ds_name, metric in metrics.items():
        print("Dataset", ds_name, ":")
        print("Accuracy: %.2f %%" % (metric["accuracy"]*100.0))
        print("confusion_matrix:")
        prettyprint_confusion_metrics(metric["confusion_matrix"], dataconfig.classes)


if __name__ == "__main__":

    if pyprind:
        progress_tracker = PyprindProgressTracker()
    else:
        progress_tracker = None

    # ========================= Main argument parser ==========================
    parser_top = argparse.ArgumentParser(description='Catalophore neural network training')

    parser_top.add_argument('--loglevel', action="store",
                            type=str.upper, dest='log_level',
                            metavar='LOG_LEVEL',
                            choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
                            default=LOGDEFAULT,
                            help='Set log level to be LOG_LEVEL. Can be one of: DEBUG,INFO,WARNING,ERROR,CRITICAL')

    parser_top.add_argument('--batchsize', action='store',
                            type=int, dest='batchsize',
                            default=50,
                            metavar="BATCHSIZE",
                            help="Size of training batches.")

    parser_top.add_argument('--datasets', action='store',
                            type=str, dest='datasets',
                            metavar="DS",
                            help="List of datasets on which the net will be evaluated, separated by comma. If not specified, all datasets "
                                 "in DATADIR will be evaluated.")

    parser_top.add_argument(action='store',
                            type=str, dest='dataset_dir',
                            metavar="DATADIR",
                            help="Dataset directory. This directory contains all the data and metadata files required "
                                 "for training.")

    parser_top.add_argument(action='store',
                            type=argparse.FileType("rb"), dest='checkpoint_file',
                            metavar="CHECKPOINT",
                            help="Path to the checkpoint file of the trained network.")

    args = parser_top.parse_args()

    logging.basicConfig(level=args.log_level, format='%(levelname)1s:%(message)s')


    dataconfig = data.read_dataconfig(os.path.join(args.dataset_dir, "datainfo.ini"))

    metrics = cavitylearn.evaluate.calc_metrics(dataset_dir=args.dataset_dir, checkpoint_path=args.checkpoint_file.name,
                                                dataset_names=args.datasets.split(","),
                                                progress_tracker=progress_tracker)

    print_metrics(metrics, dataconfig)


    if progress_tracker:
        progress_tracker.finish()
