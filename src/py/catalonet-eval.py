#!/usr/bin/env python3


import argparse
import logging

LOGDEFAULT = logging.INFO

# ========================= Main argument parser ==========================
parser_top = argparse.ArgumentParser(description='Catalophore neural network evaluation')

parser_top.add_argument('--loglevel', action="store",
                        type=str.upper, dest='log_level',
                        metavar='LOG_LEVEL',
                        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
                        default=LOGDEFAULT,
                        help='Set log level to be LOG_LEVEL. Can be one of: DEBUG,INFO,WARNING,ERROR,CRITICAL')

parser_top.add_argument('-j', '--jobs', action="store",
                        type=int, dest='num_threads',
                        metavar='NUM_THREADS',
                        default=None,
                        help='Use NUM_THREADS processors simultanously. Default is to use all processors.')

subparsers = parser_top.add_subparsers(title='Actions', description='Evaluation actions',
                                       dest='main_action')

parser_metrics = subparsers.add_parser('metrics',
                                       help="Calculate metrics for a trained neural net checkpoint once")

parser_metrics.add_argument('--batchsize', action='store',
                            type=int, dest='batchsize',
                            default=50,
                            metavar="BATCHSIZE",
                            help="Size of training batches.")

parser_metrics.add_argument('--datasets', action='store',
                            type=str, dest='datasets',
                            metavar="DS",
                            help="List of datasets on which the net will be evaluated, separated by comma. If not "
                                 "specified, all datasets in DATADIR will be evaluated.")

parser_metrics.add_argument(action='store',
                            type=str, dest='dataset_dir',
                            metavar="DATADIR",
                            help="Dataset directory. This directory contains all the data and metadata files required "
                                 "for training.")

parser_metrics.add_argument(action='store',
                            type=argparse.FileType("rb"), dest='checkpoint_file',
                            metavar="CHECKPOINT",
                            help="Path to the checkpoint file of the trained network.")



parser_watch = subparsers.add_parser('watch',
                                     help="Continuously calculate accuracy of a neural net during training")

parser_watch.add_argument('--batchsize', action='store',
                            type=int, dest='batchsize',
                            default=50,
                            metavar="BATCHSIZE",
                            help="Size of training batches.")

parser_watch.add_argument('--datasets', action='store',
                            type=str, dest='datasets',
                            metavar="DS",
                            help="List of datasets on which the net will be evaluated, separated by comma. If not "
                                 "specified, all datasets in DATADIR will be evaluated.")

parser_watch.add_argument('--wait', action='store_true',
                          dest='wait',
                          help="Wait for the checkpoint file to come into existance")

parser_watch.add_argument('--max_time', action='store',
                          type=int, dest='max_time',
                          default=0,
                          metavar="SECONDS",
                          help="Maximum time for the script to run. 0 to run indefinitely.")

parser_watch.add_argument('--max_unchanged_time', action='store',
                          type=int, dest='max_unchanged_time',
                          default=1800,
                          metavar="SECONDS",
                          help="Maximum time to wait for changes until the script terminates. 0 to wait indefinitely.")

parser_watch.add_argument(action='store',
                          type=str, dest='dataset_dir',
                          metavar="DATADIR",
                          help="Dataset directory. This directory contains all the data and metadata files required "
                               "for training.")

parser_watch.add_argument(action='store',
                          type=str, dest='checkpoint_filename',
                          metavar="CHECKPOINT",
                          help="Path to the checkpoint file of the trained network.")

args = parser_top.parse_args()

logger = logging.getLogger(__name__)
logging.basicConfig(level=args.log_level, format='%(levelname)1s:%(message)s')

import os
import sys

import numpy as np

import cavitylearn.evaluate
import cavitylearn.data

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
            print(self.bar, file=sys.stderr)


def prettyprint_confusion_metrics(confm, classes):
    # column headers
    print("     " + "  ".join(("{: >6s}".format(c[:3]) for c in classes)))

    for i in range(len(classes)):
        print(classes[i][:3] + ": " + "  ".join(("{: >6d}".format(int(v)) for v in confm[i, :])) +
              "| {: >6d}".format(np.sum(confm[i, :])))

    # separator
    print("     " + "-" * 8 * len(classes))

    print("     " + "  ".join(("{: >6d}".format(int(v)) for v in np.sum(confm, axis=0))))


def prettyprint_labeledarray(array, labels):
    print("  ".join(("{: >5s}".format(l) for l in labels)))

    print("-" * 7 * len(labels))
    print("  ".join(("{: >5.3f}".format(v) for v in array)))


def print_metrics(metrics, dataconfig):
    for ds_name, metric in metrics.items():
        print("\n" + "#" * 80)
        print("Dataset", ds_name, ":")

        print("\nAccuracy: %.2f %%" % (metric["accuracy"] * 100.0))

        print("\nConfusion Matrix:")
        prettyprint_confusion_metrics(metric["confusion_matrix"], dataconfig.classes)

        print("\nPrecision:")
        prettyprint_labeledarray(metric["precision"], dataconfig.classes)

        print("\nRecall:")
        prettyprint_labeledarray(metric["recall"], dataconfig.classes)

        print("\nF-Score:")
        prettyprint_labeledarray(metric["f_score"], dataconfig.classes)

        print("\nG-Score:")
        prettyprint_labeledarray(metric["g_score"], dataconfig.classes)


def main_metrics(args, parser):
    if pyprind:
        progress_tracker = PyprindProgressTracker()
    else:
        progress_tracker = None

    dataconfig = cavitylearn.data.read_dataconfig(os.path.join(args.dataset_dir, "datainfo.ini"))

    if args.datasets is not None:
        datasets = args.datasets.split(",")
    else:
        datasets = None

    metrics = cavitylearn.evaluate.calc_metrics(dataset_dir=args.dataset_dir, checkpoint_path=args.checkpoint_file.name,
                                                batchsize=args.batchsize,
                                                dataset_names=datasets, num_threads=args.num_threads,
                                                progress_tracker=progress_tracker)

    print_metrics(metrics, dataconfig)

    if progress_tracker:
        progress_tracker.finish()


def main_watch(args, parser):

    if args.datasets is not None:
        datasets = args.datasets.split(",")
    else:
        datasets = None

    cavitylearn.evaluate.watch_training(dataset_dir=args.dataset_dir,
                                        checkpoint_path=args.checkpoint_filename, logdir=None,
                                        batchsize=args.batchsize, max_time=args.max_time,
                                        max_unchanged_time=args.max_unchanged_time, wait_for_checkpoint=args.wait,
                                        dataset_names=datasets, num_threads=args.num_threads)



# ========================= Script start ==========================


if not args.main_action:
    parser_top.error('No action selected')
elif args.main_action == 'metrics':
    main_metrics(args, parser_metrics)
elif args.main_action == 'watch':
    main_watch(args, parser_watch)
else:
    raise AssertionError("Unknown action {}".format(args.main_action))



