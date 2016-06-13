#!/usr/bin/env python3

import argparse
import logging
import socket
from time import strftime

import cavitylearn.train


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


def make_default_runname():
    return "{}.{}".format(socket.gethostname(), strftime("%Y%m%dT%H%M%S"))


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

    parser_top.add_argument('-j', '--jobs', action="store",
                            type=int, dest='num_threads',
                            metavar='NUM_THREADS',
                            default=None,
                            help='Use NUM_THREADS processors simultanously. Default is to use all processors.')

    parser_top.add_argument(action='store',
                            type=str, dest='dataset_dir',
                            metavar="DATADIR",
                            help="Dataset directory. This directory contains all the data and metadata files required "
                                 "for training.")

    parser_top.add_argument(action='store', nargs='?',
                            type=str, dest='run_dir',
                            metavar="RUNDIR",
                            help="Run directory. This directory will contain the output of the run.")

    parser_top.add_argument('--name', action='store',
                            type=str, dest='run_name',
                            default=make_default_runname(),
                            metavar="RUN_NAME",
                            help="Training run name")

    parser_top.add_argument('--batchsize', action='store',
                            type=int, dest='batchsize',
                            default=50,
                            metavar="BATCHSIZE",
                            help="Size of training batches.")

    parser_top.add_argument('--learnrate', action='store',
                            type=float, dest='learnrate',
                            default=1e-4,
                            metavar="LEARNRATE",
                            help="Initial learning rate for optimization algorithm")

    parser_top.add_argument('--learnrate-decay', action='store',
                            type=float, dest='learnrate_decay',
                            default=0.95,
                            metavar="LEARNRATE_DECAY",
                            help="Decay rate for the learning rate")

    parser_top.add_argument('--keepprob_conv', action='store',
                            type=float, dest='keepprob_conv',
                            default=1.0,
                            metavar="KEEPPROB_CONV",
                            help="Keep probability for dropout in the conv layer")

    parser_top.add_argument('--keepprob_fc', action='store',
                            type=float, dest='keepprob_fc',
                            default=0.75,
                            metavar="KEEPPROB_FC",
                            help="Keep probability for dropout in the fc-layer")

    parser_top.add_argument('--lambda', action='store',
                            type=float, dest='l2reg_scale',
                            default=0.0,
                            metavar="LAMBDA",
                            help="L2 regularization scale parameter")

    parser_top.add_argument('--epochs', action='store',
                            type=int, dest='epochs',
                            default=1,
                            metavar="EPOCHS",
                            help="Number of times to feed the whole training set into training.")

    parser_top.add_argument('--batches', action='store',
                            type=int, dest='batches',
                            default=0,
                            metavar="BATCHES",
                            help="Stop training after at most BATCHES have been fed")

    parser_top.add_argument('--track_accuracy', action='store_true',
                            dest='track_accuracy',
                            help="Track the accuracy of the model on the test set")

    parser_top.add_argument('--continue', action='store_true',
                            dest='cont',
                            help="Pick up training from the last checkpoint, if one exists.")

    args = parser_top.parse_args()

    logging.basicConfig(level=args.log_level, format='%(levelname)1s:%(message)s')

    cavitylearn.train.run_training(args.dataset_dir, args.run_dir, args.run_name, continue_previous=args.cont,
                                   batchsize=args.batchsize, batches=args.batches, epochs=args.epochs,
                                   keep_prob_conv=args.keepprob_conv, keep_prob_hidden=args.keepprob_fc,
                                   l2reg_scale=args.l2reg_scale,
                                   learnrate=args.learnrate, learnrate_decay=args.learnrate_decay,
                                   track_test_accuracy=args.track_accuracy, num_threads=args.num_threads,
                                   progress_tracker=progress_tracker)

    if progress_tracker:
        progress_tracker.finish()
