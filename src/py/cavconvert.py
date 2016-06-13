#!/usr/bin/env python3

import os
import sys
# import lzma
import concurrent.futures

import logging
import argparse

from cavitylearn import converter

# =============================== set up logging ==============================

LOGDEFAULT = logging.INFO
logger = logging.getLogger(__name__)


try:
    import pyprind
except ImportError:
    logger.warning("Failed to import pyprind module. Can't show you a pretty progress bar :'( ")
    pyprind = None


def main_convertpcd(args, parser):
    if sys.stderr.isatty():
        progbar_stream = sys.stderr
    else:
        # We are not writing to a terminal! Disabling progress bar.
        progbar_stream = open(os.devnull, 'w')

    if pyprind:
        bar = pyprind.ProgPercent(len(args.infiles), monitor=True, stream=progbar_stream, update_interval=2)
    else:
        bar = None

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    if args.rotation_dir is None:
        rotation_dir = args.output_dir
    else:
        rotation_dir = args.rotation_dir
        if not os.path.isdir(rotation_dir):
            os.makedirs(rotation_dir)

    def task(infilename):
        basename = os.path.splitext(os.path.basename(infilename))[0]
        outfilename = os.path.join(args.output_dir, basename + '.box.xz')

        try:
            with open(infilename, 'rb') as infile, open(outfilename, 'wb') as outfile:
                converter.pcdzip_to_gridxz(infile, outfile, args.proplist.split(','), args.shape, args.resolution)

                if args.rotations > 0:
                    converter.pcdzip_to_gridxz_rotations(infile, os.path.join(rotation_dir, basename),
                                                         args.proplist.split(','), args.shape, args.resolution,
                                                         args.rotations)

        except FileNotFoundError as e:
            logger.exception(e)
            logger.warning("File `{}` not found".format(infilename))
        except Exception:
            logger.exception("Failed to process file `{}`".format(infilename))
            raise

        if bar:
            bar.update(item_id=basename)
        else:
            print(".", end="", flush=True)

    # process jobs in a thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        # flush progress bar before starting
        sys.stderr.flush()

        futures = [executor.submit(task, infile) for infile in args.infiles]

        for f in concurrent.futures.as_completed(futures):
            ex = f.exception()

            if isinstance(ex, KeyboardInterrupt):
                logger.info("Keyboard interrupt received. Shutting down workers.")
                for future in futures:
                    future.cancel()

    if bar:
        print(bar)

# def main_labelarray(arguments, parser):
#     db_connection = converter.get_db_connection()
#     if not db_connection:
#         return
#     ligands = arguments.ligands.split(",")
#
#     with lzma.open(arguments.outfile, 'w') as xzfile:
#         labels = converter.load_labels(arguments.uuids, db_connection)
#         xzfile.write(converter.labels_to_onehot(labels, ligands).tobytes())


def main_labelfile(args, parser):
    db_connection = converter.get_db_connection()
    if not db_connection:
        return

    labels = converter.load_labels(args.uuids, db_connection)
    for uuid, label in zip(args.uuids, labels):
        args.outfile.write("{uuid}\t{label}\n".format(uuid=uuid, label=label.decode("utf8")))


# ========================= Main argument parser ==========================
parser_top = argparse.ArgumentParser(
    description='Cavitylearn data management',
    fromfile_prefix_chars='@')

parser_top.add_argument('--loglevel', action="store",
                        type=str.upper, dest='loglevel',
                        metavar='LOGLEVEL',
                        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
                        default=LOGDEFAULT,
                        help='Set log level to be LOG_LEVEL. Can be one of: DEBUG,INFO,WARNING,ERROR,CRITICAL')

parser_top.add_argument('-j', '--jobs', action="store",
                        type=int, dest='num_threads',
                        metavar='NUM_THREADS',
                        default=1,
                        help='Use NUM_THREADS processors simultanously')

subparsers = parser_top.add_subparsers(title='Actions', description='Data actions',
                                       dest='main_action')

# ========================= convertpcd argument parser ==========================
parser_convertpcd = subparsers.add_parser('convertpcd',
                                          help='Convert zip archives of PCD files into xz archives '
                                               'with grids as numpy arrays.')

parser_convertpcd.add_argument(action='store', nargs='+',
                               type=str, dest='infiles',
                               metavar="INPUT_FILE",
                               help="List of cavity zip archives")

parser_convertpcd.add_argument('--output_dir', '-o', action='store',
                               type=str, dest='output_dir',
                               metavar="OUTPUT_DIR",
                               default=os.getcwd(),
                               help="Output directory")

parser_convertpcd.add_argument('--rotation_dir', action='store',
                               type=str, dest='rotation_dir',
                               metavar="ROTATION_DIR",
                               help="Output directory for rotations")

parser_convertpcd.add_argument('--resolution', action='store',
                               type=float, dest='resolution',
                               metavar="RES",
                               default=0.375,
                               help="Distance between the grid points in the same units as the PCD file")

parser_convertpcd.add_argument('--shape', action='store', nargs=3,
                               type=int, dest='shape',
                               metavar="INT",
                               required=True,
                               help="3 Integegers with the size of the grid in X, Y and Z directions "
                                    "separated by commas")

parser_convertpcd.add_argument('--properties', action='store',
                               type=str, dest='proplist',
                               metavar="PROPERTY",
                               required=True,
                               help="List of properties separated by commas")

parser_convertpcd.add_argument('--randrotations', action='store',
                               type=int, dest='rotations',
                               metavar="N",
                               help="Create N random rotations of the cavities")


# =========================     labelfile argument parser ==========================
parser_labellist = subparsers.add_parser('labelfile',
                                         help="Load labels for the cavities from the database, and store them as "
                                              "list of UUID's with the labels in a tab-separated file")

parser_labellist.add_argument(action='store', type=argparse.FileType('wt'), dest='outfile',
                              metavar="OUTFILE",
                              help="Output file for xz-compressed numpy label-array")

parser_labellist.add_argument(action='store', nargs='+', type=str, dest='uuids', metavar="UUID",
                              help="List of cavity UUID's")


arguments = parser_top.parse_args()

logging.basicConfig(level=arguments.loglevel, format='%(levelname)1s:%(message)s')

if not arguments.main_action:
    parser_top.error('No action selected')
elif arguments.main_action == 'convertpcd':
    main_convertpcd(arguments, parser_convertpcd)
elif arguments.main_action == 'labellist':
    main_labelfile(arguments, parser_labellist)
else:
    raise AssertionError("Unknown action {}".format(arguments.main_action))
