#!/usr/bin/env python3

import argparse
import logging

LOGDEFAULT = logging.INFO

# ========================= Main argument parser ==========================
parser_top = argparse.ArgumentParser(description='Cavity dataset statistics calculator',
                                     fromfile_prefix_chars='@')

parser_top.add_argument('--loglevel', action="store",
                        type=str.upper, dest='loglevel',
                        metavar='LOG_LEVEL',
                        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
                        default=LOGDEFAULT,
                        help='Set log level to be LOG_LEVEL. Can be one of: DEBUG,INFO,WARNING,ERROR,CRITICAL')

parser_top.add_argument('-j', '--jobs', action="store",
                        type=int, dest='num_threads',
                        metavar='NUM_THREADS',
                        default=None,
                        help='Use NUM_THREADS processors simultanously. Default is to use all processors.')

subparsers = parser_top.add_subparsers(title='Actions', description='metric that is to be calculated',
                                       dest='main_action')


# ========================= propcorrelation argument parser ==========================
parser_propcorrelation = subparsers.add_parser('propcorrelation',
                                          help='Calculate pearson correlation between properties for all cavities')

parser_propcorrelation.add_argument(action='store', nargs='+',
                               type=str, dest='infiles',
                               metavar="INPUT_FILE",
                               help="List of cavity zip archives")

parser_propcorrelation.add_argument('--properties', action='store',
                               type=str, dest='proplist',
                               metavar="PROPERTY",
                               required=True,
                               help="List of properties separated by commas")

arguments = parser_top.parse_args()


logger = logging.getLogger(__name__)
logging.basicConfig(level=arguments.loglevel, format='%(levelname)1s:%(message)s')

import sys
import os
import concurrent.futures
import numpy as np

from cavitylearn import converter

try:
    import pyprind
except ImportError:
    logger.warning("Failed to import pyprind module. Can't show you a pretty progress bar :'( ")
    pyprind = None


def prettyprint_matrix(matrix, headers):
    # column headers
    print("     " + "  ".join(("{: >6s}".format(c[:3]) for c in headers)))

    for i in range(len(headers)):
        print("{: >3s}: ".format(headers[i][:3]) + "  ".join(("{: >#7.4f}".format(v) for v in matrix[i, :])))

    # separator
    print("     " + "-" * 8 * len(headers))

    # print("     " + "  ".join(("{: >6d}".format(int(v)) for v in np.sum(matrix, axis=0))))


def main_propcorrelation(args, parser_propcorrelation):
    properties = args.proplist.split(',')
    propertyset = set(properties)

    if pyprind:
        bar = pyprind.ProgPercent(len(args.infiles), monitor=True, update_interval=2)
    else:
        bar = None

    def task(infilename):
        basename = os.path.splitext(os.path.basename(infilename))[0]

        try:
            with open(infilename, 'rb') as infile:
                pointcloud_dict = converter.load_pcdzip(infile)

        except FileNotFoundError as e:
            logger.exception(e)
            logger.warning("File `{}` not found".format(infilename))
            return None
        except Exception:
            logger.exception("Failed to process file `{}`".format(infilename))
            raise

        missing_props = propertyset.difference(pointcloud_dict.keys())
        if len(missing_props) != 0:
            logger.warning("Cavity zip file %s is missing requested properties %s", infilename, ", ".join(missing_props))
            return None

        # turn point cloud dictionary into point matrix
        pointvals = np.zeros([len(properties), len(pointcloud_dict[properties[0]])], dtype=np.float32)
        for i, prop in enumerate(properties):
            # take fourth column of point cloud matrix (the value of the point), and store them in the property row
            pointvals[i, :] = pointcloud_dict[prop][:, 3]

        corrm = np.corrcoef(pointvals)

        if bar:
            bar.update(item_id=basename)
        else:
            print(".", end="", flush=True)

        return corrm

    corrm_counter = 0
    corrm_total = np.zeros([len(properties), len(properties)], dtype=np.float32)


    # run in a loop when num_threads is 1
    if args.num_threads is None or args.num_threads == 0 or args.num_threads == 1:

        for infile in args.infiles:
            cavity_corrm = task(infile)

            if cavity_corrm is None:
                continue

            corrm_total += cavity_corrm
            corrm_counter += 1

    else:
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

                if ex is not None:
                    continue

                cavity_corrm = f.result()

                if cavity_corrm is None:
                    continue

                corrm_total += cavity_corrm
                corrm_counter += 1

    corrm_total = corrm_total / corrm_counter

    prettyprint_matrix(corrm_total, properties)

    if bar:
        print("", end="", flush=True, file=sys.stderr)
        print(bar, file=sys.stderr)


if not arguments.main_action:
    parser_top.error('No action selected')
elif arguments.main_action == 'propcorrelation':
    main_propcorrelation(arguments, parser_propcorrelation)
else:
    raise AssertionError("Unknown action {}".format(arguments.main_action))
