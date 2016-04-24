#!/usr/bin/env python3


import os, sys
import logging
import configparser
import concurrent.futures

import pyprind

import zipfile
import lzma

import numpy as np
from numbers import Number

# =============================== set up logging ==============================

LOGDEFAULT = logging.INFO
logger = logging.getLogger(__name__)

# =============================== set up config ===============================

THISCONF = 'cavitylearn-data'

config = configparser.ConfigParser(interpolation=None)

# default config values
config[THISCONF] = {
}

for p in sys.path:
    cfg_filepath = os.path.join(p, 'config.ini')
    if os.path.exists(cfg_filepath):
        logger.debug('Found config file in: ' + cfg_filepath)
        config.read(cfg_filepath)
        break
else:
    logger.error("config.ini not found!")


class PCDFileError(Exception):
    def __init__(self, message):
        self.strerror = message

    def __str__(self):
        return repr(self.strerror)


PCD_HEADER_LENGTH = 10
DTYPE = np.float32
gridsize = [10, 10, 10]


def read_pcd(pcd_lines):
    # PCD file header format
    #
    # VERSION
    # FIELDS
    # SIZE
    # TYPE
    # COUNT
    # HEIGHT
    # WIDTH # row 7
    # VIEWPOINT
    # POINTS
    # DATA ascii

    # eat 7 lines, if it ends here, cry about it
    for linecount, line in enumerate(pcd_lines):
        if linecount == 6:
            break
    else:
        raise PCDFileError("Truncated PCD file")

    toks = line.split()

    # get the number of points in the file
    if toks[0] != "WIDTH":
        raise PCDFileError("Malformed PCD file")

    try:
        width = int(toks[1])
    except ValueError:
        raise PCDFileError("Malformed PCD file")

    # eat until line 10
    for linecount, line in enumerate(pcd_lines, start=linecount + 1):
        if linecount == 9:
            break
    else:
        raise PCDFileError("Truncated PCD file")

    # preallocate array for points
    points = np.zeros([width, 4], dtype=DTYPE)

    # read points from file until done
    for linecount, line in enumerate(pcd_lines, start=linecount + 1):
        if linecount >= width + PCD_HEADER_LENGTH:
            break

        points[linecount - PCD_HEADER_LENGTH] = np.fromstring(line, dtype=DTYPE, count=4, sep=" ")

    if linecount != width + PCD_HEADER_LENGTH - 1:
        raise PCDFileError("Truncated PCD file")

    return points


def points_to_grid(points, shape, resolution, method='ongrid'):
    if len(shape) != 3 or \
            not all((x > 0 for x in shape)) or \
            not all((np.equal(np.mod(x, 1), 0) for x in shape)):
        raise TypeError('Shape must be a triplet of positive integers')

    if not isinstance(resolution, Number) or resolution <= 0:
        raise TypeError("Resolution must be a positive number")

    # calculate new center
    center = np.average(points[:, 0:3], axis=0)

    if method == 'ongrid':
        # shift center to lie on a resolution-boundary
        center = center - np.mod(center, resolution)

    # create grid
    grid = np.zeros(shape, dtype=DTYPE)
    shapehalf = np.array(shape) / 2

    # shift points to center, and calculate indices for the grid
    grid_indices = np.array((points[:, 0:3] - center) / resolution + shapehalf, dtype=np.int)

    # keep only points within the box
    # points >= 0 and points < shape
    valid_grid_indices_idx = np.all(grid_indices >= 0, axis=1) & np.all(grid_indices < shape, axis=1)

    valid_point_values = points[valid_grid_indices_idx, -1]

    # fill points on grid
    # for i, grid_coord in enumerate(grid_indices[valid_grid_indices_idx]):
    #    grid[grid_coord] = valid_point_values[i]

    grid[
        grid_indices[valid_grid_indices_idx, 0],
        grid_indices[valid_grid_indices_idx, 1],
        grid_indices[valid_grid_indices_idx, 2]] = valid_point_values

    return grid


def pcdzip_to_gridxz(infd, outfd, properties, boxshape, boxres):
    # open input zip file
    with zipfile.ZipFile(infd, mode='r') as pcdzip:
        # get list of files in the archive
        namelist = pcdzip.namelist()

        with lzma.open(outfd, 'w') as gridxz:
            grid = np.zeros([boxshape[0], boxshape[1], boxshape[2], len(properties)], dtype=DTYPE)
            for prop_idx, prop in enumerate(properties):
                pcd_name = 'target-cavity.{}.pcd'.format(prop)

                if pcd_name not in namelist:
                    logger.warning("PCD file {} not found in cavity archive".format(pcd_name))
                    continue

                with pcdzip.open(pcd_name, 'rU') as pcd_file:
                    pcd_stream = (line.decode('utf8') for line in pcd_file)
                    grid[:, :, :, prop_idx] = points_to_grid(read_pcd(pcd_stream), shape=boxshape, resolution=boxres)

            gridxz.write(grid.tobytes())


def load_labels(uuids, db_connection):
    cur = db_connection.cursor()
    cur.execute("""SELECT ligands FROM fridge_cavities WHERE uuid IN ({ins}) ORDER BY FIELD(uuid,{ins})""".format(
        ins=', '.join(['%s'] * len(uuids))), uuids * 2)

    ligands = [row[0] for row in cur]
    ligand_array = np.chararray(len(ligands), itemsize=3)
    ligand_array[:] = ligands

    return ligand_array


def labels_to_array(label_list, possible_labels):

    labels = np.chararray((len(label_list),), itemsize=3)
    labels[:] = label_list

    label_array = np.zeros(shape=[len(label_list), len(possible_labels)], dtype=np.bool)
    for i, lab in enumerate(possible_labels):
        label_array[:, i] = labels.startswith(lab.encode())

    nonassigend_count = np.sum(label_array.sum(axis=1) == 0)
    if nonassigend_count:
        logger.warning("%d examples were not assigned to a label" % nonassigend_count)

    return label_array


def main_convertpcd(args, parser):
    if sys.stderr.isatty():
        progbar_stream = sys.stderr
    else:
        # We are not writing to a terminal! Disabling progress bar.
        progbar_stream = open(os.devnull, 'w')

    bar = pyprind.ProgPercent(len(args.infiles), monitor=True, stream=progbar_stream, update_interval=2)

    def task(infilename):
        basename = os.path.splitext(os.path.basename(infilename))[0]
        outfilename = os.path.join(args.output_dir, basename + '.xz')

        try:
            with open(infilename, 'rb') as infile, open(outfilename, 'wb') as outfile:
                pcdzip_to_gridxz(infile, outfile, args.proplist.split(','), args.shape, args.resolution)

        except FileNotFoundError as e:
            logger.warning("File `{}` not found".format(infilename))
        except Exception:
            logger.exception("Failed to process file `{}`".format(infilename))
            raise

        bar.update(item_id=basename)

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

    print(bar)


def main_labelarray(args, parser):
    import catalobase_db
    ligands = args.ligands.split(",")

    with lzma.open(args.outfile, 'w') as xzfile:
        labels = load_labels(args.uuids, catalobase_db.get_connection())
        xzfile.write(labels_to_array(labels, ligands).tobytes())


def main_labellist(args, parser):
    import catalobase_db
    ligands = args.ligands.split(",")

    labels = load_labels(args.uuids, catalobase_db.get_connection())
    for uuid, label in zip(args.uuids, labels):
        args.outfile.write("{uuid}\t{label}\n".format(uuid=uuid, label=label.decode("utf8")))


if __name__ == "__main__":
    import argparse

    # ========================= Main argument parser ==========================
    parser_top = argparse.ArgumentParser(
        description='Cavitylearn data management',
        fromfile_prefix_chars='@')

    parser_top.add_argument('--log_level', action="store",
                            type=str.upper, dest='log_level',
                            metavar='LOG_LEVEL',
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

    # =========================     labelarray argument parser ==========================
    parser_labelarray = subparsers.add_parser('labelarray',
                                              help='Load labels for the cavities from the database, and store them as a'
                                                   'xz-compressed boolean numpy array.')

    parser_labelarray.add_argument('--ligands', action='store',
                                   type=str, dest='ligands',
                                   metavar="LIGANDS",
                                   required=True,
                                   help="List of Ligands separated by commas")

    parser_labelarray.add_argument(action='store',
                                   type=argparse.FileType('wb'), dest='outfile',
                                   metavar="OUTFILE",
                                   help="Output file for xz-compressed numpy label-array")

    parser_labelarray.add_argument(action='store', nargs='+',
                                   type=str, dest='uuids',
                                   metavar="UUID",
                                   help="List of cavity UUID's")

    # =========================     labellist argument parser ==========================
    parser_labellist = subparsers.add_parser('labellist',
                                              help="Load labels for the cavities from the database, and store them as "
                                                   "list of UUID's with the labels in a tab-separated file")

    parser_labellist.add_argument('--ligands', action='store',
                                   type=str, dest='ligands',
                                   metavar="LIGANDS",
                                   required=True,
                                   help="List of Ligands separated by commas")

    parser_labellist.add_argument(action='store',
                                   type=argparse.FileType('wt'), dest='outfile',
                                   metavar="OUTFILE",
                                   help="Output file for xz-compressed numpy label-array")

    parser_labellist.add_argument(action='store', nargs='+',
                                   type=str, dest='uuids',
                                   metavar="UUID",
                                   help="List of cavity UUID's")

    args = parser_top.parse_args()

    logging.basicConfig(level=args.log_level, format='%(levelname)1s:%(message)s')

    if not args.main_action:
        parser_top.error('No action selected')
    elif args.main_action == 'convertpcd':
        main_convertpcd(args, parser_convertpcd)
    elif args.main_action == 'labelarray':
        main_labelarray(args, parser_labelarray)
    elif args.main_action == 'labellist':
        main_labellist(args, parser_labellist)
    else:
        raise AssertionError("Unknown action {}".format(args.main_action))
