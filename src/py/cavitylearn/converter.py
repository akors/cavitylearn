#!/usr/bin/env python3
import os, sys
import mysql.connector
import logging
import configparser
import concurrent.futures

import zipfile
import lzma

import numpy as np
import scipy.interpolate
from numbers import Number

# =============================== set up logging ==============================

LOGDEFAULT = logging.INFO
logger = logging.getLogger(__name__)

# =============================== set up config ===============================


config = configparser.ConfigParser(interpolation=None)


def _load_config():
    # Look for the config file
    for p in sys.path:
        cfg_filepath = os.path.join(p, 'config.ini')
        if os.path.exists(cfg_filepath):
            logger.debug('Found config file in: ' + cfg_filepath)
            config.read(cfg_filepath)
            break
    else:
        logger.error("config.ini not found!")

module_db_connection = None


def _get_db_connection():
    global module_db_connection

    if module_db_connection:
        return module_db_connection

    if 'catalobase_db' not in config:
        logger.error("Did not find database configuration section `{}`".format("catalobase_db"))
        return None

    k = 'username'
    if k not in config['catalobase_db']:
        logger.error("Did not find {} in database configuration section".format(k))
        return None

    k = 'password'
    if k not in config['catalobase_db']:
        logger.error("Did not find {} in database configuration section".format(k))
        return None

    k = 'ip'
    if k not in config['catalobase_db']:
        logger.error("Did not find {} in database configuration section".format(k))
        return None

    k = 'db'
    if k not in config['catalobase_db']:
        logger.error("Did not find {} in database configuration section".format(k))
        return None

    # try:
    logger.debug('Connecting to CATALObase2 database')

    module_db_connection = mysql.connector.MySQLConnection(
        user=config['catalobase_db']['username'],
        password=config['catalobase_db']['password'],
        host=config['catalobase_db']['ip'],
        database=config['catalobase_db']['db'],
        #    connection_timeout=3
    )

    return module_db_connection


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


def embed_grid(in_grid, out_grid_shape, rotation_matrix=None):
    out_shapehalf = np.array(out_grid_shape) / 2.0
    in_shapehalf = np.array(in_grid.shape) / 2.0

    # create a list of points in the input grid
    in_coords = np.indices(in_grid.shape, dtype=int).reshape(3,-1).T

    # get values corresponding to the points
    in_gridpoint_values = in_grid[in_coords[:, 0], in_coords[:, 1], in_coords[:, 2]]

    # rotate grid points if requested
    if rotation_matrix is not None:
        coords_centered = np.matmul(in_coords - in_shapehalf, rotation_matrix.T)
    else:
        coords_centered = in_coords - in_shapehalf

    # create output grid, get grid points
    out_grid = np.zeros(out_grid_shape, dtype=DTYPE)
    out_coords = np.indices(out_grid.shape, dtype=int).reshape(3,-1).T

    out_gridvalues = scipy.interpolate.griddata(coords_centered, in_gridpoint_values, out_coords - out_shapehalf,
                                                fill_value=0.0, method="nearest")
    out_grid[out_coords[:,0], out_coords[:,1], out_coords[:,2]] = out_gridvalues

    return out_grid


def points_to_grid(points, shape, resolution):
    if len(shape) != 3 or \
            not all((x > 0 for x in shape)) or \
            not all((np.equal(np.mod(x, 1), 0) for x in shape)):
        raise TypeError('Shape must be a triplet of positive integers')

    if not isinstance(resolution, Number) or resolution <= 0:
        raise TypeError("Resolution must be a positive number")

    # calculate new center
    center = np.average(points[:, 0:3], axis=0)

    # shift center to lie on a resolution-boundary
    center = center - np.mod(center, resolution)

    # transform point coordinates into centered, scaled coordinates
    coords_centered_unit = (points[:, 0:3] - center) / resolution

    # create grid
    grid = np.zeros(shape, dtype=DTYPE)
    shapehalf = np.array(shape) / 2.0

    # shift points to center, and calculate indices for the grid
    grid_indices = np.array(coords_centered_unit + shapehalf, dtype=np.int)

    # keep only points within the box
    # points >= 0 and points < shape
    valid_grid_indices_idx = np.all(grid_indices >= 0, axis=1) & np.all(grid_indices < shape, axis=1)

    valid_point_values = points[valid_grid_indices_idx, -1]

    grid[
        grid_indices[valid_grid_indices_idx, 0],
        grid_indices[valid_grid_indices_idx, 1],
        grid_indices[valid_grid_indices_idx, 2]] = valid_point_values

    return grid


# rotations = [cavitylearn.math_funcs.rand_rotation_matrix for range(num_rotations)]


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
                    point_grid = points_to_grid(read_pcd(pcd_stream), shape=boxshape, resolution=boxres)
                    grid[:, :, :, prop_idx] = point_grid

            gridxz.write(grid.tobytes())


def load_labels(uuids, db_connection):
    cur = db_connection.cursor()
    cur.execute("""SELECT ligands FROM fridge_cavities WHERE uuid IN ({ins}) ORDER BY FIELD(uuid,{ins})""".format(
        ins=', '.join(['%s'] * len(uuids))), uuids * 2)

    ligands = [row[0] for row in cur]
    ligand_array = np.chararray(len(ligands), itemsize=3)
    ligand_array[:] = ligands

    return ligand_array


def labels_to_onehot(label_list, possible_labels):

    labels = np.chararray((len(label_list),), itemsize=3)
    labels[:] = label_list

    label_array = np.zeros(shape=[len(label_list), len(possible_labels)], dtype=np.bool)
    for i, lab in enumerate(possible_labels):
        label_array[:, i] = labels.startswith(lab.encode())

    nonassigend_count = np.sum(label_array.sum(axis=1) == 0)
    if nonassigend_count:
        logger.warning("%d examples were not assigned to a label" % nonassigend_count)

    return label_array


def labels_to_classindex(label_list, possible_labels):

    labels = np.chararray((len(label_list),), itemsize=3)
    labels[:] = label_list

    assigned_count = 0
    labelidx_array = np.zeros(len(label_list), dtype=np.uint32)
    for i, lab in enumerate(possible_labels):
        idx = labels.startswith(lab.encode())
        assigned_count += np.sum(idx != 0)

        labelidx_array[idx] = i

    if assigned_count != len(label_list):
        raise ValueError("%d examples were not assigned to a label" % (len(label_list) - assigned_count))

    return labelidx_array

global pyprind
pyprind = None


def main_convertpcd(args, parser):
    if sys.stderr.isatty():
        progbar_stream = sys.stderr
    else:
        # We are not writing to a terminal! Disabling progress bar.
        progbar_stream = open(os.devnull, 'w')

    global pyprind

    if pyprind:
        bar = pyprind.ProgPercent(len(args.infiles), monitor=True, stream=progbar_stream, update_interval=2)
    else:
        bar = None

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


def main_labelarray(args, parser):
    db_connection = _get_db_connection()
    if not db_connection:
        return
    ligands = args.ligands.split(",")

    with lzma.open(args.outfile, 'w') as xzfile:
        labels = load_labels(args.uuids, db_connection)
        xzfile.write(labels_to_onehot(labels, ligands).tobytes())


def main_labellist(args, parser):
    db_connection = _get_db_connection()
    if not db_connection:
        return

    ligands = args.ligands.split(",")

    labels = load_labels(args.uuids, db_connection)
    for uuid, label in zip(args.uuids, labels):
        args.outfile.write("{uuid}\t{label}\n".format(uuid=uuid, label=label.decode("utf8")))


if __name__ == "__main__":
    import argparse

    try:
        import pyprind
    except ImportError:
        logger.warning("Failed to import pyprind module. Can't show you a pretty progress bar :'( ")
        pyprind = None

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
