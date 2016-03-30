#!/usr/bin/env python3


import os, sys
import logging
import configparser

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
resolution = 0.375
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
    if len(shape) != 3 or not all((x > 0 for x in shape)) or not all((np.equal(np.mod(x, 1), 0) for x in shape)):
        raise TypeError('Shape must be a triplet of positive integers')

    if not isinstance(resolution, Number) or resolution <= 0:
        raise TypeError("Resolution must be a positive number")

    # calculate new center
    center = np.average(points[:, 0:3], axis=0)

    if method == 'ongrid':
        # shift center to lie on a resolution-boundary
        center = center - np.mod(center, resolution)

    # create grid
    grid = np.zeros(shape)
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
            for prop in properties:
                pcd_name = 'target-cavity.{}.pcd'.format(prop)

                if pcd_name not in namelist:
                    gridxz.write(np.zeros(shape=boxshape, dtype=DTYPE).tobytes())
                    logger.warning("PCD file {} not found in cavity archive".format(pcd_name))

                with pcdzip.open(pcd_name, 'rU') as pcd_file:
                    pcd_stream = (line.decode('utf8') for line in pcd_file)
                    grid = points_to_grid(read_pcd(pcd_stream), shape=boxshape, resolution=boxres)
                    gridxz.write(grid.tobytes())


def main_convertpcd(args, parser):

    if sys.stderr.isatty():
        progbar_stream = 2
    else:
        # We are not writing to a terminal! Disabling progress bar.
        progbar_stream = open(os.devnull, 'w')

    bar = pyprind.ProgBar(len(args.infiles), monitor=True, bar_char='=', stream=progbar_stream)

    proplist = args.proplist.split(',')

    for infile in args.infiles:
        basename = os.path.splitext(os.path.basename(infile.name))[0]
        outfilename = os.path.join(args.output_dir, basename + '.xz')

        with open(outfilename, 'wb') as outfile:
            pcdzip_to_gridxz(infile, outfile, proplist, args.shape,  args.resolution)

        bar.update()



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

    subparsers = parser_top.add_subparsers(title='Actions', description='Data actions',
                                           dest='main_action')

    # ========================= Download argument parser ==========================
    parser_convertpcd = subparsers.add_parser('convertpcd',
                                              help='Convert zip archives of PCD files into xz archives '
                                                   'with grids as numpy arrays.')

    parser_convertpcd.add_argument(action='store', nargs='+',
                                   type=argparse.FileType('rb'), dest='infiles',
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
                                   help="3 Integegers with the size of the grid in X, Y and Z directions separated by commas")

    parser_convertpcd.add_argument('--properties', action='store',
                                   type=str, dest='proplist',
                                   metavar="PROPERTY",
                                   required=True,
                                   help="List of properties separated by commas")


    args = parser_top.parse_args()

    logging.basicConfig(level=args.log_level, format='%(levelname)1s:%(message)s')

    if not args.main_action:
        parser_top.error('No action selected')
    elif args.main_action == 'convertpcd':
        main_convertpcd(args, parser_convertpcd)
    else:
        raise AssertionError("Unknown action {}".format(args.main_action))
