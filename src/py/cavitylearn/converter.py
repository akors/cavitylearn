#!/usr/bin/env python3

import configparser
import logging
import lzma
import sys
import zipfile

import errno
import mysql.connector
import numpy as np
import os
import re

from . import math_funcs

# =============================== set up logging ==============================

LOGDEFAULT = logging.INFO
logger = logging.getLogger(__name__)

# =============================== set up config ===============================


config = configparser.ConfigParser(interpolation=None)

module_db_connection = None


def get_db_connection():
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


RE_PCDFILE = re.compile("target-cavity\.(.+)\.pcd")


def load_pcdzip(infd, properties=None):
    pointcloud_dict = dict()

    with zipfile.ZipFile(infd, mode='r') as pcdzip:
        namelist = pcdzip.namelist()

        props_in_zip = [RE_PCDFILE.match(name).group(1) for name in namelist if RE_PCDFILE.match(name)]

        if properties is None:
            properties = props_in_zip

        for prop in properties:

            if prop not in props_in_zip:
                logger.warning("PCD file for property `{}` not found in cavity archive".format(prop))
                pointcloud_dict[prop] = np.zeros([0, 4], dtype=DTYPE)
                continue

            with pcdzip.open('target-cavity.{}.pcd'.format(prop), 'rU') as pcd_file:
                pcd_stream = (line.decode('utf8') for line in pcd_file)
                pointcloud_dict[prop] = read_pcd(pcd_stream)

    return pointcloud_dict


def pcdzip_to_gridxz_rotations(infile, outfile_basename, properties, boxshape, boxres, num_rotations):
    property_points_dict = load_pcdzip(infile, properties)

    max_extent = np.ceil(np.max(boxshape) * np.sqrt(3))

    property_grids_dict = {
        prop: math_funcs.points_to_grid(property_points_dict[prop], shape=[max_extent, max_extent, max_extent],
                                        resolution=boxres)
        for prop in properties
    }

    for i in range(num_rotations):

        outfilename = outfile_basename + ".r{:02d}.box.xz".format(i+1)

        rotation_matrix = math_funcs.rand_rotation_matrix()

        with lzma.open(outfilename, 'w') as gridxz:
            outgrid = np.zeros([boxshape[0], boxshape[1], boxshape[2], len(properties)], dtype=DTYPE)
            for prop_idx, prop in enumerate(properties):
                propgrid = embed_grid(property_grids_dict[prop], boxshape, rotation_matrix=rotation_matrix)
                outgrid[:, :, :, prop_idx] = propgrid
            gridxz.write(outgrid.tobytes())


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


def force_symlink(file1, file2):
    try:
        os.symlink(file1, file2)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(file2)
            os.symlink(file1, file2)


RE_BOXFILE = re.compile('^(.*?)(\.r\d\d)?\.box(\.xz)?$')
RE_BOXFILE_ROT = re.compile('^(.*?)(\.r\d\d)\.box(\.xz)?$')


def symlink_rotations(main_dir: str, rotated_dir: str):
    """
    Link rotation files of cavity boxes into the directory of unrotated boxes.

    This looks for cavity box files in the target directory, and records all UUID's in it. Then it looks in the
    rotated_dir for rotated cavity box files corresponding to the UUID's, and creates symbolic links to them in the
    target directory.

    :param main_dir: Directory where to create the symlinks
    :param rotated_dir: Directory with the corresponding rotations
    :return: None
    """
    uuids = set((RE_BOXFILE.match(direntry.name).group(1)
                 for direntry in os.scandir(main_dir) if direntry.is_file() and RE_BOXFILE.match(direntry.name)))

    uuid_to_rotated_dict = {
        uuid: list()
        for uuid in uuids
    }

    rotated = [direntry.name for direntry in os.scandir(rotated_dir)
               if direntry.is_file() and RE_BOXFILE_ROT.match(direntry.name)]

    rotated_relpath = os.path.relpath(rotated_dir, main_dir)

    for r in rotated:
        uuid = RE_BOXFILE_ROT.match(r).group(1)
        if uuid in uuid_to_rotated_dict:
            uuid_to_rotated_dict[uuid].append(r)

    for uuid in uuids:
        for f in uuid_to_rotated_dict[uuid]:
            force_symlink(os.path.join(rotated_relpath, f), os.path.join(main_dir, os.path.basename(f)))
