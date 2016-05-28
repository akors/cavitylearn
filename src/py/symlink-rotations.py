#!/usr/bin/env python3

import os
import sys
import re

import errno

RE_BOXXZFILE = re.compile('^(.*?)(\.r\d\d)?\.box\.xz$')
RE_ROT_BOXXZFILE = re.compile('^(.*?)(\.r\d\d)\.box\.xz$')

main_dir = sys.argv[1]
rotated_dir = sys.argv[2]

def force_symlink(file1, file2):
    try:
        os.symlink(file1, file2)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(file2)
            os.symlink(file1, file2)


uuids = set((RE_BOXXZFILE.match(f).group(1) for f in os.listdir(main_dir) if RE_BOXXZFILE.match(f)))
print("len(uuids)", len(uuids))

uuid_to_rotated_dict = {
    uuid: list()
    for uuid in uuids
}

rotated = [f for f in os.listdir(rotated_dir) if RE_ROT_BOXXZFILE.match(f)]
print("len(rotated)", len(rotated))

for r in rotated:
    uuid = RE_ROT_BOXXZFILE.match(r).group(1)
    if uuid in uuid_to_rotated_dict:
        uuid_to_rotated_dict[uuid].append(r)

for uuid in uuids:
    for f in uuid_to_rotated_dict[uuid]:
        force_symlink(f, os.path.join(main_dir, os.path.basename(f)))


