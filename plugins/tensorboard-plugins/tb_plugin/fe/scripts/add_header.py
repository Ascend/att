#!/usr/bin/env python
import glob
import os
import sys

HEADER = '''/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

'''


def add_header(file):
    with open(file, 'r') as f:
        contents = f.readlines()

    # do nothing if there is already header
    if contents and contents[0].startswith('/*-'):
        return

    with open(file, 'w') as out:
        out.write(HEADER)
        out.writelines(contents)


if __name__ == '__main__':
    directory = sys.argv[1]
    if not os.path.isdir(directory):
        raise ValueError('{} is not a directory'.format(directory))

    for ts_file in glob.glob(directory + '/*.ts'):
        add_header(ts_file)
