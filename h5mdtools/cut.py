#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2019  Felix Höfling
#
# cut - cut out particles from a region in space
#



def main(args):
    import h5py
    import numpy as np

    # copy input to destination file
    from shutil import copyfile
    copyfile(args.input, args.output)

    # open output file
    f = h5py.File(args.output, 'r+')

    if not "particles" in f:
        SystemExit("No particle trajectory in input file")

    # iterate over particle groups
    for p in list(f["particles"].values()):
        if not "position" in list(p.keys()) or not "box" in list(p.keys()):
            print("Skipping particle group: {0}".format(p.name))
            continue

        dimension = p["box"].attrs["dimension"]
        box_length = np.diagonal(p["box/edges"])

        # evaluate command line arguments
        centre = np.array(args.centre) if args.centre != None else np.zeros((dimension,))
        assert(len(centre) == dimension)

        # read positions of given sample and
        # fold back particle positions to box at origin
        pos = p["position/value"][args.sample]
        pos -= np.round(pos / box_length) * box_length

        # evaluate selection criteria
        idx = None
        if args.cuboid != None:
            assert(len(args.cuboid) == dimension)
            cuboid = np.array(args.cuboid)
            for i,x in enumerate(cuboid):
                if x == 0:
                    cuboid[i] = box_length[i]

            # minimal and maximal coordinates of cuboid region
            min_pos = centre - cuboid / 2
            max_pos = centre + cuboid / 2

            # find indices of matching particles,
            # store only first tuple entry
            idx, = np.where(np.all(min_pos <= pos, axis=1) & np.all(pos < max_pos, axis=1))

            if args.verbose:
                print("particle group: {0}".format(p.name))
                print("cuboid centre: {0}".format(centre))
                print("cuboid size: {0}".format(cuboid))
                print('{0:d} particles selected'.format(len(idx)))

        if idx is None:
            continue

        # apply selection to all data arrays within particle group, except for 'box'
        for a in list(p.values()):
            if not "value" in list(a.keys()):
                continue

            value = a["value"]
            del a["value"]
            a["value"] = value[:, idx]


def add_parser(subparsers):
    parser = subparsers.add_parser('cut', help='cut out a region from H5MD particle data')
    parser.add_argument('input', metavar='INPUT', help='H5MD file with particle data')
    parser.add_argument('-o', '--output', required=True, help='output filename')
    parser.add_argument('-n', '--dry-run', action='store_true', help='perform a quick trial run without generating the output file')
    parser.add_argument('-v', '--verbose', action='store_true', help='print detailed information')
    parser.add_argument('--sample', default=-1, type=int, help='index of sample in particle data')
    parser.add_argument('--centre', nargs='+', type=float, help='centre of region in space')
    parser.add_argument('--cuboid', nargs='+', type=float, help='edge lengths of cuboid aligned with Cartesian axes (0 means full box)')
#    parser.add_argument('--sphere', type=float, help='radius of sphere')

