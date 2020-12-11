#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2011  Felix Höfling
#
# cat - concatenate phase space samples from different H5MD files
#
# The param group is copied from the first file, only /param/box is adjusted accordingly.
# Currently, only a single particle species is supported.
#

from __future__ import print_function

def main(args):
    from numpy import array, concatenate, prod, where, cross, sum, sqrt 
    from os.path import basename
    import h5py

    if len(args.input) < 2:
        print('Need 2 input files at least')
        return

    try:
        # open input files
        input = []
        box_vector = []
        box_length = []
        for i, fn in enumerate(args.input):
            try:
                f = h5py.File(fn, 'r')
            except IOError:
                print('Failed to open H5MD file: {0}. Skipped'.format(fn))
                continue
            H5in = f['particles/all']

            try:
                # on first input file
                if i == 0:
                    dimension = f['particles/all/box'].attrs['dimension']   
                    
                else:
                    # check that dimension and box size are compatible
                    if f['particles/all/box'].attrs['dimension']!= dimension:
                        raise SystemExit('Space dimension of input files must match')

                    idx=[]
                    for j in range(dimension):
                        if j == args.axis % dimension and dimension in (2,3):
                            if sum(abs(cross(f['particles/all/box/edges'][:,j],box_vector[0][:,j])))>1e-6:
                                raise SystemExit('Box edges parallel to concatenation axis must point in the same direction')
                        
                        elif sum(abs(f['particles/all/box/edges'][:,j]-box_vector[0][:,j]))>1e-6:
                            idx+= [j]
                    if len(idx) > 0 and idx[0] != args.axis % dimension:
                        raise SystemExit('Box edges perpendicular to concatenation axis must match')
                        
                input += [(f, H5in['position/value'], H5in['velocity/value'], H5in['mass/value'])]#, H5in['species/value'])]
                  
                box_vector += [f['particles/all/box/edges'][:]]
                box_length += [sqrt(sum(f['particles/all/box/edges'][:]**2,axis=0))] 
            
            except KeyError:
                f.close()
                raise SystemExit('Missing particles/all data in file: %s' % fn)

        # determine size of resulting box
        box_length_out = box_length[0].copy() # deep copy, don't store just a reference!
        box_length_out[args.axis] = sum(array(box_length)[:, args.axis])
        # output particle numbers and box extents
        if args.verbose:
            for i, (f,r,v,m) in enumerate(input):
                print('input file #{0:d}: {1:s}'.format(i+1, f.filename))
                
            #print 'Use phase space sample {0}'.format(args.sample)
            axis_name = { 0: '1st', 1: '2nd', 2: '3rd', 3: '4th', -1: 'last' }
            print('Concatenate along {0} axis'.format(axis_name[args.axis]))
            print('Introduce spacing of {0} by compression'.format(args.spacing))
            print('\n    particles   box extents')
            npart = 0
            for i, (f,r,v,m) in enumerate(input):
                print('#{0:d}: {1:8d}   {2}'.format(i+1, r.shape[1], box_length[i]))
                npart += r.shape[1]
            print('\n=>  {0:8d}   {1}'.format(npart, box_length_out))

        if args.dry_run:
            return

        # open output file
        try:
            of = h5py.File(args.output, 'w') 
        except IOError:
            raise SystemExit('Failed to write H5MD file: {0}'.format(args.output))
        H5out = of.create_group('particles/all')

        # copy group 'h5md' from first input file # FIXME these entries should be updated
        (f,r,v,m) = input[0]
        H5in = f['particles/all']
        f.copy('h5md', of)
        # copy group 'box'
        f.copy('particles/all/box', of['particles/all'])
        # create group 'time' and append time stamp of last sample
        shape = H5in['position/time'].shape # workaround reverse slicing bug
        group = H5out.require_group('position')
        
        ds = group.create_dataset('time', data=(H5in['position/time'][shape[0]-1],), maxshape=(None,))
        H5out.require_group('velocity')['time'] = ds # make hard link
        H5out.require_group('mass')['time'] = ds 
        #H5out.require_group('species')['time'] = ds 
        
        ds = group.create_dataset('step', data=(H5in['position/step'][shape[0]-1],), maxshape=(None,))
        H5out.require_group('velocity')['step'] = ds
        H5out.require_group('mass')['step'] = ds
        #H5out.require_group('species')['step'] = ds

        
        # store input file names and spacing parameter
        group = of.create_group('h5md_cat/input')
        group.attrs.create('files', array([basename(f.filename) for (f,r,v,m) in input]), dtype=h5py.new_vlen(str))
        group.attrs.create('spacing', args.spacing)

        # concatenate positions:
        # string subsystems together along specified axis by appropriate shifts
        shift = -.5 * sum(array(box_length)[:, args.axis])
        box_offset = shift + .5 * box_length[0][args.axis]

        position = ()
        for i,(f,r,v,m) in enumerate(input):
            L = box_length[i]
            r_ =((r[args.sample]+.5*L) % L)-.5*L # map positions back to a 0-centered simulation box
            r_[..., args.axis] *= 1 - args.spacing / L[args.axis] # compress to allow for spacing
            r_[..., args.axis] += shift+.5*L[args.axis] ###
            position += (r_,)
            shift += box_length[i][args.axis]
        position = concatenate(position)

        H5out.require_group('position').create_dataset(
            'value', data=(position,),
            maxshape=(None,) + position.shape, dtype=input[0][1].dtype,
        )

        # concatenate velocities
        velocity = concatenate([v[args.sample] for (f,r,v,m) in input])
        H5out.require_group('velocity').create_dataset(
            'value', data=(velocity,),
            maxshape=(None,) + velocity.shape, dtype=input[0][2].dtype,
        )

        # concatenate masses
        mass = concatenate([m[args.sample] for (f,r,v,m) in input])
        H5out.require_group('mass').create_dataset(
            'value', data=(mass,),
            maxshape=(None,) + mass.shape, dtype=input[0][3].dtype,
        )
        
        # concatenate species
        #species = concatenate([s[args.sample] for (f,r,v,m) in input])
        #H5out.require_group('species').create_dataset(
        #    'value', data=(species,),
        #    maxshape=(None,) + species.shape, dtype=input[0][4].dtype,
        #)
        
        
        # update box length, particle number, and average density
        of['particles/all/box'].attrs.modify('length', box_length_out)
        of['particles/all/box'].attrs.modify('particles', array([position.shape[0],]))
        of['particles/all/box'].attrs.modify('density', position.shape[0] / prod(box_length_out)) 

        of['particles/all/box/edges'][args.axis, args.axis] = box_length_out[args.axis]
        ds = group.create_dataset('offset', data=(box_offset,), maxshape=(None,))
        #H5out.require_group('box')['offset']=ds


        of.close()

    finally:
        # close files
        for (f,r,v,m) in input:
            f.close()

def add_parser(subparsers):
    parser = subparsers.add_parser('cat', help='concatenate H5MD phase space samples')
    parser.add_argument('input', metavar='INPUT', nargs='+', help='H5MD files with particles/all data')
    parser.add_argument('-o', '--output', required=True, help='output filename')
    parser.add_argument('-n', '--dry-run', action='store_true', help='perform a quick trial run without generating the output file')
    parser.add_argument('-v', '--verbose', action='store_true', help='print detailed information')
    parser.add_argument('--axis', default=-1, type=int, help='concatenation axis')
    parser.add_argument('--sample', default=-1, type=int, help='index of phase space sample')
    parser.add_argument('--spacing', default=0, type=float, help='spacing between concatenated samples')
