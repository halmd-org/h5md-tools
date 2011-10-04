#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2010-2011  Felix Höfling
#
# compute - compute time averages of macroscopic state variables
# from the 'observables' group of an H5MD file
#

def main(args):
    from numpy import concatenate, floor, mean, reshape, sqrt, std
    from math import pi
    import h5py

    # descriptions of derived quantities
    description = {
        'c_V': 'isochoric specific heat',
        'chi_T': 'isothermal compressibility',
        'chi_S': 'adiabatic compressibility',
    }

    # abbreviations of dataset names FIXME unify with plot/msv and move to separate module
    dset_abbrev = {
        'ETOT': 'total_energy',
        'EPOT': 'potential_energy',
        'EKIN': 'kinetic energy',
        'ENHC': 'total_energy_nose_hoover_chain',
        'PRESS': 'pressure',
        'TEMP': 'temperature',
        'VCM': 'center_of_mass_velocity',
        'XVIR': 'hypervirial',
    }

    # convert abbreviations to full qualifiers
    datasets = []
    for dset in args.datasets:
        datasets.append(dset in dset_abbrev.keys() and dset_abbrev[dset] or dset)

    # optional response coefficients
    coeff = {}
    if args.ensemble == 'nve':
        'temperature' in datasets and coeff.update({'c_V': []})
        set(('temperature','pressure','hypervirial')).issubset(datasets) and coeff.update({'chi_S': []})
    elif args.ensemble == 'nvt':
        set(('temperature','potential_energy')).issubset(datasets) and coeff.update({'c_V': []})
        set(('temperature','pressure','hypervirial')).issubset(datasets) and coeff.update({'chi_T': []})

    # construct header of table
    if args.table:
        header = '# 1:Density  2:Cutoff  3:Particles '
        col = 4
        for dset in datasets:
            # use abbreviations in table header
            if dset in dset_abbrev.values():
                name = (k for k,v in dset_abbrev.items() if v == dset).next()
            else:
                name = dset
            name = name[0].upper() + name[1:].lower() # make first letter only upper case

            header = header + '{0:2d}:{1:6s} '.format(col, name)
            header = header + '{0:2d}:{1:11s} '.format(col + 1, 'err(' + name + ')')
            name = 'Δ' + name   # the unicode character has length 2
            header = header + '{0:2d}:{1:7s} '.format(col + 2, name)
            header = header + '{0:2d}:{1:12s} '.format(col + 3, 'err(' + name + ')')
            col += 4

        for name in coeff.keys():
            header = header + '{0:2d}:{1:6s} '.format(col, name)
            header = header + '{0:2d}:{1:11s} '.format(col + 1, 'err(' + name + ')')
            col += 2

        print header[:-2]

    # equilibrium or stationary property
    for i, fn in enumerate(args.input):
        try:
            f = h5py.File(fn, mode='r')
        except IOError:
            raise SystemExit('failed to open HDF5 file: %s' % fn)

        H5 = f['observables']
        H5param = f['halmd']

        msv_mean = dict()      # mean values of observable
        msv_std = dict()       # standard deviation of observable (fluctuations)
        try:
            # read parameters
            dimension = H5param['box'].attrs['dimension']
            density = H5param['box'].attrs['density']
            npart = sum(H5param['box'].attrs['particles']).astype(int)
            force_param = H5param[H5param.attrs['force']]
            if 'cutoff' in force_param.attrs.keys():
                cutoff = force_param.attrs['cutoff'][0]
            else:
                cutoff = 0

            if args.table:
                print '# {0}, skip={1}'.format(fn, args.skip)
                print '  {0:<9.4g}  {1:^8g}  {2:8d}    '.format(density, cutoff, npart),
            else:
                print 'Filename: %s' % fn
                if cutoff > 0:
                    print 'Cutoff: %g' % cutoff
                print 'Particles: %d' % npart
                print 'Density: %.4g' % density

            # iterate over datasets
            for dset in datasets:
                # open dataset ...
                values = H5[dset]['sample']
                # ... and skip first number of entries
                values = values[args.skip:]

                # compute error from different blocks
                # divide data in blocks, support tensor data as well
                shape = list(values.shape)
                if shape[0] > args.blocks:
                    # cut size of data to a multiple of args.blocks
                    a = int(args.blocks * floor(shape[0] / args.blocks))
                    values = values[:a]
                    shape[0] = args.blocks
                    shape.insert(1, -1)
                else:
                    shape[0] = 1
                    shape.insert(1, -1)
                values = reshape(values, shape)

                # compute mean and standard deviation as well as error estimates
                mean_values = [
                    mean(mean(values, axis=1), axis=0),
                    std(mean(values, axis=1), axis=0) / sqrt(args.blocks - 1)
                ]
                # compute std dev for data from all blocks
                shape = concatenate(([shape[0] * shape[1]], shape[2:]))
                std_values = [
                    std(reshape(values, shape), axis=0),        # independent of args.blocks
#                    sqrt(mean(var(values, axis=1), axis=0)),   # this yields a different result
#                    mean(std(values, axis=1), axis=0),         # this one too
                    std(std(values, axis=1), axis=0) / sqrt(args.blocks - 1)
                ]

                if args.table:
                    print '{0:<12.6g} {1:<10.3g} '.format(*mean_values),
                    print '{0:<12.6g} {1:<10.3g} '.format(*std_values),
                else:
                    try:
                        desc = H5[dset].attrs['description']
                    except KeyError:
                        desc = dset.lower()
                    print '{0:s}: {1:.4g} ± {2:.3g}'.format(
                        desc.capitalize(), mean_values[0], mean_values[1]
                    )
                    print '  Δ{0:s}: {1:.4g} ± {2:.3g}'.format(
                        dset.capitalize(), std_values[0], std_values[1]
                    )

                # store in dictionary for later use
                msv_mean[dset] = mean_values
                msv_std[dset] = std_values

            # tail correction for truncated Lennard-Jones potential
            if False:  # disabled
                rc3i = 1 / pow(cutoff, 3)
                en_corr = (8./9) * pi * density * (rc3i * rc3i - 3.) * rc3i
                press_corr = (32./9) * pi * pow(density, 2) * (rc3i * rc3i - 1.5) * rc3i
                print '%.5g  ' % (msv_mean['potential_energy'][0] + en_corr),
                print '%.5g  ' % (msv_mean['pressure'][0] + press_corr)

            # compute response coefficients
            if args.ensemble:
                # specific heat in the microcanonical ensemble (NVE)
                if 'c_V' in coeff.keys() and args.ensemble == 'nve':
                    temp = msv_mean['temperature'][0]
                    temp_err = msv_mean['temperature'][1]
                    DeltaT = msv_std['temperature'][0]
                    DeltaT_err = msv_std['temperature'][1]
                    x =  npart * pow(DeltaT / temp, 2)
                    coeff['c_V'] = [
                        1 / (2./3 - x),
                        # error estimate
                        x * sqrt(pow(2 * DeltaT_err / DeltaT, 2) + pow(2 * temp_err / temp, 2))
                    ]

                # specific heat in the canonical ensemble (NVT)
                if 'c_V' in coeff.keys() and args.ensemble == 'nvt':
                    temp = msv_mean['temperature'][0]
                    temp_err = msv_mean['temperature'][1]
                    Delta_Epot = msv_std['potential_energy'][0]
                    Delta_Epot_err = msv_std['potential_energy'][1]
                    Cv_ = npart * pow(Delta_Epot/temp, 2)
                    coeff['c_V'] = [
                        .5 * dimension + Cv_,
                        Cv_ * sqrt(
                            pow(2 * temp_err / temp, 2)
                          + pow(2 * Delta_Epot_err / Delta_Epot, 2)
                        )     # error estimate
                    ]

                # adiabatic compressibility in the microcanonical ensemble (NVE)
                # isothermal compressibility in the canonical ensemble (NVT)
                # 
                # the formulae look the same, but the interpretation is different
                if not set(('chi_S','chi_T')).isdisjoint(coeff.keys()):
                    temp = msv_mean['temperature'][0]
                    temp_err = msv_mean['temperature'][1]
                    press = msv_mean['pressure'][0]
                    press_err = msv_mean['pressure'][1]
                    DeltaP = msv_std['pressure'][0]
                    DeltaP_err = msv_std['pressure'][1]
                    hypervirial = msv_mean['hypervirial'][0]
                    hypervirial_err = msv_mean['hypervirial'][1]
                    x = pow(DeltaP, 2) * npart / density / temp
                    # compressibility
                    chi = 1 / (2. / dimension * temp * density + press + hypervirial * density - x)
                    # error estimate
                    chi_err = pow(chi, 2) * sqrt(
                        pow((2. / dimension * density + x / temp) * temp_err, 2)
                      + pow(press_err, 2)
                      + pow(hypervirial_err * density, 2)
                      + pow(2 * x * DeltaP_err / DeltaP, 2)
                    )
                    if args.ensemble == 'nve':
                        coeff['chi_S'] = [chi, chi_err]
                    elif args.ensemble == 'nvt':
                        coeff['chi_T'] = [chi, chi_err]

                for name in coeff.keys():
                    if args.table:
                        print '{0:<12.6g} {1:<10.3g} '.format(*coeff[name]),
                    else:
                        desc = description[name].capitalize()
                        print '{0:s}: {1:.4g} ± {2:.3g}'.format(desc, coeff[name][0], coeff[name][1])
                    # clear data after output
                    coeff[name] = []

            # finish output line
            print

        except KeyError:
            print 'missing simulation data in file: %s' % fn
            print 'Skipped\n'

        finally:
            f.close()

def add_parser(subparsers):
    parser = subparsers.add_parser('compute', help='compute averages of macroscopic state variables')
    parser.add_argument('input', metavar='INPUT', nargs='+', help='H5MD file')
    parser.add_argument('--datasets', metavar='NAME', nargs='+', help='list of data sets in group \'/observables\'')
    parser.add_argument('--blocks', type=int, help='number of blocks for error estimate')
    parser.add_argument('--skip', type=int, help='number of data points to skip from the beginning')
    parser.add_argument('--table', action='store_true', help='output results in table format')
    parser.add_argument('--ensemble', metavar='TYPE', choices=['nvt', 'nve'], help='compute response coefficients for specified statistical ensemble')
    parser.set_defaults(
        datasets=('TEMP', 'PRESS', 'EPOT'),
        blocks=10,
        skip=0,
    )

