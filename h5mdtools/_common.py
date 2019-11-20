#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2010-2011  Felix Höfling
#
# compute - compute time averages of macroscopic state variables
# from the 'observables' group of an H5MD file
#

# dictionary of abbreviations for dataset names
dset_abbrev = {
    'PRESS': 'pressure',
    'TEMP': 'temperature',
    'DENS': 'density',
    'EPOT': 'potential_energy',
    'EKIN': 'kinetic_energy',
    'EINT': 'internal_energy',
    'ENHC': 'nose_hoover_chain/internal_energy',
    'ENTH': 'enthalpy',
    'VCM': 'center_of_mass_velocity',
    'VX': 'center_of_mass_velocity',
    'VY': 'center_of_mass_velocity',
    'VZ': 'center_of_mass_velocity',
    'MSD': 'mean_square_displacement',
    'MQD': 'mean_quartic_displacement',
    'VACF': 'velocity_autocorrelation',
    'ISF': 'intermediate_scattering_function',
}

