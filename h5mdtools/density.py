#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2019 Robert Kernke
# Copyright © 2019 Felix Höfling
#
# This file is part of HALMD.
#
# HALMD is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program.  If not, see
# <http://www.gnu.org/licenses/>.
#



import h5py
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from os.path import basename
import sys
import re
import warnings

#Functions

def get_group(h5md):
    '''Determine first particle group name'''
    try:
        group_name=list(h5md['structure'].keys())[0]
        return group_name
    except KeyError:
        print("Could not read group_name")
        print("H5MD-File is expected to be structured like:'structure/<group_name>/density_mode' ")
        return 0

def read_density_mode_data(h5md, group):   
    full        = h5md['structure/' + group + '/density_mode']
    step        = np.array(full['step'])
    time        = np.array(full['time'])
    value       = np.array(full['value'])
    wavevector  = np.array(full['wavevector'])

    # convert pairs of reals to complex numbers
    value = value[..., 0] + 1j * value[..., 1]

    return step, time, value, wavevector

def get_box_edges(f,group):
    return np.diagonal(f['particles/'+group+'/box/edges'][:]) 

def check_width(wavevector):
    k_max = np.max(np.sqrt(np.sum( wavevector**2 , axis=1)))
    return np.pi / k_max

def reduce_data(wavevector,box_edges):
    '''Function only has an effect, if halmd.observables.utility.wavevector is called with filter {...}'''
    a=np.sum(wavevector**2,axis=0)
    reduced_wavevector=[]
    newbox=[]
    coordinate=[0,1,2]
    newcoord=[]
    for i in range(len(a)):
        if a[i]!=0:
            reduced_wavevector.append(wavevector[:,i])
            newbox.append(box_edges[i])
            newcoord.append(coordinate[i])
    return np.array(reduced_wavevector).T,np.array(newbox),newcoord

def compute_density_map(wavevector,value,sample_of_interest,box_edges,width):
    w=[]
    addlength=[]
    dlength=[]
    mesh=[]
    volume=np.prod(box_edges)
    wavevector,box_edges,coord=reduce_data(wavevector,box_edges)
    # Gaussian filter, width = 0 disables the filter (gaussian = 1)
    gaussian = np.exp(-0.5 * width**2 * norm(wavevector,axis=1)**2)

    for i in range(len(box_edges)):
        w.append(np.array(np.round(wavevector[:,i]*box_edges[i]/(2*np.pi)),dtype=int))
        assert np.min(w[i])==-np.max(w[i]) , "Density-modes need to be on a symmetric grid around 0, i.e. k_max = -k_min"
        dlength.append(2*np.max(w[i])+1)
        mesh.append(box_edges[i]*np.linspace(np.min(w[i]),np.max(w[i]),dlength[i])/np.double(dlength[i]))

    dens_modes=np.zeros(dlength[::-1],dtype=complex)
    dens_modes[tuple(w)[::-1]]=value[sample_of_interest]*gaussian
    density=np.real(np.fft.fftshift(np.fft.ifftn(dens_modes)))
    if len(density.shape)==3:##Maybe ifftn changes axis between 3 and 2 dimensions
        density=density.T
    return mesh, density*np.prod(np.shape(density))/np.double(volume),coord

def compute_density_profile(wavevector,value,sample_of_interest,box_edges,width):
    mesh=[]
    density=[]
    volume=np.prod(box_edges)
    notused,notused2,coord=reduce_data(wavevector,box_edges)
    # Gaussian filter, width = 0 disables the filter (gaussian = 1)
    gaussian = np.exp(-0.5 * width**2 * norm(wavevector,axis=1)**2)

    su=np.sum(wavevector**2,axis=1)
    for i in coord:
        idx = np.where(su == wavevector[:,i]**2)
        w=np.array(np.round(wavevector[idx,i]*box_edges[i]/(2*np.pi)),dtype=int)
        assert np.min(w)==-np.max(w) , "Density-modes need to be on a symmetric grid around 0, i.e. k_max = -k_min"
        dlength=2*np.max(w)+1
        mesh.append(box_edges[i]*np.linspace(np.min(w),np.max(w),dlength)/np.double(dlength))
        dens_modes=np.zeros(dlength,dtype=complex)            
        dens_modes[w]=value[sample_of_interest][idx]*gaussian[idx]
        densit=np.real(np.fft.fftshift(np.fft.ifftn(dens_modes)))
        density.append(densit*len(densit)/np.double(volume))
    return mesh,density,coord


def plot_3d(mesh,density):
    x,y,z=mesh

    fig, (ax1, ax2,ax3) = plt.subplots(ncols=3,figsize=(12,4.5))
    levels = np.arange(0, 0.75, 0.01)  # set color bar limits

    zg,yg=np.meshgrid(z,y)
    cset1=ax1.contourf(yg,zg,np.mean(density,axis=0), levels=levels)
    ax1.set_xlabel(r'y [$\sigma$]')
    ax1.set_ylabel(r'z [$\sigma$]')

    zg,xg=np.meshgrid(z,x)
    cset2=ax2.contourf(xg,zg,np.mean(density,axis=1), levels=levels)
    ax2.set_xlabel(r'x [$\sigma$]')
    ax2.set_ylabel(r'z [$\sigma$]')

    xg,yg=np.meshgrid(x,y)
    cset3=ax3.contourf(xg,yg,np.mean(density,axis=2), levels=levels)
    ax3.set_xlabel(r'x [$\sigma$]')
    ax3.set_ylabel(r'y [$\sigma$]')

    fig.colorbar(cset1,ax=ax1)
    fig.colorbar(cset2,ax=ax2)
    fig.colorbar(cset3,ax=ax3)

    plt.suptitle('mean number densities',y=.99)
    fig.tight_layout()

    plt.show()

def plot_2d(mesh,density,coord):
    x,y=mesh
    coordinates=['x','y','z']
    xg,yg=np.meshgrid(x,y)
    fig=plt.figure()
    plt.contourf(xg,yg,density)
    plt.xlabel(coordinates[coord[0]]+r' [$\sigma$]')
    plt.ylabel(coordinates[coord[1]]+r' [$\sigma$]')
    plt.title('number density')
    plt.colorbar()
    fig.tight_layout()
    plt.show()

def profile_plot(mesh,density,coord):
    coordinates=['x','y','z']
    fig, ax = plt.subplots(ncols=len(coord),figsize=(12,4.5)) 
    plt.suptitle('mean number density profiles',y=.99)
    if len(coord)==1:
        if isinstance(density, list):
            plt.plot(mesh[0],density[0])
        else:
            plt.plot(mesh[0],density)

        plt.xlabel(coordinates[coord[0]]+r' [$\sigma$]')
        plt.ylabel(r'number density')
    else:
        for i in coord: 
            ax[i].plot(mesh[i],density[i])
            ax[i].set_xlabel(coordinates[i]+r' [$\sigma$]')
            ax[i].set_ylabel(r'number density')
    fig.tight_layout()
    plt.show()

def check_overwrite(group_name,of,group,full_dim,verbose):
    '''Check if Data already exists'''
    if verbose:
        if group_name in list(of['structure/'+group].keys()):
            print('structure/'+group+'/'+group_name+' already exists')
            # raw_input returns the empty string for "enter"
            yes = {'yes','y', ''}
            no = {'no','n'}
            print("Overwrite "+group_name+" ? [y,n] (default yes)")
            choice = input().lower()
            if choice not in yes and choice not in no :
                print('invalid response')
                return False
            elif choice in no:
                return False
            else:
                return True
        else:
            return True

    else:
        return True


def main(args):

    #Reading from File

    for i, fn in enumerate(args.input):
        try:
            with h5py.File(fn, 'r') as h5md:
                #Particle group
                group= args.group or get_group(h5md)
                if group==0:
                    return
                #Density mode data
                step, time, value, wavevector = read_density_mode_data(h5md,group)
                #Dimensions of the box
                box_edges=get_box_edges(h5md,group)

        except IOError:
            print('Failed to open H5MD file: {0}. Skipped'.format(fn))
            continue

        #Reading from User Input 

        #Gaussian smoothing  
        kmax = np.max(np.linalg.norm(wavevector, axis=-1))
        width = args.width if args.width is not None else np.pi / kmax
        if np.exp(-0.5*width**2*kmax**2)>0.05:
            warnings.warn("Warning the Gaussian filter at k_max is still bigger than 5 percent")

        # create (extended) slicing index from string,
        # select a single sample if a single integer is given
        idx = [int(x) for x in re.split(':', args.sample)]
        idx = slice(*idx) if len(idx) > 1 else [idx[0]]
        samplelist=np.arange(len(step))[idx]

        #Check profile or map and average or timeseries
        if args.map:
            if args.average:
                newgroup_name='mean_density_map'
            else:
                newgroup_name='density_map'
        else:
            if args.average:
                newgroup_name='mean_density_profile_'
            else:
                newgroup_name='density_profile_'


        #Computing

        if args.map:
            density=[]
            for i in samplelist:
                mesh,densities,coord=compute_density_map(wavevector,value,i,
                                           box_edges=box_edges,width=width)
                density.append(densities)
            density=np.array(density)
            if args.average:
                density=np.average(density,axis=0)

        else:
            densitylist=[]
            for i in samplelist:
                mesh,densities,coord=compute_density_profile(wavevector,value,i,
                                            box_edges=box_edges,width=width)
                densitylist.append(densities)
            number_of_dim=len(coord)
            density=[]
            for i in range(number_of_dim):
                density.append(np.zeros([len(samplelist),len(densitylist[0][i])]))
                for j in range(len(samplelist)):
                    density[i][j,:]=densitylist[j][i]

            if args.average:
                density_average=[]
                for i in range(number_of_dim):
                    density_average.append(np.average(density[i],axis=0))
                density=density_average

        coordinates=['x','y','z']
        if args.axis is None or args.map:
            pass
        else:
            ncoord=[]
            ncoord.append(coord[args.axis])
            coord=ncoord

        #Verbose User Information        

        if args.verbose:
            print("Edge lengths of simulation box:", box_edges)
            print("Obtained density modes for {0} wavevectors, k_max = {1:.3g}".format(wavevector.shape[0], kmax))
            print("Resulting in a spatial resolution with a smallest distance of {0:.3g}".format(np.pi/kmax))
            print("Width of reciprocal Gaussian filter: {0:.3g}".format(width))
            print("Corresponding to a standard deviation of the Gaussian in real space of {0:.3g}".format(2*np.pi*width))
            if not args.map:
                print("Computing density profile along {0}-axis".format(coordinates[coord[0]]))
            if not args.dry_run:
                print("")
                print('density profile data will be written to')
                if args.map:
                    print('structure/'+group+'/'+newgroup_name)
                    print("")
                else:
                    for i in coord:
                        print('structure/'+group+'/'+newgroup_name+coordinates[i])
                    #print ""   
                if args.average:
                    print("the data sets 'step' and 'time' contain the information over which steps and times the average is applied")
                print("")

            if len(box_edges)!=len(coord):
                print("from "+ str(len(box_edges))+"-dimensional input only data of "+str(len(coord))+"-dimension has been used")
                coordstring=" "
                for i in range(len(coord)):
                    coordstring += coordinates[coord[i]] + ','
                print("the selected axis/axes is/are" + coordstring[:-1])
                print("")



        #Writing


        if not args.dry_run:
            with h5py.File(fn, 'r+') as of:
                H5out=of['structure/'+group]
                if args.map:
                    try:
                        group_write = H5out.create_group(newgroup_name)
                    except ValueError:
                        if check_overwrite(newgroup_name,of,group,args.map,args.verbose):
                            del of['structure/'+group+'/'+newgroup_name]
                            group_write=H5out.create_group(newgroup_name)
                        else:
                            break
                    ds1 = group_write.create_dataset('step', data=np.array(step[samplelist]))
                    ds2 = group_write.create_dataset('time', data=np.array(time[samplelist]))
                    ds3 = group_write.create_dataset('value', data=density)
                    ds4 = group_write.create_dataset('position_grid', data=np.array(np.meshgrid(*mesh)))

                else:
                    for i in range(len(coord)):
                        try:
                            group_write = H5out.create_group(newgroup_name+coordinates[coord[i]])
                        except ValueError:
                            if check_overwrite(newgroup_name+coordinates[coord[i]],of,group,args.map,args.verbose):
                                del of['structure/'+group+'/'+newgroup_name+coordinates[coord[i]]]
                                group_write=H5out.create_group(newgroup_name+coordinates[coord[i]])
                            else:
                                break
                        group_write.create_dataset('step', data=np.array(step[samplelist]))
                        group_write.create_dataset('time', data=np.array(time[samplelist]))
                        group_write.create_dataset('value', data=np.array(density[i]))
                        group_write.create_dataset('position', data=np.array(mesh[i]))

        #Plotting

        if args.plot and len(args.input)<2: 
            if args.map:
                if not args.average:
                    if len(coord)==3:
                        plot_3d(mesh,density[-1])
                    elif len(coord)==2:
                        plot_2d(mesh,density[-1],coord)
                    else: 
                        profile_plot(mesh,density[-1],coord)
                else:
                    if len(coord)==3:
                        plot_3d(mesh,density)
                    elif len(coord)==2:
                        plot_2d(mesh,density,coord)
                    else:
                        profile_plot(mesh,density,coord)
            else:
                if not args.average:
                    densityplot=[]
                    for i in range(number_of_dim):
                        densityplot.append(density[i][-1])
                    profile_plot(mesh,densityplot,coord)
                else:
                    profile_plot(mesh,density,coord) 

        #Scatter-Plot-test
        if args.scatter and len(args.input)<2:
            if -1 or len(step) in samplelist:
                sc=1
            else: #args.sample==0:
                sc=0
            f = h5py.File(fn, 'r')
            pos=f['particles/'+group+'/position/value'][:]
            f.close()
            if len(box_edges)==3:
                from mpl_toolkits.mplot3d import axes3d
                fig = plt.figure(figsize=[10,4])
                ax = fig.add_subplot(111, projection='3d')

                ax.scatter(np.mod(pos[sc,::2,2]+box_edges[2]/2.,box_edges[2]),
                           np.mod(pos[sc,::2,1]+box_edges[1]/2.,box_edges[1]),
                           np.mod(pos[sc,::2,0]+box_edges[0]/2.,box_edges[0]),
                           alpha=.1)
                ax.set_xlabel(r'z [$\sigma$]')
                ax.set_ylabel(r'y [$\sigma$]')
                ax.set_zlabel(r'x [$\sigma$]')
                ax.view_init(azim=110,elev=25)
                plt.show()
            if len(box_edges)==2:

                plt.scatter(np.mod(pos[sc,:,0]+box_edges[0]/2.,box_edges[0]),
                            np.mod(pos[sc,:,1]+box_edges[1]/2.,box_edges[1]),marker='.')

                plt.ylabel(r'y [$\sigma$]')
                plt.xlabel(r'x [$\sigma$]')
                plt.show()


def add_parser(subparsers):
    parser = subparsers.add_parser('density', help='compute 1D-density profiles from density modes')
    parser.add_argument('input', metavar='INPUT', nargs='+',
                        help='H5MD files with structure/<group>/density_mode data, mulitple files disable plotting')
    parser.add_argument('--map', action='store_true',default=False,help='computes the full dimensional density map')
    parser.add_argument('-n', '--dry-run', action='store_true',
                        help='quick trial run or plotting without writing to H5-file')
    parser.add_argument('-v', '--verbose', action='store_true', help='print detailed information')
    parser.add_argument('--axis', nargs='?',type=int,
                        help='axis for density profile, integer like {0 , 1 , 2 , -1}')
    parser.add_argument('--sample', default='-1', type=str,
                        help='(slicing) index of density mode sample(s) [integers seperated by :]')
    parser.add_argument('--average', action='store_true',default=False,
                        help='time average of the density over all selected samples with --sample , or all samples if --sample is not given')
    parser.add_argument('--group', type=str, help='particle group [Default: use first group]')
    parser.add_argument('--width', type=float, help='width of gaussian filter (0: disabled)[Default: 2*pi/(k_max/3)]')
    parser.add_argument('--plot', default=False, action='store_true', help='number density plot of last given sample (or average)')
    parser.add_argument('--scatter', default=False,action='store_true',
                        help='additional scatter plot for testing (requires particle/<group_name>/positon data)')

def parse_args():
    import argparse

    # define and parse command line arguments
    parser = argparse.ArgumentParser(description='compute 1D-density profiles from density modes')
    parser.add_argument('input', metavar='INPUT', nargs='+',
                        help='H5MD files with structure/<group>/density_mode data, multiple files disable plotting')
    parser.add_argument('--map', action='store_true',default=False,help='computes the full dimensional density map')
    parser.add_argument('-n', '--dry-run', action='store_true',
                        help='quick trial run or plotting without writing to H5-file')
    parser.add_argument('-v', '--verbose', action='store_true', help='print detailed information')
    parser.add_argument('--axis', nargs='?',type=int,
                        help='axis for density profile, integer like {0 , 1 , 2 , -1}')
    parser.add_argument('--sample', default='-1', type=str,
                        help='(slicing) index of density mode sample(s) [integers seperated by :]')
    parser.add_argument('--average', action='store_true',default=False,
                        help='time average of the density over all selected samples with --sample , or all samples if --sample is not given')
    parser.add_argument('--group', type=str, help='particle group [Default: use first group]')
    parser.add_argument('--width', type=float, help='width of gaussian filter (0: disabled)[Default: 2*pi/(k_max/3)]')
    parser.add_argument('--plot', default=False, action='store_true', help='number density plot of last given sample (or average)')
    parser.add_argument('--scatter', default=False, action='store_true',
                        help='additional scatter plot for testing (requires particle/<group_name>/positon data)')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)

