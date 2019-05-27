import h5py
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from os.path import basename
import argparse
import sys

def get_density_mode_data(f):
    
    full        = f['structure/all/density_mode']
    step        = np.array(full['step'])
    time        = np.array(full['time'])
    value       = np.array(full['value'])
    wavevector  = np.array(full['wavevector'])

    value = value[:,:,0] + value[:,:,1]*1j

    return step, time, value, wavevector

def get_box_edges(f):
    L=np.max(f['particles/all/box/edges'][:],axis=0)
    return L

def get_sigma(wavevector):
    k_max = np.max(np.sqrt(np.sum( wavevector**2 , axis=1)))
    return 2 * np.pi / k_max

def density_from_densitymodes(wavevector,value,time_of_interest,box_edges,gauss_convolution=False,sigma=0):
    if sigma == 0 :
        sigma=get_sigma(wavevector)
    w=[]
    addlength=[]
    dlength=[]
    mesh=[]
    
    if gauss_convolution:
        gaussian = np.exp( - 0.5 * sigma**2 * norm(wavevector,axis=1)**2 )
    else:
        gaussian=1.
    
    for i in range(len(box_edges)):
        w.append(np.array(wavevector[:,i]*box_edges[i]/(2*np.pi),dtype=int)) 
        if 0 in w[i]:
            addlength.append(1)
        else:
            addlength.append(0)
        dlength.append(np.max(w[i])-np.min(w[i])+int(addlength[i]))
        mesh.append(np.linspace(np.min(w[i]),np.max(w[i]),dlength[i])/(dlength[i]-1.)*box_edges[i])

    dens_modes=np.zeros(dlength[::-1],dtype=complex)
    dens_modes[tuple(w)[::-1]]=value[time_of_interest]*gaussian
    density=np.real(np.fft.fftshift(np.fft.ifftn(dens_modes)))
    if len(density.shape)==3:##Maybe ifftn changes axis between 3 and 2 dimensions
        density=density.T
    return mesh, density*np.prod(np.shape(density))/np.double(np.prod(box_edges))


def plot3d(mesh,density):
    x,y,z=mesh

    fig, (ax1, ax2,ax3) = plt.subplots(ncols=3,figsize=(12,4.5))

    zg,yg=np.meshgrid(z,y)
    cset1=ax1.contourf(yg,zg,np.mean(density,axis=0))
    ax1.set_xlabel(r'y [$\sigma$]')
    ax1.set_ylabel(r'z [$\sigma$]')

    zg,xg=np.meshgrid(z,x)
    cset2=ax2.contourf(xg,zg,np.mean(density,axis=1))
    ax2.set_xlabel(r'x [$\sigma$]')
    ax2.set_ylabel(r'z [$\sigma$]')

    xg,yg=np.meshgrid(x,y)
    cset3=ax3.contourf(xg,yg,np.mean(density,axis=2))
    ax3.set_xlabel(r'x [$\sigma$]')
    ax3.set_ylabel(r'y [$\sigma$]')

    fig.colorbar(cset1,ax=ax1)
    fig.colorbar(cset2,ax=ax2)
    fig.colorbar(cset3,ax=ax3)

    plt.suptitle('mean number densities',y=.99)
    fig.tight_layout()

    plt.show()


def plot2d(mesh,density):
    x,y=mesh
    xg,yg=np.meshgrid(x,y)
    fig=plt.figure()
    plt.contourf(xg,yg,density)
    plt.xlabel(r'x [$\sigma$]')
    plt.ylabel(r'y [$\sigma$]')
    plt.title('number density')
    plt.colorbar()
    fig.tight_layout()
    plt.show()



def main(args):
    
    #Reading
    
    filename=args.input
    gaus=not args.no_gaussian
    time_of_interest=args.sample
    sigma=args.sigma
    for i, fn in enumerate(args.input):
        try:
            f = h5py.File(fn, 'r')
        except IOError:
            print 'Failed to open H5MD file: {0}. Skipped'.format(fn)
            continue
        step, time, value, wavevector = get_density_mode_data(f)
        box_edges=get_box_edges(f)
        sigma= get_sigma(wavevector)
        
	try:
	    test=f['structure/all/density_profile'].keys()
	    if 'all_samples' in test:
	        print('all_samples already exists')
		# raw_input returns the empty string for "enter"
		yes = {'yes','y', ''}
		no = {'no','n'}
		print("Compute again and overwrite? [y,n] (default yes)")		    
		choice = raw_input().lower()
		if choice not in yes and choice not in no :
		    print('invalid response')
		    f.close()
		    return
		elif choice in no:
		    f.close()
		    return
	except KeyError:
	    pass

        f.close()
        
        #Computing
        
        if args.all_samples:
            number_of_samples=len(step)
            densityall=[]
            for i in range(number_of_samples):
                mesh,density=density_from_densitymodes(wavevector,value,i,
                                                box_edges=box_edges,gauss_convolution=gaus,sigma=sigma)
                densityall.append(density)
            densityall=np.array(densityall)
        else:
            mesh,density=density_from_densitymodes(wavevector,value,time_of_interest,
                                                box_edges=box_edges,gauss_convolution=gaus,sigma=sigma)

        if args.verbose:
            print 'density profile data is written to'
            if args.all_samples:    
                print 'structure/all/density_profile/all_samples'
            else:
                print 'structure/all/density_profile/single_sample'
            
        #Writing
        
        if args.dry_run:
            pass        
        else:
            of = h5py.File(fn, 'r+')
            H5out=of['structure/all']
            uppergroup=H5out.require_group('density_profile')
            if args.all_samples:
                try:
                    group = uppergroup.create_group('all_samples')
                except ValueError:
		    del of['structure/all/density_profile/all_samples']
		    group=uppergroup.create_group('all_samples')
      
                shape = H5out['density_mode/time'].shape
                
                ds1 = group.create_dataset('step', data=(H5out['density_mode/step'][shape[0]-1],), maxshape=(None,))            
                ds2 = group.create_dataset('time', data=(H5out['density_mode/time'][shape[0]-1],), maxshape=(None,))
                ds3 = group.create_dataset('value', data=densityall)
                ds4 = group.create_dataset('position_grid', data=np.array(np.meshgrid(*mesh)))
                
            else:
                try:
                    group = uppergroup.create_group('single_sample')
                except ValueError:
                    del of['structure/all/density_profile/single_sample']
                    group = uppergroup.create_group('single_sample')
                
                ds1 = group.create_dataset('step', data=np.array([step[time_of_interest]]))           
                ds2 = group.create_dataset('time', data=np.array([time[time_of_interest]]))
                ds3 = group.create_dataset('value', data=density)
                ds4 = group.create_dataset('position_grid', data=np.array(np.meshgrid(*mesh)))
                           
            of.close()
            
        #Plotting
        
        if args.plot and len(args.input)<2:
            if len(box_edges)==3:
                plot3d(mesh,density)
            if len(box_edges)==2:
                plot2d(mesh,density)

        
        if args.scatter and len(args.input)<2 and args.plot==False:
            f = h5py.File(fn, 'r')
            pos=f['particles/all/position/value'][:]
            f.close()
            if len(box_edges)==3:
		from mpl_toolkits.mplot3d import axes3d
                fig = plt.figure(figsize=[10,4])
                ax = fig.add_subplot(111, projection='3d')

                ax.scatter(np.mod(pos[-1,::2,2]+box_edges[2]/2.,box_edges[2]),
                           np.mod(pos[-1,::2,1]+box_edges[1]/2.,box_edges[1]),
                           np.mod(pos[-1,::2,0]+box_edges[0]/2.,box_edges[0]),
                           alpha=.1)
                ax.set_xlabel(r'z [$\sigma$]')
                ax.set_ylabel(r'y [$\sigma$]')
                ax.set_zlabel(r'x [$\sigma$]')
                ax.view_init(azim=110,elev=25 )
                plt.show()
            if len(box_edges)==2:

                plt.scatter(np.mod(pos[0,:,0]+box_edges[0]/2.,box_edges[0]),
                            np.mod(pos[0,:,1]+box_edges[1]/2.,box_edges[1]),marker='.')

                plt.ylabel(r'y [$\sigma$]')
                plt.xlabel(r'x [$\sigma$]')
                plt.show()        

                

def add_parser(subparsers):
    parser = subparsers.add_parser('density', help='compute density-profile from density-modes')
    parser.add_argument('input', metavar='INPUT', nargs='+', 
                        help='H5MD files with structure/all/density_mode data, mulitple files disable plotting')
    parser.add_argument('-n', '--dry-run', action='store_true', 
                        help='quick trial run or plotting without writing to H5-file')
    parser.add_argument('-v', '--verbose', action='store_true', help='print detailed information')
    parser.add_argument('--all-samples', action='store_true',default=False, 
                        help='computing density profiles for all timesteps')

    parser.add_argument('--sample', default=-1, type=int, 
                        help='index of phase space sample / selection used for plot')
    parser.add_argument('--no-gaussian', default=False, action='store_true', help='no smoothing, makes sigma irrelevant')
    parser.add_argument('--sigma', default=0, type=float, help='parameter of gaussian smoothing')
    parser.add_argument('--plot', default=False, action='store_true', help='number density plot')
    parser.add_argument('--scatter', default=False,action='store_true', 
                        help='additional scatter plot for testing / disables plot') #requires particle/all/positon data')
