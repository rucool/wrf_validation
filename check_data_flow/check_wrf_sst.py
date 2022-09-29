#!/usr/bin/env python

"""
Author: Laura Nazzaro 9/19/2022
Last modified: Laura Nazzaro 9/19/2022
Return info on WRF input SST
"""

import argparse
import sys
import os
import glob
import numpy as np
import pandas as pd
import xarray as xr

def main(args):
    sst_file=args.sst_file
    goes_dir=args.goes_dir
    goes_subdir=args.goes_subdir
    sst_dt=args.sst_dt
    
    if sst_file=='default':
        sst_file=os.path.join('/home/coolgroup/ru-wrf/real-time/sst-input/',pd.to_datetime(sst_dt).strftime('%Y%m%d'),'SST_raw_yesterday.nc')
    
    print('WRF SST Input Info:')
    print('    - SST file: '+sst_file)

    # does file exist
    if not os.path.exists(sst_file):
        print('    ******RED FLAG RED FLAG RED FLAG******')
        print('    - SST file '+sst_file+' not found.')
        exit()

    sst=xr.open_dataset(sst_file)
    if args.goes_subdir:
        goes_dir=os.path.join(goes_dir,pd.to_datetime(sst.time.values[0]).strftime('%Y'))

    goes_file=glob.glob(goes_dir+'/*'+pd.to_datetime(sst.time.values[0]).strftime('%Y%m%dT%H%M')+'.nc')
    if len(goes_file)==0:
        print('    ******RED FLAG RED FLAG RED FLAG******')
        print('    - Original GOES file for time '+pd.to_datetime(sst.time.values[0]).strftime('%Y%m%dT%H%M')+' not found in '+goes_dir)
    goes_file=goes_file[0]
    goes=xr.open_dataset(goes_file)

    # links to where the sst is coming from
    print('    - '+': '.join(sst['sst'].comment.split(':')))

    # number of images from each sst source, and time ranges
    datatypes=[]
    times=[]
    for img in sst['included_images'].values[0].split(' '):
        x=img.split('_')
        if len(x)<2:
            continue
        dt=' '.join(x[:-1])
        t=x[-1]
        datatypes.append(dt)
        times.append(t)

    for dt in np.unique(datatypes):
        tsub=[]
        for t in range(len(times)):
            if datatypes[t]==dt:
                tsub.append(times[t])
        print('    - '+dt+': '+str(len(tsub))+' images from '+min(tsub)+' to '+max(tsub))

    # goes coverage; coverage with rtg
    if len(goes_file)>0:
        goesnan=np.sum(np.isnan(goes['sst'].values))/np.shape(goes['sst'].values.flatten())[0]
        if goesnan==1:
            print('    ******RED FLAG RED FLAG RED FLAG******')
            print('    - No GOES data included!')
        else:
            mint=np.nanmin(goes['sst'].values)
            maxt=np.nanmax(goes['sst'].values)
            print('    - min GOES temp: '+f'{mint:.2f}'+'C; max GOES temp: '+f'{maxt:.2f}'+'C')
        print('    - '+f'{goesnan*100:.2f}'+'% of GOES image is missing (land is approximately 51.5% in 3km domain)')
        goes.close()

    rtgnan=np.sum(np.isnan(sst['sst'].values))/np.shape(sst['sst'].values.flatten())[0]
    if rtgnan>0:
        print('    ******RED FLAG RED FLAG RED FLAG******')
        print('    - WRF SST composite contains missing data')
    mint=np.nanmin(sst['sst'].values)
    maxt=np.nanmax(sst['sst'].values)
    print('    - min WRF SST temp: '+f'{mint:.2f}'+'C; max WRF SST temp: '+f'{maxt:.2f}'+'C')
    print('    - '+f'{rtgnan*100:.2f}'+'% of WRF SST image is missing')
    sst.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('-f', '--sst_file',
                            dest='sst_file',
                            default='default',
                            type=str,
                            #choices=['3km', '9km'],
                            help='RU-WRF SST file (file name or default=/home/coolgroup/ru-wrf/real-time/sst-input/yyyymmdd/SST_raw_yesterday.nc)')
    arg_parser.add_argument('-g', '--goes_dir',
                            dest='goes_dir',
                            default='/home/coolgroup/bpu/wrf/data/goes_composites/composites',
                            type=str,
                            #choices=['3km', '9km'],
                            help='GOES composite directory (default: /home/coolgroup/bpu/wrf/data/goes_composites/composites')
    arg_parser.add_argument('-s', '--subdirectory',
                            dest='goes_subdir',
                            default=True,
                            type=bool,
                            #choices=['3km', '9km'],
                            help='annual subdirectories for GOES composites (default: True)')
    arg_parser.add_argument('-d', '--date',
                            dest='sst_dt',
                            default='today',
                            type=str,
                            #choices=['3km', '9km'],
                            help='date to process (yyyymmdd) (default: today; only used in conjunction with default sst_file)')

    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))
