#!/usr/bin/env python

"""
Author: Laura Nazzaro 2/13/2023
Last modified: Laura Nazzaro 2/13/2023
Check progress of 1km wind farm and control directories (raw on boardwalk and processed on thredds)
"""

import argparse
import sys
import numpy as np
import pandas as pd
from datetime import timedelta
import glob
import os
import xarray as xr

def main(args):
    wfdir = args.turbsdir
    cdir = args.ctrldir
    wftds = args.turbstds
    ctds = args.ctrltds

    wfsubdirs = [os.path.basename(x) for x in glob.glob(os.path.join(wfdir,'20*'))]
    csubdirs = [os.path.basename(x) for x in glob.glob(os.path.join(cdir,'20*'))]
    wfrtimes = pd.to_datetime(sorted(wfsubdirs))
    crtimes = pd.to_datetime(sorted(csubdirs))
    wfrdiff = (wfrtimes[1:]-wfrtimes[:-1]).days
    crdiff = (crtimes[1:]-crtimes[:-1]).days
    wfrgaps = np.where(wfrdiff>1)[0]
    crgaps = np.where(crdiff>1)[0]

    wfproc = xr.open_dataset(wftds)
    cproc = xr.open_dataset(ctds)
    wfptimes = pd.to_datetime(sorted(wfproc['time'].data))
    cptimes = pd.to_datetime(sorted(cproc['time'].data))
    wfpdiff = (wfptimes[1:]-wfptimes[:-1]).days
    cpdiff = (cptimes[1:]-cptimes[:-1]).days
    wfpgaps = np.where(wfpdiff>1)[0]
    cpgaps = np.where(cpdiff>1)[0]

    wfrtext = 'Raw wind farm output (on boardwalk) available ' + wfrtimes[0].strftime('%Y-%m-%d') + ' to '
    for i in wfrgaps:
        wfrtext += wfrtimes[i].strftime('%Y-%m-%d') + ', ' + wfrtimes[i+1].strftime('%Y-%m-%d') + ' to '
    wfrtext += wfrtimes[-1].strftime('%Y-%m-%d')
    
    crtext = 'Raw control output (on boardwalk) available ' + crtimes[0].strftime('%Y-%m-%d') + ' to '
    for i in crgaps:
        crtext += crtimes[i].strftime('%Y-%m-%d') + ', ' + crtimes[i+1].strftime('%Y-%m-%d') + ' to '
    crtext += crtimes[-1].strftime('%Y-%m-%d')
    
    wfptext = 'Processed wind farm output (via thredds) available ' + wfptimes[0].strftime('%Y-%m-%d') + ' to '
    for i in wfpgaps:
        wfptext += wfptimes[i].strftime('%Y-%m-%d') + ', ' + wfptimes[i+1].strftime('%Y-%m-%d') + ' to '
    wfptext += wfptimes[-1].strftime('%Y-%m-%d')
    
    cptext = 'Processed control output (via thredds) available ' + cptimes[0].strftime('%Y-%m-%d') + ' to '
    for i in cpgaps:
        cptext += cptimes[i].strftime('%Y-%m-%d') + ', ' + cptimes[i+1].strftime('%Y-%m-%d') + ' to '
    cptext += cptimes[-1].strftime('%Y-%m-%d')

    print('1km runs completed:')
    print(f'    - {wfrtext}')
    print(f'    - {crtext}')
    print(f'    - {wfptext}')
    print(f'    - {cptext}')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('-tr', '--windfarmraw',
                            dest='turbsdir',
                            default='/home/coolgroup/ru-wrf/real-time/v4.1_parallel/wrfout_windturbs/1km_wf2km',
                            type=str,
                            help='directory containing 1km wind farm model output')

    arg_parser.add_argument('-cr', '--controlraw',
                            dest='ctrldir',
                            default='/home/coolgroup/ru-wrf/real-time/v4.1_parallel/wrfout_windturbs/1km_ctrl',
                            type=str,
                            help='directory containing 1km wind farm model output')
    
    arg_parser.add_argument('-ttds', '--windfarmthredds',
                            dest='turbstds',
                            default='https://tds.marine.rutgers.edu/thredds/dodsC/cool/ruwrf/wrf_4_1_1km_wf2km_processed/WRF_4.1_1km_with_Wind_Farm_Processed_Dataset_Best',
                            type=str,
                            help='thredds link (opendap nc) containing processed 1km wind farm model output')

    arg_parser.add_argument('-ctds', '--controlthredds',
                            dest='ctrltds',
                            default='https://tds.marine.rutgers.edu/thredds/dodsC/cool/ruwrf/wrf_4_1_1km_ctrl_processed/WRF_4.1_1km_Control_Processed_Dataset_Best',
                            type=str,
                            help='thredds link (opendap nc) containing processed 1km wind farm model output')

    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))