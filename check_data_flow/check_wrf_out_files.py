#!/usr/bin/env python

"""
Author: Laura Nazzaro 11/2/2022
Last modified: Laura Nazzaro 11/2/2022
Return info on GFS files pulled in real-time
"""

import argparse
import sys
import os
import glob
import pandas as pd

def main(args):
    print_all = not args.alerts
    wrf_dir = args.wrf_dir
    model_ver = args.model_ver

    if wrf_dir=='/home/coolgroup/ru-wrf/real-time/v4.1_parallel/wrfout/today':
        wrf_dir = os.path.join('/home/coolgroup/ru-wrf/real-time/v4.1_parallel/wrfout',pd.Timestamp.now().strftime('%Y%m%d'))

    if model_ver=='3km':
        n_expected = 49
        wrf_files = glob.glob(os.path.join(wrf_dir,'wrfout_d02*'))
    elif model_ver=='9km':
        n_expected = 121
        wrf_files = glob.glob(os.path.join(wrf_dir,'wrfout_d01*'))

    if not print_all and len(wrf_files) != n_expected:
        print(f'ALERT: {len(wrf_files)} WRF {model_ver} input files available, expected {n_expected}')
    
    if print_all:
        print(f"Raw WRF files for today's {model_ver} run:")
        print(f'    - {wrf_dir}')
        print(f'    - {len(wrf_files)} files available')
    
    small_files=0
    for f in wrf_files:
        fsize = os.path.getsize(f)
        if fsize < 1000000:
            small_files+=1
    
    if not print_all and small_files>0:
        print(f'ALERT: {small_files}/{len(wrf_files)} WRF {model_ver} output files are small, possibly corrupt')
    

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('-x', '--alerts_only',
                            dest='alerts',
                            default=False,
                            type=bool,
                            choices=[True, False],
                            help='True to print alerts only (default False prints all info)')
    
    arg_parser.add_argument('-d', '--wrf_out_dir',
                            dest='wrf_dir',
                            default='/home/coolgroup/ru-wrf/real-time/v4.1_parallel/wrfout/today',
                            type=str,
                            help='directory with raw WRF output files')
    
    arg_parser.add_argument('-m', '--model_ver',
                            dest='model_ver',
                            default='3km',
                            type=str,
                            choices=['3km', '9km'],
                            help='RU-WRF model version (3km or 9km)')
    
    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))
