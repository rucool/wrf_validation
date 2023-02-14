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

def main(args):
    print_all = not args.alerts
    gfs_dir = args.gfs_dir
    h1 = args.h1
    t_int = args.t_int

    gfs_files = sorted(glob.glob(os.path.join(gfs_dir,'gfs.t00z*')))
    n_expected = h1/t_int + 1

    if not print_all and len(gfs_files) != n_expected:
        print(f'ALERT: {len(gfs_files)} GFS input files available, expected {n_expected}')
    
    if print_all:
        print("GFS files available for today's run:")
        print(f'    - {len(gfs_files)} GFS input files available, expected {n_expected}')
    
    small_files=0
    for f in gfs_files:
        fsize = os.path.getsize(f)
        if fsize < 1000000:
            small_files+=1
        if print_all:
            print(f'    - {f}: {fsize/1000000}MB')
    
    if not print_all and small_files>0:
        print(f'ALERT: {small_files}/{len(gfs_files)} GFS input files are small, possibly corrupt')
    

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('-x', '--alerts_only',
                            dest='alerts',
                            default=False,
                            type=bool,
                            choices=[True, False],
                            help='True to print alerts only (default False prints all info)')
    
    arg_parser.add_argument('-d', '--gfs_dir',
                            dest='gfs_dir',
                            default='/home/coolgroup/ru-wrf/real-time/gfs-input',
                            type=str,
                            help='directory with GFS files')
    
    arg_parser.add_argument('-h1', '--end_hour',
                            dest='h1',
                            default=120,
                            type=int,
                            help='last hour of GFS pulled')
    
    arg_parser.add_argument('-i', '--hourly_interval',
                            dest='t_int',
                            default=6,
                            type=int,
                            help='gap between GFS input files (hours)')
    
    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))
