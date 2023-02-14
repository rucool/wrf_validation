#!/usr/bin/env python

"""
Author: Lori Garzio on 8/24/2022
Last modified: Laura Nazzaro 11/2/2022
Check latest timestamp for the Coastal Metocean Monitoring Station
Plots here: https://rucool.marine.rutgers.edu/data/meteorological-modeling/coastal-metocean-monitoring-station/
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import glob
from datetime import timedelta


def main(args):
    print_all = not args.alerts

    if print_all:
        print('Coastal Metocean Monitoring Station data:')
        print('    - checked: /home/coolgroup/MetData/CMOMS/surface_temp/surface')
        print('    - plots: https://rucool.marine.rutgers.edu/data/meteorological-modeling/coastal-metocean-monitoring-station/')
    met_dir = '/home/coolgroup/MetData/CMOMS/surface_temp/surface'

    # find the most recent file
    last_file = sorted(glob.glob(os.path.join(met_dir, '*.csv')))[-1]
    df = pd.read_csv(last_file)

    tm = pd.to_datetime(np.array(df['time_stamp(utc)']))
    last_ts = pd.to_datetime(np.nanmax(tm)).strftime('%Y-%m-%d %H:%M')

    if not print_all and pd.Timestamp.now()-pd.to_datetime(last_ts) > timedelta(days=3):
        print('ALERT: Coastal Metocean Monitoring Station data is outdated > 3 days')

    if print_all:
        # return the most recent timestamp
        print(f'    - latest data {last_ts} UTC')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('-x', '--alerts_only',
                            dest='alerts',
                            default=False,
                            type=bool,
                            choices=[True, False],
                            help='True to print alerts only (default False prints all info)')
    
    arg_parser.add_argument('-m', '--model_ver',
                            dest='model_ver',
                            default='3km',
                            type=str,
                            choices=['3km', '9km'],
                            help='RU-WRF model version (3km or 9km)')

    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))
