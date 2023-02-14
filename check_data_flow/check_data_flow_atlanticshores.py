#!/usr/bin/env python

"""
Author: Lori Garzio on 8/24/2022
Last modified: Laura Nazzaro 11/2/2022
Check latest timestamp for the wind dataset from Atlantic Shores buoy data:
https://erddap.maracoos.org/erddap/search/index.html?page=1&itemsPerPage=1000&searchFor=atlantic+shores
"""

import argparse
import sys
import numpy as np
import pandas as pd
from datetime import timedelta
from erddapy import ERDDAP


def main(args):
    buoy = args.buoy
    height = args.height
    print_all = not args.alerts

    if print_all:
        print(f'Atlantic Shores {buoy} data availability:')
        print(f'    - checked: https:// erddap.maracoos.org/erddap/tabledap/AtlanticShores_{buoy}_wind.html')
    
    url_erddap = 'https://erddap.maracoos.org/erddap/'
    dataset_id = f'AtlanticShores_{buoy}_wind'

    e = ERDDAP(server=url_erddap,
               protocol='tabledap',
               response='nc'
               )

    variables = [
        'wind_speed',
        'altitude',
        'time'
    ]

    e.dataset_id = dataset_id
    e.variables = variables

    alert_outdated=[]
    alert_lowdata=[]

    for h in height:
        constraints = {
            'altitude=': h
        }
        e.constraints = constraints

        ds = e.to_xarray()

        # find the latest timestamp with valid data (it appears that their fill value is 59.99255)
        idx = np.where(np.logical_and(ds.wind_speed >= 0, ds.wind_speed < 59.9))[0]
        valid_ts = ds.time.values[idx]

        last_ts = pd.to_datetime(np.nanmax(valid_ts)).strftime('%Y-%m-%d %H:%M')
        valid_2wk = np.sum(valid_ts>pd.Timestamp.now()-timedelta(days=14))

        if pd.Timestamp.now()-pd.to_datetime(last_ts) > timedelta(days=3):
            alert_outdated = np.append(alert_outdated,h)
        if valid_2wk < 24*7:
            alert_lowdata = np.append(alert_lowdata,h)

        if print_all:
            # return the most recent timestamp containing valid data for the height of interest
            print(f'    - {h}m: latest data {last_ts} GMT ({valid_2wk} valid data points in last 14 days)')
    
    if not print_all:
        if len(alert_outdated)>0:
            print(f'ALERT: Atlantic Shores data is outdated > 3 days at heights {alert_outdated}')
        if len(alert_lowdata)>0:
            print(f'ALERT: Atlantic Shores data is sparse for last two weeks at heights {alert_lowdata}')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('-x', '--alerts_only',
                            dest='alerts',
                            default=False,
                            type=bool,
                            choices=[True, False],
                            help='True to print alerts only (default False prints all info)')
    
    arg_parser.add_argument('-b', '--buoy',
                            dest='buoy',
                            default='ASOW-4',
                            type=str,
                            choices=['ASOW-4', 'ASOW-6'],
                            help='Atlantic Shores buoy code')

    arg_parser.add_argument('-z', '--height',
                            dest='height',
                            default=[10, 80, 160],
                            choices=[10, 80, 160],
                            help='choose a height in meters')

    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))
