#!/usr/bin/env python

"""
Author: Lori Garzio on 8/24/2022
Last modified: Lori Garzio 8/29/2022
Check latest timestamp for the wind dataset from Atlantic Shores buoy data:
https://erddap.maracoos.org/erddap/search/index.html?page=1&itemsPerPage=1000&searchFor=atlantic+shores
"""

import argparse
import sys
import numpy as np
import pandas as pd
from erddapy import ERDDAP


def main(args):
    buoy = args.buoy
    height = args.height

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

    constraints = {
        'altitude=': height
    }

    e.dataset_id = dataset_id
    e.constraints = constraints
    e.variables = variables

    ds = e.to_xarray()

    # find the latest timestamp with valid data (it appears that their fill value is 59.99255)
    idx = np.where(np.logical_and(ds.wind_speed >= 0, ds.wind_speed < 59.9))[0]
    valid_ts = ds.time.values[idx]

    last_ts = pd.to_datetime(np.nanmax(valid_ts)).strftime('%Y-%m-%d %H:%M')

    # return the most recent timestamp containing valid data for the height of interest
    print(f'{last_ts} GMT')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('-b', '--buoy',
                            dest='buoy',
                            default='ASOW-4',
                            type=str,
                            choices=['ASOW-4', 'ASOW-6'],
                            help='Atlantic Shores buoy code')

    arg_parser.add_argument('-z', '--height',
                            dest='height',
                            default=80,
                            type=int,
                            choices=[10, 80, 160],
                            help='choose a height in meters')

    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))
