#!/usr/bin/env python

"""
Author: Lori Garzio on 8/24/2022
Last modified: Lori Garzio 8/24/2022
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
        'altitude=': 10
    }

    e = ERDDAP(server=url_erddap,
               protocol='tabledap',
               response='nc'
               )

    e.dataset_id = dataset_id
    e.constraints = constraints
    e.variables = variables

    ds = e.to_xarray()

    # find the latest timestamp
    last_ts = pd.to_datetime(np.nanmax(ds.time.values)).strftime('%Y-%m-%d %H:%M')

    # return the most recent model run timestamp
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

    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))
