#!/usr/bin/env python

"""
Author: Lori Garzio on 8/24/2022
Last modified: Laura Nazzaro 11/2/2022
Check latest timestamp for RU-WRF in MARACOOS Oceansmap: https://oceansmap.maracoos.org/.
"""

import argparse
import sys
import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta


def main(args):
    print_all = not args.alerts
    model_ver = args.model_ver

    if model_ver == '3km':
        url = 'https://tds.maracoos.org/thredds/dodsC/Rutgers_WRF_3km.nc/Best'
    elif model_ver == '9km':
        url = 'https://tds.maracoos.org/thredds/dodsC/Rutgers_WRF_9km.nc/TwoD/LambertConformal_224X254-38p80N-71p12W'
    else:
        url = None
        if not print_all:
            print(f'ALERT: Invalid WRF model version: {model_ver}')
    
    if url:
        if print_all:
            url_split=': '.join(url.split(':'))
            print(f'WRF {model_ver} data in MARACOOS Oceansmap:')
            print(f'    - checked: {url_split}')

        ds = xr.open_dataset(url)

        # find the latest model run time
        last_ts = pd.to_datetime(np.nanmax(ds.reftime.values)).strftime('%Y-%m-%d %H:%M')

        if not print_all and pd.Timestamp.now()-pd.to_datetime(last_ts) > timedelta(days=1):
            print(f'ALERT: Oceansmap {model_ver} is outdated > 1 day')

        if print_all:
            # return the most recent model run timestamp
            print(f'    - available through {last_ts} GMT')


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
