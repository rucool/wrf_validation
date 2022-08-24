#!/usr/bin/env python

"""
Author: Lori Garzio on 8/24/2022
Last modified: Lori Garzio 8/24/2022
Check latest timestamp for RU-WRF in MARACOOS Oceansmap: https://oceansmap.maracoos.org/.
"""

import argparse
import sys
import numpy as np
import pandas as pd
import xarray as xr


def main(args):
    model_ver = args.model_ver
    if model_ver == '3km':
        url = 'https://tds.maracoos.org/thredds/dodsC/Rutgers_WRF_3km.nc/Best'
    elif model_ver == '9km':
        url = 'https://tds.maracoos.org/thredds/dodsC/Rutgers_WRF_9km.nc/TwoD/LambertConformal_224X254-38p80N-71p12W'
    else:
        url = None
        print(f'Invalid WRF model version: {model_ver}')

    if url:
        ds = xr.open_dataset(url)

        # find the latest model run time
        last_ts = pd.to_datetime(np.nanmax(ds.reftime.values)).strftime('%Y-%m-%d %H:%M')

        # return the most recent model run timestamp
        print(f'{last_ts} GMT')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('-m', '--model_ver',
                            dest='model_ver',
                            default='3km',
                            type=str,
                            choices=['3km', '9km'],
                            help='RU-WRF model version (3km or 9km)')

    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))
