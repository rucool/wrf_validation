#!/usr/bin/env python

"""
Author: Lori Garzio on 8/24/2022
Last modified: Lori Garzio 8/24/2022
Check latest timestamp for the Coastal Metocean Monitoring Station
Plots here: https://rucool.marine.rutgers.edu/data/meteorological-modeling/coastal-metocean-monitoring-station/
"""

import os
import sys
import numpy as np
import pandas as pd
import glob


def main():
    met_dir = '/home/coolgroup/MetData/CMOMS/surface_temp/surface'

    # find the most recent file
    last_file = sorted(glob.glob(os.path.join(met_dir, '*.csv')))[-1]
    df = pd.read_csv(last_file)

    tm = pd.to_datetime(np.array(df['time_stamp(utc)']))
    last_ts = pd.to_datetime(np.nanmax(tm)).strftime('%Y-%m-%d %H:%M')

    # return the most recent timestamp
    print(f'{last_ts} UTC')


if __name__ == '__main__':
    main()
    sys.exit(0)
