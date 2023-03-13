#!/usr/bin/env python

"""
Author: Lori Garzio 1/3/2023 (modified from Matlab code from Sage Lichtenwalner 1/18/2019)
Last modified: Lori Garzio 1/12/2023
Power Potential (sum of power binned by wind speed)
"""

import datetime as dt
import os
import argparse
import sys
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import calendar
pd.set_option('display.width', 320, "display.max_columns", 15)  # for display in pycharm console
plt.rcParams.update({'font.size': 16})  # all font sizes are 12 unless otherwise specified


def main(args):
    start_str = args.start_date
    end_str = args.end_date
    height = args.height
    buoy = args.buoy
    point_location = args.point_location
    save_dir = args.save_dir

    # grab the location of the specified buoy
    df = pd.read_csv(point_location, skipinitialspace=True)
    site = df.loc[df['name'] == buoy]
    site_lat = site.latitude.values[0]
    site_lon = site.longitude.values[0]

    start = dt.datetime.strptime(start_str, '%Y%m%d')
    end = dt.datetime.strptime(end_str, '%Y%m%d') + dt.timedelta(hours=23)
    start_datetime = dt.datetime.strptime(start_str, '%Y%m%d')        #.strftime('%Y-%m-%d')
    end_datetime = dt.datetime.strptime(end_str, '%Y%m%d')            #.strftime('%Y-%m-%d')

    #new title with month and year
    m_start = start_datetime.month ; m_end = end_datetime.month
    start_title = calendar.month_name[m_start] +'-'+ str(start_datetime.year)
    end_title =  calendar.month_name[m_end] +'-'+ str(end_datetime.year)
    

    site_lat_title = np.round(site_lat, 2)
    site_lon_title = np.round(site_lon, 2)

    sdir = os.path.join(save_dir, str(start.year))
    os.makedirs(sdir, exist_ok=True)

    times = np.array([], dtype='datetime64[ns]')
    u = np.array([], dtype='float32')
    v = np.array([], dtype='float32')

    mlink = 'https://tds.marine.rutgers.edu/thredds/dodsC/cool/ruwrf/wrf_4_1_3km_processed/WRF_4.1_3km_Processed_Dataset_Best'
    with xr.open_dataset(mlink) as ds:
        ds = ds.sel(time=slice(start, end))
        wrf_lat = ds['XLAT']
        wrf_lon = ds['XLONG']

        # find the closest WRF gridpoint to the site
        a = abs(wrf_lat - site_lat) + abs(wrf_lon - site_lon)
        i, j = np.unravel_index(a.argmin(), a.shape)

        # subset the dataset for just that grid point
        ds = ds.sel(south_north=i, west_east=j, height=height)

        for tm in ds.time.values:
            u = np.append(u, ds.sel(time=tm)['U'].values)
            v = np.append(v, ds.sel(time=tm)['V'].values)
            times = np.append(times, tm)

    wind_speed = np.sqrt(u ** 2 + v ** 2)
    df = pd.read_csv('./files/wrf_lw15mw_power.csv')
    df['Power'] = df['Power']* 0.000001          #convert to gW
    power = np.interp(wind_speed, df['Wind Speed'], df['Power'])  # kW
    centers = np.arange(.5, 25.5, 1)
    power_sum = centers * np.nan

    #total power for time period
    power_total = ("{:.3f}".format(np.sum(power)))

    for i, c in enumerate(centers):
        idx = np.where(np.logical_and(wind_speed >= c - 0.5, wind_speed < c + 0.5))[0]
        if len(idx) > 0:
            power_sum[i] = np.sum(power[idx])


    #add in percent below max power
    count = sum(speed < 10.9 for speed in wind_speed)
     # Calculate the percentage
    percentage = count / len(wind_speed) * 100


    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(centers, power_sum, width=1, edgecolor='k', alpha=.9)
    ax.set_xlim(0, 25)
    ax.set_ylim(0,3.0)
    ax.set_ylabel('Total Energy (GWh)')
    ax.set_xlabel('Wind Speed Bin (m/s)')
    ax.set_title(f'Total Energy by Wind Speed at {buoy} ({site_lon_title}, {site_lat_title}) {height}m\n{start_title} to {end_title}, Total Power Generated = {power_total}GWh')

    # add max power line
    ax.axvline(x=10.9, color='k', linestyle='--')

    # calculate cumulative sum
    cs = np.cumsum(np.nan_to_num(power_sum, 0))
    cs = cs/np.nanmax(cs) * 100

    ax0 = ax.twinx()
    ax0.plot(centers, cs, linewidth=2, c='k')
    ax0.set_ylim(0, 101)
    ax0.set_ylabel('Cumulative Sum (%)')

    save_file = os.path.join(sdir, f'power_binned_ws_{buoy}_{start_str}_{end_str}.png')
    plt.savefig(save_file, dpi=300)
    plt.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('-s', '--start_date',
                            dest='start_date',
                            default='20220901',
                            type=str,
                            help='Start Date in format YYYYMMDD')

    arg_parser.add_argument('-e', '--end_date',
                            dest='end_date',
                            default='20221130',
                            type=str,
                            help='End Date in format YYYYMMDD')

    arg_parser.add_argument('-b', '--buoy',
                            dest='buoy',
                            default='N2',
                            type=str,
                            help='Enter a buoy code, they can be found in wrf_validation_points.csv or virtual_met_towers.csv (make sure point_location argument matches)')

    arg_parser.add_argument('-p', '--point_location',
                            dest='point_location',
                            default='virtual_met_towers.csv',
                            type=str,
                            help='choose .csv file of lat, lons, and buoy codes')

    arg_parser.add_argument('-z', '--height',
                            dest='height',
                            default=160,
                            type=list,
                            help='choose a height in meters')

    arg_parser.add_argument('-save_dir',
                            default='/www/web/rucool/windenergy/ru-wrf/report_figs',
                            type=str,
                            help='Full directory path to save output plots.')

    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))
