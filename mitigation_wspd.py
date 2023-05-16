#Author: James Kim
#HFradar mitigation windspeed timeseries 

import datetime as dt
import os
import argparse
import sys
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import calendar

#####     Arguments   ######

def main(args):

    start_str = args.start_date
    end_str = args.end_date
    height = args.height
    buoy = args.buoy
    point_location = args.point_location
    save_dir = args.save_dir

    


    point_location = '/users/jameskim/Documents/rucool/Repositories/wrf_validation/virtual_met_towers.csv'
    df_locations = pd.read_csv(point_location, skipinitialspace=True)
    site =df_locations.buoy[df_locations['name'] == buoy]
    site_lat = site.latitude.values[0]
    site_lon = site.longitude.values[0]

    # create empty arrays for u and v compnonets of wind with time array

    times = np.array([],dtype='datetime64[ns]')
    u = np.array([],dtype = 'float32')
    v = np.array([],dtype = 'float32')


    mlink = 'https://tds.marine.rutgers.edu/thredds/dodsC/cool/ruwrf/wrf_4_1_3km_processed/WRF_4.1_3km_Processed_Dataset_Best'
    with xr.open_dataset(mlink) as ds:
        ds = ds.sel(time=slice(start,end))
        wrf_lat =ds['XLAT']
        wrf_lon =ds['XLONG']

        #find the closest WRF gridpoint to the site
        a = abs(wrf_lat-site_lat)+abs(wrf_lon - site_lon)
        i,j = np.unravel_index(a.argmin(),a.shape) #tranforms into a tuple for lat/lon coords

    
        #subset the dataset for jist that gridpoint
        ds = ds.sel(south_north=i, west_east=j, height=height)

        for tm in ds.time.values:
            u = np.append (u,ds.sel(time=tm)['U'].values)
            v = np.append(v,ds.sel(time = tm)['V'].values)
            times = np.append(times,tm)


    wind_speed = np.sqrt(u ** 2 + v ** 2)


    #calculate percentages
    total = len(wind_speed)
    below_3 = len(wind_speed[wind_speed < 3])
    between_3_10_9 = len(wind_speed[(wind_speed >= 3) & (wind_speed <= 10.9)])
    between_10_9_25 = len(wind_speed[(wind_speed > 10.9) & (wind_speed <= 25)])
    above_25 = len(wind_speed[wind_speed > 25])

    p_below_3 = round(below_3/total*100, 2)
    p_between_3_10_9 = round(between_3_10_9/total*100, 2)
    p_between_10_9_25 = round(between_10_9_25/total*100, 2)
    p_above_25 = round(above_25/total*100, 2)

    sdir = os.path.join(save_dir, str(start.year))
    os.makedirs(sdir, exist_ok=True)
    #plotting
    # set up the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # plot the timeseries of windspeed
    ax.plot(times, wind_speed, color='blue', linewidth=0.75)

    # add a dashed horizontal line at 3.0 m/s and 10.9 m/s
    ax.axhline(y=3.0, linestyle='--', color='black', linewidth=1)
    ax.axhline(y=10.9, linestyle='--', color='black', linewidth=1)

    # add labels and a title
    ax.set_xlabel('Time')
    ax.set_ylabel('Windspeed (m/s)')
    ax.set_title(f'Windspeeds(160m) at {buoy} from {start_str} to {end_str}')

    # add gridlines
    ax.grid(True, linestyle='--')

    # add a label with the percentage of windspeed in different ranges to the corner of the plot
    text = (f'Below 3.0m/s: {p_below_3}%\nBetween 3.0m/s and 10.9m/s: {p_between_3_10_9}%\nBetween 10.9m/s and 25m/s: {p_between_10_9_25}%\nAbove 25m/s: {p_above_25}%')
    ax.text(0.02, 0.85, text, transform=ax.transAxes, fontsize=12)

    # save to diretory


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
                            default='/www/web/rucool/windenergy/ru-wrf/mitigation/centroids/timeseries',
                            type=str,
                            help='Full directory path to save output plots.')

    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))