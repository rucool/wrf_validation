#!/usr/bin/env python

import functions_and_loaders as fnl
import sys
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import cmocean as cmo
from scipy import stats


def plot_val_time_series(start_date, end_date, buoy, height, ws_df, dt_df):
    # variable reassign
    obs_ws = ws_df[0]
    wrf_v41_ws = ws_df[1]
    nam_ws = ws_df[2]
    gfs_ws = ws_df[3]
    hrrr_ws = ws_df[4]

    obs_time = dt_df[0]
    wrf_v41_time = dt_df[1]
    nam_dt = dt_df[2]
    gfs_dt = dt_df[3]
    hrrr_dt = dt_df[4]

    # Statistics Setup
    mf_41 = fnl.metrics(obs_ws, wrf_v41_ws)
    nam_m = fnl.metrics(obs_ws, nam_ws)
    hrrr_m = fnl.metrics(obs_ws, hrrr_ws)
    gfs_m = fnl.metrics(obs_ws, gfs_ws)

    # Plotting Start
    plt.figure(figsize=(14, 5))
    plt.style.use(u'seaborn-colorblind')
    lw = 1

    line3, = plt.plot(obs_time, obs_ws, color='black', label=buoy[0], linewidth=lw+.5, zorder=3)
    line1, = plt.plot(wrf_v41_time, wrf_v41_ws, color='red', label='RU WRF', linewidth=lw, zorder=5)

    # Power Law Wind Speed Change
    if height[0] == 160:
        alpha = 0.14
        nam_ws = nam_ws*(160/80)**alpha
        gfs_ws = gfs_ws*(160/100)**alpha
        hrrr_ws = hrrr_ws*(160/80)**alpha
        print('Power Law used')
    else:
        print(str(height[0]) + 'm was used, no power law')
        line5, = plt.plot(hrrr_dt, hrrr_ws, color='tab:blue', label='HRRR', linewidth=lw, zorder=4)
        # line4, = plt.plot(nam_dt, nam_ws, color='tab:olive', label='NAM', linewidth=lw-1, zorder=2)
        # line6, = plt.plot(gfs_dt, gfs_ws, color='tab:green', label='GFS', linewidth=lw-1, zorder=1)

    plt.ylabel('wind speed (m/s)')
    plt.xlabel('start date: ' + start_date.strftime("%Y/%m/%d"))
    plt.title('Wind Speeds at ' + buoy[0] + ' at ' + str(height[0]) + 'm')
    plt.legend(loc='best', fontsize='medium')
    plt.ylim(bottom=0)
    plt.grid(True)
    ax = plt.gca()
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    columns = ('Model', 'RMS', 'CRMS', 'MB', 'Count')

    metric_frame = {'Model': ['RU WRF', 'NAM', 'GFS', 'HRRR'],
                    'RMS':   np.round([mf_41[0], nam_m[0], gfs_m[0], hrrr_m[0]], 3),
                    'CRMS':  np.round([mf_41[1], nam_m[1], gfs_m[1], hrrr_m[1]], 3),
                    'MB':    np.round([mf_41[2], nam_m[2], gfs_m[2], hrrr_m[2]], 3),
                    'Count': [mf_41[3], nam_m[3], gfs_m[3], hrrr_m[3]]
                    }

    metric_frame = pd.DataFrame(metric_frame)

    metric_frame_1 = {'Model': ['RU WRF', 'HRRR'],
                      'RMS':   np.round([mf_41[0], hrrr_m[0]], 3),
                      'CRMS':  np.round([mf_41[1], hrrr_m[1]], 3),
                      'MB':    np.round([mf_41[2], hrrr_m[2]], 3),
                      'Count':          [mf_41[3], hrrr_m[3]]
                      }

    metric_frame_1 = pd.DataFrame(metric_frame_1)

    ds_table_1 = plt.table(metric_frame_1.values, colLabels=columns, bbox=([.1, -.5, .3, .3]))

    plt.savefig('/Users/JadenD/PycharmProjects/wrf_validation/figures/yearly_validation/ws' +
                '_' + buoy[0] +
                '_' + str(height[0]) + 'm'
                '_' + start_date.strftime("%Y%m%d") +
                '_' + end_date.strftime("%Y%m%d") + '.png',
                dpi=300, bbox_inches='tight')

    os.makedirs('/Volumes/www/cool/mrs/weather/RUWRF/validation/yearly/' +
                buoy[0] + '/time_series/wind_speed/' + start_date.strftime("%Y%m"), exist_ok=True)
    plt.savefig('/Volumes/www/cool/mrs/weather/RUWRF/validation/yearly/' + buoy[0] + '/time_series/wind_speed' +
                '/' + start_date.strftime("%Y%m") + '/'
                'ws' +
                '_' + buoy[0] +
                '_' + str(height[0]) + 'm'
                '_' + start_date.strftime("%Y%m%d") +
                '_' + end_date.strftime("%Y%m%d") + '.png',
                dpi=300, bbox_inches='tight')

    metric_frame.to_csv('/Users/JadenD/PycharmProjects/wrf_validation/figures/yearly_validation/stats' +
                        '_' + buoy[0] +
                        '_' + str(height[0]) + 'm'
                        '_' + start_date.strftime("%Y%m%d") +
                        '_' + end_date.strftime("%Y%m%d") + '.csv', index=None)

    os.makedirs('/Volumes/www/cool/mrs/weather/RUWRF/validation/yearly/' +
                buoy[0] + '/statistics/wind_speed/' + start_date.strftime("%Y%m"), exist_ok=True)
    metric_frame.to_csv('/Volumes/www/cool/mrs/weather/RUWRF/validation/yearly/' + buoy[0] + '/statistics/wind_speed' +
                        '/' + start_date.strftime("%Y%m") + '/'
                        'stats' +
                        '_' + buoy[0] +
                        '_' + str(height[0]) + 'm'
                        '_' + start_date.strftime("%Y%m%d") +
                        '_' + end_date.strftime("%Y%m%d") + '.csv', index=None)

    print(metric_frame)

    plt.clf()
    plt.close()

    return


def plot_heatmap(start_date, end_date, buoy, height, ws_df):
    total_time = pd.date_range(start_date, end_date, freq='H')
    # variable reassign
    obs_ws = ws_df[0]
    wrf_ws = ws_df[1]
    nam_ws = ws_df[2]
    gfs_ws = ws_df[3]
    hrrr_ws = ws_df[4]

    # Statistics Setup
    wrf_m = fnl.metrics(obs_ws, wrf_ws)
    nam_m = fnl.metrics(obs_ws, nam_ws)
    hrrr_m = fnl.metrics(obs_ws, hrrr_ws)
    gfs_m = fnl.metrics(obs_ws, gfs_ws)

    # Statistics Setup for Wind Speeds between 3m/s and 15m/s
    # binning and making a new dataset so original doesn't get NaN
    obs_ws_b = obs_ws.copy()
    wrf_ws_b = wrf_ws.copy()
    nam_ws_b = nam_ws.copy()
    gfs_ws_b = gfs_ws.copy()
    hrrr_ws_b = hrrr_ws.copy()
    obs_ws_b[(obs_ws_b > 15) | (obs_ws_b < 3)] = np.nan
    # wrf_ws_b[(wrf_ws_b > 15) | (wrf_ws_b < 3)] = np.nan
    # nam_ws_b[(nam_ws_b > 15) | (nam_ws_b < 3)] = np.nan
    # gfs_ws_b[(gfs_ws_b > 15) | (gfs_ws_b < 3)] = np.nan
    # hrrr_ws_b[(hrrr_ws_b > 15) | (hrrr_ws_b < 3)] = np.nan

    wrf_b = fnl.metrics(obs_ws_b, wrf_ws_b)
    nam_b = fnl.metrics(obs_ws_b, nam_ws_b)
    hrrr_b = fnl.metrics(obs_ws_b, hrrr_ws_b)
    gfs_b = fnl.metrics(obs_ws_b, gfs_ws_b)

    # Loop df setup
    wind_speeds = [wrf_ws, nam_ws, gfs_ws, hrrr_ws]
    wind_speeds_b = [wrf_ws_b, nam_ws_b, gfs_ws_b, hrrr_ws_b]
    model_names = ['RU WRF', 'NAM', 'GFS', 'HRRR']
    model_names_dir = ['RUWRF', 'NAM', 'GFS', 'HRRR']
    metrics_n = [wrf_m, nam_m, gfs_m, hrrr_m]
    metrics_b = [wrf_b, nam_b, gfs_b, hrrr_b]

    for ii in range(0, 4):
        # Line stats setup
        # unbinned data
        idx = np.isfinite(obs_ws) & np.isfinite(wind_speeds[ii])
        slope, intercept, r_value, p_value, std_err = stats.linregress(obs_ws[idx],
                                                                       wind_speeds[ii][idx])
        r2_value = r_value ** 2

        # binned data
        idx_b = np.isfinite(obs_ws_b) & np.isfinite(wind_speeds_b[ii])
        slope_b, intercept_b, r_value_b, p_value_b, std_err_b = stats.linregress(obs_ws_b[idx_b],
                                                                                 wind_speeds_b[ii][idx_b])
        r2_value_b = r_value_b ** 2

        # figure setup
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        cmap = cmo.cm.algae
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
        hexplot = plt.hexbin(obs_ws[idx], wind_speeds[ii][idx],
                             cmap=cmap, linewidths=.1, gridsize=50, mincnt=1, vmin=0, vmax=60) #, bins='log', cmap='jet')

        plt.plot([0, 25], [0, 25], 'silver')
        line1 = plt.plot(obs_ws_b[idx_b], intercept_b + slope_b * obs_ws_b[idx_b], linestyle='-', color='red')
        line2 = plt.plot(obs_ws[idx], intercept + slope * obs_ws[idx], linestyle='-', color='tab:red')

        plt.xlabel('Buoy: ' + buoy[0] + ' Wind Speed (m/s)', fontsize='x-large')
        plt.ylabel(model_names[ii] + ' Wind Speed (m/s)', fontsize='x-large')
        plt.text(2.5, -11,
                 'All Wind Speeds' + '\n' +
                 'slope: ' + str("{0:.2f}".format(slope)) + '\n' +
                 'intercept: ' + str("{0:.2f}".format(intercept)) + '\n' +
                 'R-squared: ' + str("{0:.2f}".format(r2_value)) + '\n' +
                 'RMS: ' + str("{0:.2f}".format(metrics_n[ii][0])) + '\n' +
                 'model bias: ' + str("{0:.2f}".format(metrics_n[ii][2])) + '\n' +
                 'percent uptime: ' + str("{0:.2f}%".format((metrics_n[ii][3] / len(total_time))*100)) + '\n' +
                 'obs counts above 25 m/s: ' + str("{0:.0f}".format(sum(obs_ws > 25))) + '\n' +
                 'model counts above 25 m/s: ' + str("{0:.0f}".format(sum(wind_speeds[ii][idx] > 25))),
                 bbox=dict(facecolor='white', alpha=1), fontsize='medium', ha="left",
                 )
        plt.text(14.5, -8.7,
                 'Between 3 and 15 (m/s)' + '\n' +
                 'slope: ' + str("{0:.2f}".format(slope_b)) + '\n' +
                 'intercept: ' + str("{0:.2f}".format(intercept_b)) + '\n' +
                 'R-squared: ' + str("{0:.2f}".format(r2_value_b)) + '\n' +
                 'RMS: ' + str("{0:.2f}".format(metrics_b[ii][0])) + '\n' +
                 'model bias: ' + str("{0:.2f}".format(metrics_b[ii][2])),
                 bbox=dict(facecolor='white', alpha=1), fontsize='medium', ha="left",
                 )

        plt.title('Wind Speeds at ' + buoy[0] + ' at ' + str(height[0]) + 'm ' + '\n' + start_date.strftime("%Y%m%d") +
                  ' to ' + end_date.strftime("%Y%m%d"),
                  fontsize='large')

        plt.grid(True)
        plt.xlim(left=0, right=25)
        plt.ylim(bottom=0, top=25)

        cb = fig.colorbar(
            hexplot,
            ax=ax,
            # cmap=cmap,
            extend='max',
            spacing='proportional',
            label='counts',
            # norm=norm,
            # ticks=bounds
        )

        os.makedirs('/Volumes/www/cool/mrs/weather/RUWRF/validation/yearly/' +
                    buoy[0] + '/heatmap/wind_speed/' + start_date.strftime("%Y%m"), exist_ok=True)
        plt.savefig('/Volumes/www/cool/mrs/weather/RUWRF/validation/yearly/' + buoy[0] + '/heatmap/wind_speed' +
                    '/' + start_date.strftime("%Y%m") + '/'
                    'ws' +
                    '_' + buoy[0] +
                    '_' + model_names_dir[ii] +
                    '_' + str(height[0]) + 'm'
                    '_' + start_date.strftime("%Y%m%d") +
                    '_' + end_date.strftime("%Y%m%d") + '.png',
                    dpi=300, bbox_inches='tight')

        plt.clf()
        plt.close()


def main(args):
    # Model Range and NYSERDA BUOY and other models
    print(args)
    start_date = datetime.strptime(args.start_date, '%Y%m%d')
    end_date = datetime.strptime(args.end_date, '%Y%m%d') - timedelta(hours=1)
    buoy = [args.buoy, bytes(args.buoy, 'utf-8')]
    height = [args.height]

    # WRF Load
    wrf_v41_ds = fnl.load_wrf(start_date, end_date, 1, 'v4.1', args.point_location, buoy=buoy, height=height)
    wrf_v41_ws = wrf_v41_ds.wind_speed.sel(time=slice(start_date, end_date), station=buoy[1], height=height).data
    wrf_v41_ws = wrf_v41_ws.reshape(wrf_v41_ws.__len__())
    wrf_v41_time = wrf_v41_ds.time.sel(time=slice(start_date, end_date)).data

    if buoy[0] in ['NYNE05', 'NYSWE05', 'NYSE06']:
        nys_ws_1hr_nonav = fnl.load_nyserda_ws(buoy, height[0], start_date, end_date)
        nys_ws_1hr_nonav[nys_ws_1hr_nonav > 55] = np.nan
        obs_time = nys_ws_1hr_nonav.index
        obs_ws = nys_ws_1hr_nonav.values
    elif buoy[0] == 'SODAR':
        r = pd.date_range(start=start_date, end=end_date, freq='H')
        df_ws, df_wd, df_dt = fnl.sodar_loader(start_date, end_date,  height=height)
        df_ws = df_ws[df_ws['height'] == height[0]]
        df_ws = df_ws.set_index('dt').reindex(r).fillna(np.nan).rename_axis('dt').reset_index()
        obs_time = df_ws['dt'].values
        obs_ws = df_ws['value'].values
    elif buoy[0] == 'ASOSB6':
        try:
            as_ds = fnl.load_ASOSB(start_date, end_date, ASbuoy=6, height=height)
        except:
            print('Atlantic Shores loader failed, date might not exist')
            as_ds = []

        as_dt = pd.to_datetime(as_ds.time.data, format='%m-%d-%Y %H:%M')
        time_h = pd.date_range(start_date, end_date, freq='H')
        time_m = pd.date_range(start_date, end_date, freq='10min')
        as_ds.wind_speed[as_ds.wind_speed > 55] = np.nan
        as_ds.wind_speed[as_ds.wind_speed < 0] = np.nan
        as_ws = pd.Series(as_ds.wind_speed.values, index=as_dt)
        as_ws = as_ws.reindex(time_h)
        as_ws_m = pd.Series(as_ds.wind_speed.values, index=as_dt)
        as_ws_m = as_ws_m.reindex(time_m)
        # as_ds.time.data = pd.to_numeric(as_ds.time.data, errors='coerce')
        as_ws_av = []
        for i in range(0, len(as_ws_m), 6):
            as_ws_av.append(np.mean(as_ws_m[i:i + 6]))
        asosb_ws_1hr_avg = pd.Series(as_ws_av, index=as_ws.index)
        obs_time = asosb_ws_1hr_avg.index
        obs_ws = asosb_ws_1hr_avg.values
    elif buoy[0] == 'ASOSB4':
        try:
            as_ds = fnl.load_ASOSB(start_date, end_date, ASbuoy=4, height=height)
        except:
            print('Atlantic Shores loader failed, date might not exist')
            as_ds = []

        as_dt = pd.to_datetime(as_ds.time.data, format='%m-%d-%Y %H:%M')
        time_h = pd.date_range(start_date, end_date, freq='H')
        time_m = pd.date_range(start_date, end_date, freq='10min')
        as_ds.wind_speed[as_ds.wind_speed > 55] = np.nan
        as_ds.wind_speed[as_ds.wind_speed < 0] = np.nan
        as_ws = pd.Series(as_ds.wind_speed.values, index=as_dt)
        as_ws = as_ws.reindex(time_h)
        as_ws_m = pd.Series(as_ds.wind_speed.values, index=as_dt)
        as_ws_m = as_ws_m.reindex(time_m)
        # as_ds.time.data = pd.to_numeric(as_ds.time.data, errors='coerce')
        as_ws_av = []
        for i in range(0, len(as_ws_m), 6):
            as_ws_av.append(np.mean(as_ws_m[i:i + 6]))
        asosb_ws_1hr_avg = pd.Series(as_ws_av, index=as_ws.index)
        obs_time = asosb_ws_1hr_avg.index
        obs_ws = asosb_ws_1hr_avg.values
    else:
        print('Not a valid validation point.')
        obs_time = []
        obs_ws = []

    nam_ws, nam_dt = fnl.load_nam(start_date, end_date, buoy[0], args.point_location, height=height)
    gfs_ws, gfs_dt = fnl.load_gfs(start_date, end_date, buoy[0], args.point_location, height=height)
    hrrr_ws, hrrr_dt = fnl.load_hrrr(start_date, end_date, buoy[0], args.point_location, height=height)

    ws_df = [obs_ws, wrf_v41_ws, nam_ws, gfs_ws, hrrr_ws]
    dt_df = [obs_time, wrf_v41_time, nam_dt, gfs_dt, hrrr_dt]

    plot_heatmap(start_date, end_date, buoy, height, ws_df)
    plot_val_time_series(start_date, end_date, buoy, height, ws_df, dt_df)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('-s', '--start_date',
                            dest='start_date',
                            default='20201101',
                            type=str,
                            help='Start Date for run in YYYYMMDD ')

    arg_parser.add_argument('-e', '--end_date',
                            dest='end_date',
                            default='20201108',
                            type=str,
                            help='End Date for run in YYYYMMDD (one hour will be subtracted from this date)')

    arg_parser.add_argument('-b', '--buoy',
                            dest='buoy',
                            default='ASOSB6',
                            type=str,
                            help='Enter a buoy code, they can be found in wrf_validation_points.csv')

    arg_parser.add_argument('-p', '--point_location',
                            dest='point_location',
                            default='wrf_validation_points.csv',
                            type=str,
                            help='choose .csv file of lat, lons, and buoy codes')

    arg_parser.add_argument('-z', '--height',
                            dest='height',
                            default=80,
                            type=list,
                            help='choose a height in meters 80 and 160 supported')

    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))
