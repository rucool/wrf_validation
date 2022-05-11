import functions_and_loaders as fnl
import xarray as xr
import seaborn as sns
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


def plot_val_time_series(start_date, end_date, buoy, height, ws_df, dt_df, time_span):
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

    plt.savefig('/Users/JadenD/PycharmProjects/wrf_validation/figures/' + time_span + '_validation/ws' +
                '_' + buoy[0] +
                '_' + str(height[0]) + 'm' +
                '_' + start_date.strftime("%Y%m%d") +
                '_' + end_date.strftime("%Y%m%d") + '.png',
                dpi=300, bbox_inches='tight')

    os.makedirs('/Volumes/www/cool/mrs/weather/RUWRF/validation/' + time_span + '/' +
                buoy[0] + '/time_series/wind_speed/' + start_date.strftime("%Y%m"), exist_ok=True)
    plt.savefig('/Volumes/www/cool/mrs/weather/RUWRF/validation/' + time_span + '/' + buoy[0] + '/time_series/wind_speed' +
                '/' + start_date.strftime("%Y%m") + '/' +
                'ws' +
                '_' + buoy[0] +
                '_' + str(height[0]) + 'm' +
                '_' + start_date.strftime("%Y%m%d") +
                '_' + end_date.strftime("%Y%m%d") + '.png',
                dpi=300, bbox_inches='tight')

    metric_frame.to_csv('/Users/JadenD/PycharmProjects/wrf_validation/figures/' + time_span + '_validation/stats' +
                        '_' + buoy[0] +
                        '_' + str(height[0]) + 'm' +
                        '_' + start_date.strftime("%Y%m%d") +
                        '_' + end_date.strftime("%Y%m%d") + '.csv', index=None)

    os.makedirs('/Volumes/www/cool/mrs/weather/RUWRF/validation/' + time_span + '/' +
                buoy[0] + '/statistics/wind_speed/' + start_date.strftime("%Y%m"), exist_ok=True)
    metric_frame.to_csv('/Volumes/www/cool/mrs/weather/RUWRF/validation/' + time_span + '/' + buoy[0] + '/statistics/wind_speed' +
                        '/' + start_date.strftime("%Y%m") + '/'
                        'stats' +
                        '_' + buoy[0] +
                        '_' + str(height[0]) + 'm' +
                        '_' + start_date.strftime("%Y%m%d") +
                        '_' + end_date.strftime("%Y%m%d") + '.csv', index=None)

    print(metric_frame)

    plt.clf()
    plt.close()

    return


def plot_heatmap(start_date, end_date, buoy, height, ws_df, time_span):
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

        if time_span == 'yearly':
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

        os.makedirs('/Volumes/www/cool/mrs/weather/RUWRF/validation/' + time_span + '/' +
                    buoy[0] + '/heatmap/wind_speed/' + start_date.strftime("%Y%m"), exist_ok=True)
        plt.savefig('/Volumes/www/cool/mrs/weather/RUWRF/validation/' + time_span + '/' + buoy[0] + '/heatmap/wind_speed' +
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
