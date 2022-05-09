import functions_and_loaders as fnl
import xarray as xr
import seaborn as sns
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta


def plot_val_time_series(start_date, end_date, buoy, point_location, height):
    # Model Range and NYSERDA BUOY and other models

    # WRF Load
    wrf_v41_ds = fnl.load_wrf(start_date, end_date, 1, 'v4.1', point_location, buoy=buoy, height=height)
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
    elif buoy[0][0:5] == 'ASOSB':
        try:
            if buoy[0][-1] == '6':
                as_ds = fnl.load_ASOSB(start_date, end_date, ASbuoy=6, height=height)
            elif buoy[0][-1] == '4':
                as_ds = fnl.load_ASOSB(start_date, end_date, ASbuoy=4, height=height)
            else:
                print('Atlantic Shores Buoy Number might not exist')
                as_ds = []
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

    print(buoy[0] + ' is being used')

    nam_ws, nam_dt = fnl.load_nam(start_date, end_date, buoy[0], point_location, height=height)
    gfs_ws, gfs_dt = fnl.load_gfs(start_date, end_date, buoy[0], point_location, height=height)
    hrrr_ws, hrrr_dt = fnl.load_hrrr(start_date, end_date, buoy[0], point_location, height=height)

    # Statistics Setup
    mf_41 = fnl.metrics(obs_ws, wrf_v41_ws)
    nam_m = fnl.metrics(obs_ws, nam_ws)
    hrrr_m = fnl.metrics(obs_ws, hrrr_ws)
    gfs_m = fnl.metrics(obs_ws, gfs_ws)

    # Plotting Start
    plt.figure(figsize=(14, 5))
    plt.style.use(u'seaborn-colorblind')
    lw = 2

    line3, = plt.plot(obs_time, obs_ws, color='black', label=buoy[0], linewidth=lw+.75, zorder=4)
    line1, = plt.plot(wrf_v41_time, wrf_v41_ws, color='red', label='RU WRF', linewidth=lw, zorder=6)

    # Power Law Wind Speed Change
    if height[0] == 160:
        alpha = 0.14
        nam_ws = nam_ws*(160/80)**alpha
        gfs_ws = gfs_ws*(160/100)**alpha
        hrrr_ws = hrrr_ws*(160/80)**alpha
        print('Power Law used')
    else:
        print(str(height[0]) + 'm was used, no power law')
        line5, = plt.plot(hrrr_dt, hrrr_ws, color='tab:blue', label='HRRR', linewidth=lw, zorder=5)
        line4, = plt.plot(nam_dt, nam_ws, color='sienna', label='NAM', linewidth=lw-1, zorder=3)
        line6, = plt.plot(gfs_dt, gfs_ws, color='darkgreen', label='GFS', linewidth=lw-1, zorder=2)

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
                    'RMS': np.round([mf_41[0], nam_m[0], gfs_m[0], hrrr_m[0]], 3),
                    'CRMS': np.round([mf_41[1], nam_m[1], gfs_m[1], hrrr_m[1]], 3),
                    'MB': np.round([mf_41[2], nam_m[2], gfs_m[2], hrrr_m[2]], 3),
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

    metric_frame_2 = {'Model': ['NAM', 'GFS'],
                      'RMS':   np.round([nam_m[0], gfs_m[0]], 3),
                      'CRMS':  np.round([nam_m[1], gfs_m[1]], 3),
                      'MB':    np.round([nam_m[2], gfs_m[2]], 3),
                      'Count':          [nam_m[3], gfs_m[3]]
                      }

    metric_frame_2 = pd.DataFrame(metric_frame_2)

    ds_table_1 = plt.table(metric_frame_1.values, colLabels=columns, bbox=([.1, -.5, .3, .3]))
    ds_table_2 = plt.table(metric_frame_2.values, colLabels=columns, bbox=([.6, -.5, .3, .3]))

    plt.savefig('/Users/JadenD/PycharmProjects/wrf_validation/figures/weekly_validation/ws' +
                '_' + buoy[0] +
                '_' + str(height[0]) + 'm'
                '_' + start_date.strftime("%Y%m%d") +
                '_' + end_date.strftime("%Y%m%d") + '.png',
                dpi=300, bbox_inches='tight')

    os.makedirs('/Volumes/www/cool/mrs/weather/RUWRF/validation/weekly/' +
                buoy[0] + '/time_series/wind_speed/' + start_date.strftime("%Y%m"), exist_ok=True)
    plt.savefig('/Volumes/www/cool/mrs/weather/RUWRF/validation/weekly/' + buoy[0] + '/time_series/wind_speed' +
                '/' + start_date.strftime("%Y%m") + '/'
                'ws' +
                '_' + buoy[0] +
                '_' + str(height[0]) + 'm'
                '_' + start_date.strftime("%Y%m%d") +
                '_' + end_date.strftime("%Y%m%d") + '.png',
                dpi=300, bbox_inches='tight')

    metric_frame.to_csv('/Users/JadenD/PycharmProjects/wrf_validation/figures/weekly_validation/stats' +
                        '_' + buoy[0] +
                        '_' + str(height[0]) + 'm'
                        '_' + start_date.strftime("%Y%m%d") +
                        '_' + end_date.strftime("%Y%m%d") + '.csv', index=None)

    os.makedirs('/Volumes/www/cool/mrs/weather/RUWRF/validation/weekly/' +
                buoy[0] + '/statistics/wind_speed/' + start_date.strftime("%Y%m"), exist_ok=True)
    metric_frame.to_csv('/Volumes/www/cool/mrs/weather/RUWRF/validation/weekly/' + buoy[0] + '/statistics/wind_speed' +
                        '/' + start_date.strftime("%Y%m") + '/'
                        'stats' +
                        '_' + buoy[0] +
                        '_' + str(height[0]) + 'm'
                        '_' + start_date.strftime("%Y%m%d") +
                        '_' + end_date.strftime("%Y%m%d") + '.csv', index=None)

    print(metric_frame)

    return


def plot_val_fig_NYSERDA(start_date, end_date, buoy, point_location, height, folder):
    # Model Range and NYSERDA BUOY and other models
    plt.figure(figsize=(18, 4))
    plt.style.use(u'seaborn-colorblind')

    lw = 2
    wrf_v41_ds = fnl.load_wrf(start_date, end_date, 1, 'v4.1', point_location, buoy=buoy, height=height)
    wrf_v41_ws = wrf_v41_ds.wind_speed.sel(time=slice(start_date, end_date), station=buoy[1], height=height).data
    wrf_v41_ws = wrf_v41_ws.reshape(wrf_v41_ws.__len__())
    wrf_v41_time = wrf_v41_ds.time.sel(time=slice(start_date, end_date)).data

    nys_ws_1hr_nonav = fnl.load_nyserda_ws(buoy, height, start_date, end_date)
    nys_ws_1hr_nonav[nys_ws_1hr_nonav > 55] = np.nan

    nam_ws, nam_dt = fnl.load_nam(start_date, end_date, buoy[0], point_location, height=height)
    gfs_ws, gfs_dt = fnl.load_gfs(start_date, end_date, buoy[0], point_location, height=height)
    hrrr_ws, hrrr_dt = fnl.load_hrrr(start_date, end_date, buoy[0], point_location, height=height)

    # Power Law Wind Speed Change
    if height[0] == 160:
        alpha = 0.14
        nam_ws = nam_ws*(160/80)**alpha
        gfs_ws = gfs_ws*(160/100)**alpha
        hrrr_ws = hrrr_ws*(160/80)**alpha
        print('Power Law used')
    else:
        print(str(height[0]) + 'm was used, no power law')

    line1, = plt.plot(wrf_v41_time, wrf_v41_ws, label='WRF 4.1', linewidth=lw)
    line3, = plt.plot(nys_ws_1hr_nonav.index, nys_ws_1hr_nonav.values,
                      color='black', label=buoy[0], linewidth=lw)
    line4, = plt.plot(nam_dt, nam_ws, '-.', label='NAM', linewidth=lw)
    line5, = plt.plot(hrrr_dt, hrrr_ws, '-.', label='HRRR', linewidth=lw)
    line6, = plt.plot(gfs_dt, gfs_ws, '-.', label='GFS', linewidth=lw)

    plt.ylabel('wind speed (m/s)')
    plt.xlabel('start date: ' + start_date.strftime("%Y/%m/%d"))
    plt.title('Wind Speeds at ' + buoy[0] + ' at ' + str(height[0]) + 'm')
    plt.legend(handles=[line1, line3, line4, line5, line6], loc='best', fontsize='medium')
    plt.ylim(bottom=0)
    plt.grid(True)
    ax = plt.gca()
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    params = {
        'axes.labelsize': 8,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.usetex': False
    }
    plt.rcParams.update(params)

    plt.savefig('/Users/JadenD/PycharmProjects/wrf_validation/figures/weekly_validation/ws' +
                '_' + buoy[0] +
                '_' + str(height[0]) + 'm'
                '_' + start_date.strftime("%Y%m%d") +
                '_' + end_date.strftime("%Y%m%d") + '.png',
                dpi=300)

    os.makedirs('/Volumes/www/cool/mrs/weather/RUWRF/validation/' + folder + '/' +
             buoy[0] + '/time_series/wind_speed/' + start_date.strftime("%Y%m%d"), exist_ok=True)
    plt.savefig('/Volumes/www/cool/mrs/weather/RUWRF/validation/' + folder + '/' + buoy[0] + '/time_series/wind_speed' +
                '/' + start_date.strftime("%Y%m%d") + '/'
                'ws' +
                '_' + buoy[0] +
                '_' + str(height[0]) + 'm'
                '_' + start_date.strftime("%Y%m%d") +
                '_' + end_date.strftime("%Y%m%d") + '.png',
                dpi=300)

    mf_41 = fnl.metrics(nys_ws_1hr_nonav.values, wrf_v41_ws)
    nam_m = fnl.metrics(nys_ws_1hr_nonav.values, nam_ws)
    hrrr_m = fnl.metrics(nys_ws_1hr_nonav.values, hrrr_ws)
    gfs_m = fnl.metrics(nys_ws_1hr_nonav.values, gfs_ws)

    metric_frame = {'Model': ['WRF 4.1', 'NAM', 'GFS', 'HRRR'],
                    'RMS': np.round([mf_41[0], nam_m[0], gfs_m[0], hrrr_m[0]], 3),
                    'CRMS': np.round([mf_41[1], nam_m[1], gfs_m[1], hrrr_m[1]], 3),
                    'MB': np.round([mf_41[2], nam_m[2], gfs_m[2], hrrr_m[2]], 3),
                    'Count': [mf_41[3], nam_m[3], gfs_m[3], hrrr_m[3]]
                    }

    metric_frame = pd.DataFrame(metric_frame)

    metric_frame.to_csv('/Users/JadenD/PycharmProjects/wrf_validation/figures/weekly_validation/stats' +
                        '_' + buoy[0] +
                        '_' + str(height[0]) + 'm' 
                        '_' + start_date.strftime("%Y%m%d") +
                        '_' + end_date.strftime("%Y%m%d") + '.csv', index=None)

    os.makedirs('/Volumes/www/cool/mrs/weather/RUWRF/validation/' + folder + '/' +
             buoy[0] + '/statistics/wind_speed/' + start_date.strftime("%Y%m"), exist_ok=True)
    metric_frame.to_csv('/Volumes/www/cool/mrs/weather/RUWRF/validation/' + folder + '/' + buoy[0] + '/statistics' +
                        '/' + start_date.strftime("%Y%m%d") + '/'
                        'stats' +
                        '_' + buoy[0] +
                        '_' + str(height[0]) + 'm'
                        '_' + start_date.strftime("%Y%m%d") +
                        '_' + end_date.strftime("%Y%m%d") + '.csv', index=None)

    print(metric_frame)

    return


def plot_val_fig_ASOSB(start_date, end_date, buoy, point_location, height):
    # Model Range and ASOSB BUOY and other models
    lw = 2
    wrf_v41_ds = fnl.load_wrf(start_date, end_date, 1, 'v4.1', point_location, buoy=buoy, height=height)
    wrf_v41_ws = wrf_v41_ds.wind_speed.sel(time=slice(start_date, end_date), station=buoy[1], height=height).data
    wrf_v41_ws = wrf_v41_ws.reshape(wrf_v41_ws.__len__())
    wrf_v41_time = wrf_v41_ds.time.sel(time=slice(start_date, end_date)).data

    try:
        as_ds = fnl.load_ASOSB(start_date, end_date, ASbuoy=6, height=height)
    except:
        print('Atlantic Shores loader failed, date might not exist')
        as_ds = []

    as_dt = pd.to_datetime(as_ds.time.data, format='%m-%d-%Y %H:%M')
    time_h = pd.date_range(start_date, end_date - timedelta(hours=1), freq='H')
    time_m = pd.date_range(start_date, end_date - timedelta(hours=1), freq='10min')

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

    nam_ws, nam_dt = fnl.load_nam(start_date, end_date, buoy[0], point_location, height=height)
    gfs_ws, gfs_dt = fnl.load_gfs(start_date, end_date, buoy[0], point_location, height=height)
    hrrr_ws, hrrr_dt = fnl.load_hrrr(start_date, end_date, buoy[0], point_location, height=height)

    # Power Law Wind Speed Change
    if height[0] == 160:
        print('Power Law Used to bring models to 160m')
        alpha = 0.14
        nam_ws = nam_ws*(160/80)**alpha
        gfs_ws = gfs_ws*(160/100)**alpha
        hrrr_ws = hrrr_ws*(160/80)**alpha
    else:
        print(str(height[0]) + 'm was used, no power law')

    # Figure Generation
    plt.figure(figsize=(18, 4))
    plt.style.use(u'seaborn-colorblind')

    line1, = plt.plot(wrf_v41_time, wrf_v41_ws, label='WRF 4.1', linewidth=lw)
    line3, = plt.plot(asosb_ws_1hr_avg.index, asosb_ws_1hr_avg.values,
                      color='black', label=buoy[0], linewidth=lw
                      )
    line4, = plt.plot(nam_dt, nam_ws, '-.', label='NAM', linewidth=lw)
    line5, = plt.plot(hrrr_dt, hrrr_ws, '-.', label='HRRR', linewidth=lw)
    line6, = plt.plot(gfs_dt, gfs_ws, '-.', label='GFS', linewidth=lw)

    plt.ylabel('wind speed (m/s)')
    plt.xlabel('start date: ' + start_date.strftime("%Y/%m/%d"))
    plt.title('Wind Speeds at ' + buoy[0] + ' at ' + str(height[0]) + 'm')

    plt.legend(handles=[line1, line3, line4, line5, line6], loc='best', fontsize='medium')
    ###
    plt.ylim(bottom=0)
    plt.grid(True)
    ax = plt.gca()
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    params = {
        'axes.labelsize': 8,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.usetex': False
    }
    plt.rcParams.update(params)

    plt.savefig('ws_' + buoy[0] + '_' +
                start_date.strftime("%Y%m%d") + '_' + end_date.strftime("%Y%m%d") + '_' +
                str(height[0]) + 'm.png',
                dpi=300)

    mf_41 = fnl.metrics(asosb_ws_1hr_avg.values, wrf_v41_ws)
    nam_m = fnl.metrics(asosb_ws_1hr_avg.values, nam_ws)
    hrrr_m = fnl.metrics(asosb_ws_1hr_avg.values, hrrr_ws)
    gfs_m = fnl.metrics(asosb_ws_1hr_avg.values, gfs_ws)

    metric_frame = {'Model': ['WRF 4.1', 'NAM', 'GFS', 'HRRR'],
                    'RMS': np.round([mf_41[0], nam_m[0], gfs_m[0], hrrr_m[0]], 3),
                    'CRMS': np.round([mf_41[1], nam_m[1], gfs_m[1], hrrr_m[1]], 3),
                    'MB': np.round([mf_41[2], nam_m[2], gfs_m[2], hrrr_m[2]], 3),
                    'Count': [mf_41[3], nam_m[3], gfs_m[3], hrrr_m[3]]
                    }

    metric_frame = pd.DataFrame(metric_frame)

    metric_frame.to_csv('~/PycharmProjects/covid19/ASOSB_weekly_' + str(height[0]) + 'm_' + buoy[0] + '_' +
                        start_date.strftime("%Y%m%d") + '_' + end_date.strftime("%Y%m%d") + '.csv', index=None)

    print(metric_frame)

    return


start_date = datetime(2020, 6, 23)
end_date = datetime(2020, 6, 30) - timedelta(hours=1)
point_location = 'wrf_validation_lite_points_v2.csv'
#
# buoy = ['NYSE06', b'NYSE06']
# plot_val_time_series(start_date, end_date, buoy, point_location, height=[80])
# plot_val_time_series(start_date, end_date, buoy, point_location, height=[160])
# buoy = ['NYSWE05', b'NYSWE05']
# plot_val_time_series(start_date, end_date, buoy, point_location, height=[80])
# plot_val_time_series(start_date, end_date, buoy, point_location, height=[160])
# # buoy = ['NYNE05', b'NYNE05']
# # plot_val_time_series(start_date, end_date, buoy, point_location, height=[80])
# # plot_val_time_series(start_date, end_date, buoy, point_location, height=[160])
# buoy = ['SODAR', b'SODAR']
# plot_val_time_series(start_date, end_date, buoy, point_location, height=[80])
# plot_val_time_series(start_date, end_date, buoy, point_location, height=[160])
buoy = ['ASOSB6', b'ASOSB6']
plot_val_time_series(start_date, end_date, buoy, point_location, height=[80])
plot_val_time_series(start_date, end_date, buoy, point_location, height=[160])
