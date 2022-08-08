import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from erddapy import ERDDAP
import glob
import os


# WRF Functions
def max_min_finder(dataset1, dataset2, dataset3):
    max_ds = []
    min_ds = []
    nds = dataset1, dataset2, dataset3
    for ii in range(0, len(dataset1)):
        max_ds = np.append(max_ds, max([x[ii] for x in nds]))
        min_ds = np.append(min_ds, min([x[ii] for x in nds]))

    return max_ds, min_ds


def load_buoy(start_date, end_date, buoy, recent):
    start_date = start_date - timedelta(hours=1)
    end_date = end_date - timedelta(hours=1)
    rng = pd.date_range(start_date, end_date - timedelta(hours=1), freq="H")
    rng = rng + timedelta(minutes=50)
    if recent == 1:
        buoy_dir = 'http://dods.ndbc.noaa.gov//thredds/dodsC/data/stdmet/{}/{}h9999.nc'.format(buoy, buoy)
        nd = xr.open_dataset(buoy_dir)
        buoy_ws = nd.wind_spd.sel(time=slice(start_date, end_date))[:, 0, 0]
        buoy_ws = buoy_ws.reindex({'time': rng})
        time = nd.time.sel(time=slice(start_date, end_date)).data
    else:
        if abs(end_date.year-start_date.year) == 0:
            buoy_dir = 'http://dods.ndbc.noaa.gov//thredds/dodsC/data/stdmet/{}/{}h{}.nc'.format(buoy, buoy, start_date.year)
            nd = xr.open_dataset(buoy_dir)
            buoy_ws = nd.wind_spd.sel(time=slice(start_date, end_date))[:, 0, 0]
            buoy_ws = buoy_ws.reindex({'time': rng})
            time = nd.time.sel(time=slice(start_date, end_date)).data
        elif abs(end_date.year-start_date.year) == 1:
            buoy_dir = 'http://dods.ndbc.noaa.gov//thredds/dodsC/data/stdmet/{}/{}h{}.nc'.format(buoy, buoy, start_date.year)
            nd = xr.open_dataset(buoy_dir)
            buoy_dir2 = 'http://dods.ndbc.noaa.gov//thredds/dodsC/data/stdmet/{}/{}h{}.nc'.format(buoy, buoy, end_date.year)
            nd2 = xr.open_dataset(buoy_dir2)
            bdsw = xr.concat([nd.wind_spd, nd2.wind_spd], dim='time')
            bdst = xr.concat([nd.time, nd2.time], dim='time')
            buoy_ws = bdsw.sel(time=slice(start_date, end_date))[:, 0, 0]
            buoy_ws = buoy_ws.reindex({'time': rng})
            time = bdst.sel(time=slice(start_date, end_date)).data

        return buoy_ws, time, buoy_dir
    return buoy_ws, time, buoy_dir


def make_wrf_file(dtime, fo=0, pd=0):
    t2 = dtime.replace()  # Copy variable to mess with
    if pd:
        t2 = t2-timedelta(pd)  # Previous Day
    if t2.hour < fo:
        t2 = t2-timedelta(1)  # Previous model run
        hour = t2.hour + 24
    else:
        hour = t2.hour
    if pd:
        hour = hour+24*pd  # Later in model run
    datestr = '%d%02d%02d' % (t2.year, t2.month, t2.day)
    return '%s/wrfproc_3km_%s_00Z_H%03d.nc' % (datestr, datestr, hour)


def load_wrf(start_date, end_date, forecast_offset, version_num, point_location, buoy, height):
    directory = '/home/coolgroup/ru-wrf/real-time/v4.1_parallel/processed/3km/'  # server
    if version_num == 'v4.1':
        directory = '/home/coolgroup/ru-wrf/real-time/v4.1_parallel/processed/3km/'
    elif version_num == 'v3.9':
        directory = '/home/coolgroup/ru-wrf/real-time/processed/3km/'
    else:
        print('Wrong Version Number')

    # end_date = end_date - timedelta(hours=1)
    times = pd.date_range(start_date, end_date, freq="H")
    heights = np.append(np.array([10], dtype='int32'), height)
    # heights = np.array([10, 80, 120, 140, 160], dtype='int32')
    sites = pd.read_csv(point_location, skipinitialspace=True)
    sites = sites[sites['name'] == buoy[0]]
    stations = sites.name.astype('S')
    data = np.empty(shape=(len(times), len(stations), len(heights))) * np.NAN
    uVel = xr.DataArray(data, coords=[times, stations, heights], dims=['time', 'station', 'height'], attrs={
        'units': 'm s-1',
        'standard_name': 'eastward_wind',
        'long_name': 'Wind Speed, Zonal',
        'comment': 'The zonal wind speed (m/s) indicates the u (positive eastward) component'
                   ' of where the wind is going.',
    })

    uVel['time'].attrs['standard_name'] = 'time'
    uVel['time'].attrs['long_name'] = 'Time'

    uVel['station'].attrs['standard_name'] = 'station_id'
    uVel['station'].attrs['long_name'] = 'Station ID'
    uVel['station'].attrs['comment'] = 'A string specifying a unique station ID, created to allow easy referencing of' \
                                       ' the selected grid points extracted from the WRF model files.'

    uVel['height'].attrs['units'] = 'm'
    uVel['height'].attrs['standard_name'] = 'height'
    uVel['height'].attrs['long_name'] = 'Height'

    vVel = uVel.copy()
    vVel.attrs['standard_name'] = 'northward_wind'
    vVel.attrs['long_name'] = 'Wind Speed, Meridional'
    vVel.attrs['comment'] = 'The meridional wind speed (m/s) indicates the v (positive northward) component of ' \
                            'where the wind is going.'

    latitude = xr.DataArray(sites['latitude'], coords=[stations], dims=['station'], attrs={
        'units': 'degrees_north',
        'comment': 'The latitude of the station.',
        'long_name': 'Latitude',
        'standard_name': 'latitude'
    })
    longitude = xr.DataArray(sites['longitude'], coords=[stations], dims=['station'], attrs={
        'units': 'degrees_east',
        'comment': 'The longitude of the station.',
        'long_name': 'Longitude',
        'standard_name': 'longitude'
    })
    for t in times:
        try:
            wrf_file = make_wrf_file(t, forecast_offset)
            ncdata = xr.open_dataset(directory + wrf_file)
            print(directory + wrf_file)
            lats = ncdata.XLAT.squeeze()
            lons = ncdata.XLONG.squeeze()

            for index, site in sites.iterrows():
                # Step 4 - Find the closest model point
                a = abs(lats - site.latitude) + abs(lons - site.longitude)
                i, j = np.unravel_index(a.argmin(), a.shape)
                uVel.loc[{'time': t, 'station': stations[index], 'height': 10}] = ncdata.U10[0][i][j].item()
                vVel.loc[{'time': t, 'station': stations[index], 'height': 10}] = ncdata.V10[0][i][j].item()

                levels = pd.Series(np.arange(40, 260, 10), index=np.arange(1, 23))
                hindex = []

                for qq in height:
                    hindex.append(levels[levels == qq].index[0])

                for ii, jj in zip(height, hindex):
                    uVel.loc[{'time': t, 'station': stations[index], 'height': ii}] = ncdata.U[0][jj][i][j].item()
                    vVel.loc[{'time': t, 'station': stations[index], 'height': ii}] = ncdata.V[0][jj][i][j].item()
            ncdata.close()

        except:
            print('Could not process file: ' + wrf_file)

    # Wind Speed
    wind_speed = np.sqrt(uVel ** 2 + vVel ** 2)
    wind_speed.attrs['units'] = 'm s-1'
    wind_speed.attrs['comment'] = 'Wind Speed is calculated from the Zonal and Meridional wind speeds.'
    wind_speed.attrs['long_name'] = 'Wind Speed'
    wind_speed.attrs['standard_name'] = 'wind_speed'

    # Wind Direction
    wind_dir = 270 - np.arctan2(vVel, uVel) * 180 / np.pi
    wind_dir = wind_dir % 360  # Use modulo to keep degrees between 0-360
    wind_dir.attrs['units'] = 'degree'
    wind_dir.attrs['comment'] = 'The direction from which winds are coming from, in degrees clockwise from true N.'
    wind_dir.attrs['long_name'] = 'Wind Direction'
    wind_dir.attrs['standard_name'] = 'wind_from_direction'

    final_dataset = xr.Dataset({
        'u_velocity': uVel, 'v_velocity': vVel,
        'wind_speed': wind_speed, 'wind_dir': wind_dir,
        'latitude': latitude, 'longitude': longitude
    })

    encoding = {}
    encoding['time'] = dict(units='days since 2010-01-01 00:00:00', calendar='gregorian', dtype=np.double)

    return final_dataset


def load_nam(start_date, end_date, buoy, point_location, height):
    sites = pd.read_csv(point_location, skipinitialspace=True)
    time_span_D = pd.date_range(start_date, end_date-timedelta(hours=22), freq='D')  # -timedelta(days=1) was removed from here
    directory = '/home/jad438/validation_data/namdata/'
    nam_ws = []
    nam_wd = []
    nam_dt = np.empty((0,), dtype='datetime64[m]')

    if height >= [80]:
        height = 80
        print('Height of 80 used for NAM')
    elif height < [80]:
        height = 10
        print('Height of 10 used for NAM')
    else:
        print('NAM Height selection fail')

    height_i = pd.Series([10, 80], index=[0, 1])
    height_i = height_i[height_i == height].index[0]

    for ind, date in enumerate(time_span_D):
        file = 'nams_data_' + date.strftime("%Y%m%d") + '.nc'
        try:
            nam_ds = xr.open_dataset(directory + file)
            lats = nam_ds.gridlat_0.squeeze()
            lons = nam_ds.gridlon_0.squeeze()
            site_code = sites[sites['name'] == buoy].index[0]
            a = abs(lats - sites.latitude[site_code]) + abs(lons - sites.longitude[site_code])
            i, j = np.unravel_index(a.argmin(), a.shape)
            nam_ws = np.append(nam_ws, nam_ds.wind_speed[:, height_i, i, j])
            # nam_wd = np.append(nam_wd, nam_ds.wind_from_direction[:, height, i, j])
            nam_dt = np.append(nam_dt, nam_ds.time)
        except:
            time_span_H = pd.date_range(date, date + timedelta(hours=23), freq='H')
            nam_ws = np.append(nam_ws, np.empty(shape=(len(time_span_H))) * np.NAN)
            # nam_wd = np.append(nam_wd, np.empty(shape=(len(time_span_H))) * np.NAN)
            nam_dt = np.append(nam_dt, time_span_H)

    return nam_ws, nam_dt


def load_gfs(start_date, end_date, buoy, point_location, height):
    sites = pd.read_csv(point_location, skipinitialspace=True)
    time_span_D = pd.date_range(start_date, end_date-timedelta(hours=23), freq='D')  # -timedelta(days=1) was removed from here
    directory = '/home/jad438/validation_data/gfsdata/'
    gfs_ws = []
    gfs_dt = np.empty((0,), dtype='datetime64[m]')

    if height >= [100]:
        height = 100
        print('Height of 100 used for GFS')
    elif height == [80]:
        height = 80
        print('Height of 80 used for GFS')
    elif height == [50]:
        height = 50
        print('Height of 50 used for GFS')
    elif height == [40]:
        height = 40
        print('Height of 40 used for GFS')
    elif height == [30]:
        height = 30
        print('Height of 30 used for GFS')
    elif height <= [10]:
        height = 10
        print('Height of 10 used for GFS')
    else:
        print('GFS Height selection fail')

    height_i = pd.Series([10, 30, 40, 50, 80, 100], index=[0, 1, 2, 3, 4, 5])
    height_i = height_i[height_i == height].index[0]

    for ind, date in enumerate(time_span_D):
        file = 'gfs_data_' + date.strftime("%Y%m%d") + '.nc'
        try:
            gfs_ds = xr.open_dataset(directory + file)
            lats = gfs_ds.lat_0.squeeze()
            lons = gfs_ds.lon_0.squeeze()-360
            site_code = sites[sites['name'] == buoy].index[0]
            a = abs(lats - sites.latitude[site_code]) + abs(lons - sites.longitude[site_code])
            i, j = np.unravel_index(a.argmin(), a.shape)
            gfs_ws = np.append(gfs_ws, gfs_ds.wind_speed[:, height_i, i, j])
            gfs_dt = np.append(gfs_dt, gfs_ds.time)

        except:
            time_span_H = pd.date_range(date, date + timedelta(hours=23), freq='H')
            gfs_ws = np.append(gfs_ws, np.empty(shape=(len(time_span_H))) * np.NAN)
            gfs_dt = np.append(gfs_dt, time_span_H)

    return gfs_ws, gfs_dt


def load_hrrr(start_date, end_date, buoy, point_location, height):
    sites = pd.read_csv(point_location, skipinitialspace=True)
    time_span_D = pd.date_range(start_date, end_date-timedelta(hours=23), freq='D')  #  -timedelta(days=1) was removed from here
    directory = '/home/jad438/validation_data/hrrrdata/'
    hrrr_ws = []
    hrrr_dt = np.empty((0,), dtype='datetime64[m]')

    if height >= [80]:
        height = 80
        print('Height of 80 used for HRRR')
    elif height < [80]:
        height = 10
        print('Height of 10 used for HRRR')
    else:
        print('HRRR Height selection fail')

    height_i = pd.Series([10, 80], index=[0, 1])
    height_i = height_i[height_i == height].index[0]

    for ind, date in enumerate(time_span_D):
        file = 'hrrr_data_' + date.strftime("%Y%m%d") + '.nc'
        try:
            hrrr_ds = xr.open_dataset(directory + file)
            lats = hrrr_ds.gridlat_0.squeeze()
            lons = hrrr_ds.gridlon_0.squeeze()
            site_code = sites[sites['name'] == buoy].index[0]
            a = abs(lats - sites.latitude[site_code]) + abs(lons - sites.longitude[site_code])
            i, j = np.unravel_index(a.argmin(), a.shape)
            hrrr_ws = np.append(hrrr_ws, hrrr_ds.wind_speed[:, height_i, i, j])
            hrrr_dt = np.append(hrrr_dt, hrrr_ds.time)

        except:
            time_span_H = pd.date_range(date, date + timedelta(hours=23), freq='H')
            hrrr_ws = np.append(hrrr_ws, np.empty(shape=(len(time_span_H))) * np.NAN)
            hrrr_dt = np.append(hrrr_dt, time_span_H)

    return hrrr_ws, hrrr_dt


def sodar_loader(start_date, end_date, height):
    sodar_dir = '/home/coolgroup/MetData/CMOMS/sodar/daily/'

    all_files_sodar = glob.glob(os.path.join(sodar_dir, 'DAT.data.ftp.wxflow.sodar.202*'))
    all_files_sodar.sort()
    r = pd.date_range(start=start_date, end=end_date, freq='H')

    shl = [str(i) for i in height]
    height_s = [s + 'm' for s in shl]

    sodar_ds1 = []
    for f in all_files_sodar:
        try:
            ds = pd.read_csv(f, parse_dates=['Date and Time'])
            sodar_ds1.append(ds)
            print(f)
        except:
            print('failed at ' + f)

    sodar_ds2 = pd.concat(sodar_ds1)
    sodar_ds3 = sodar_ds2.set_index('Date and Time').reindex(r).fillna(np.nan).rename_axis('dt').reset_index()
    sodar_ds4 = pd.DataFrame()

    for wh in height_s:
        cols = [x for x in sodar_ds3.columns.to_list() if wh == x.split(' ')[0]]
        cols.append('dt')
        dfc = sodar_ds3[cols]
        qc_colname = '{} Quality'.format(wh)
        dfc = dfc[dfc[qc_colname] >= 60]
        dfc = dfc.drop(columns=qc_colname)
        dfc = dfc.melt(id_vars='dt')
        sodar_ds4 = sodar_ds4.append(dfc)

    sodar_ds4['height'] = sodar_ds4['variable'].map(lambda x: int(x.split(' ')[0][:-1]))

    df_ws = sodar_ds4[sodar_ds4['variable'].str.contains('Wind Speed')]
    df_wd = sodar_ds4[sodar_ds4['variable'].str.contains('Wind Direction')]
    df_dt = pd.to_datetime(np.array(df_ws['dt']))

    return df_ws, df_wd, df_dt


def load_nyserda_temp(buoy, start_date, end_date):
    if buoy[0] == 'NYNE05':
        filename10 = 'E05_Hudson_North_10_min_avg.csv'
        filenamehr = 'E05_Hudson_North_hourly_avg.csv'
    elif buoy[0] == 'NYSE06':
        filename10 = 'E06_Hudson_South_10_min_avg.csv'
        filenamehr = 'E06_Hudson_South_hourly_avg.csv'
    else:
        print('Not a correct buoy')

    dir = '/Users/JadenD/PycharmProjects/covid19/data/nyserda/'
    nys_ds_10 = pd.read_csv(dir + filename10, error_bad_lines=False, delimiter=', ')  # , delim_whitespace=True)
    nys_ds_hr = pd.read_csv(dir + filenamehr, error_bad_lines=False, delimiter=', ')  # , delim_whitespace=True)

    nys_ws_1hr_nonav_10 = nys_ds_10  # data every 6 steps (1 hour)
    nys_ws_1hr_nonav_hr = nys_ds_hr  # data every 6 steps (1 hour)
    nys_ws_1hr_nonav_10 = nys_ws_1hr_nonav_10.reset_index()
    nys_ws_1hr_nonav_hr = nys_ws_1hr_nonav_hr.reset_index()

    time = pd.date_range(start_date, end_date, freq='H')

    nys_ws_1hr_nonav_10['timestamp'] = pd.to_datetime(nys_ws_1hr_nonav_10['timestamp'], format='%m-%d-%Y %H:%M')
    nys_ws_1hr_nonav_hr['timestamp'] = pd.to_datetime(nys_ws_1hr_nonav_hr['timestamp'], format='%m-%d-%Y %H:%M')
    nys_ws_1hr_nonav_10['meteo_Ta_avg'] = pd.to_numeric(nys_ws_1hr_nonav_10['meteo_Ta_avg']
                                                                    , errors='coerce')
    nys_ws_1hr_nonav_hr['ADCP_ADCPtemp'] = pd.to_numeric(nys_ws_1hr_nonav_hr['ADCP_ADCPtemp']
                                                                    , errors='coerce')

    nys_ws_1hr_nonav_10 = pd.Series(nys_ws_1hr_nonav_10['meteo_Ta_avg'].values,
                                 index=nys_ws_1hr_nonav_10['timestamp'])
    nys_ws_1hr_nonav_10 = nys_ws_1hr_nonav_10.reindex(time)

    nys_ws_1hr_nonav_hr = pd.Series(nys_ws_1hr_nonav_hr['ADCP_ADCPtemp'].values,
                                 index=nys_ws_1hr_nonav_hr['timestamp'])
    nys_ws_1hr_nonav_hr = nys_ws_1hr_nonav_hr.reindex(time)

    return nys_ws_1hr_nonav_10, nys_ws_1hr_nonav_hr


def load_nyserda(buoy, start_date, end_date):
    if buoy[0] == 'NYNE05':
        filename = 'E05_Hudson_North_10_min_avg.csv'
    elif buoy[0] == 'NYSE06':
        filename = 'E06_Hudson_South_10_min_avg.csv'
    elif buoy[0] == 'NYSWE05':
        filename = 'E05_Hudson_South_West_10_min_avg.csv'
    else:
        print('Not a correct buoy')

    dir = '/Users/JadenD/PycharmProjects/wrf_validation/data/nyserda/'
    nys_ds = pd.read_csv(dir + filename, error_bad_lines=False, delimiter=', ')  # , delim_whitespace=True)

    nys_ws_1hr_nonav = nys_ds  # data every 6 steps (1 hour)
    nys_ws_1hr_nonav = nys_ws_1hr_nonav.reset_index()

    time = pd.date_range(start_date, end_date, freq='H')

    nys_ws_1hr_nonav['timestamp'] = pd.to_datetime(nys_ws_1hr_nonav['timestamp'], format='%m-%d-%Y %H:%M')
    nys_ws_1hr_nonav['lidar_lidar158m_Z10_HorizWS'] = pd.to_numeric(nys_ws_1hr_nonav['lidar_lidar158m_Z10_HorizWS']
                                                                    , errors='coerce')

    nys_ws_1hr_nonav = pd.Series(nys_ws_1hr_nonav['lidar_lidar158m_Z10_HorizWS'].values,
                                 index=nys_ws_1hr_nonav['timestamp'])
    nys_ws_1hr_nonav = nys_ws_1hr_nonav.reindex(time)

    return nys_ws_1hr_nonav


def load_nyserda_ws(buoy, height, start_date, end_date):
    if buoy[0] == 'NYNE05':
        filename = 'E05_Hudson_North_10_min_avg.csv'
    elif buoy[0] == 'NYSE06':
        filename = 'E06_Hudson_South_10_min_avg.csv'
    elif buoy[0] == 'NYSWE05':
        filename = 'E05_Hudson_South_West_10_min_avg.csv'
    else:
        print('Not a correct buoy')

    # correction for buoy height
    height = height - 2

    directory = '/home/coolgroup/bpu/wrf/data/validation_data/nyserda'
    nys_ds = pd.read_csv(os.path.join(directory, filename), error_bad_lines=False, delimiter=',', engine='python')  # , delim_whitespace=True)

    # remove whitespace from column headers
    nys_ds.columns = nys_ds.columns.str.strip()

    time = pd.date_range(start_date, end_date, freq='H')

    nys_ds['timestamp'] = nys_ds['timestamp'].map(lambda t: pd.to_datetime(t))  # fix timestamp formatting
    colname = f'lidar_lidar{height}m_Z10_HorizWS'
    nys_ds[colname] = pd.to_numeric(nys_ds[colname], errors='coerce')

    df = pd.Series(nys_ds[colname].values, index=nys_ds['timestamp'])
    df = df.reindex(time)

    return df


def load_ASOSB_ws_heights(start_date, end_date):
    as_file = '/home/coolgroup/bpu/wrf/data/validation_data/atlantic_shores_buoy/AtlanticShores_ASOW-6.nc'
    as_ds = xr.open_dataset(as_file)

    as_ds.time.data = pd.to_datetime(as_ds.time.data, format='%m-%d-%Y %H:%M')
    time_h = pd.date_range(start_date, end_date - timedelta(hours=1), freq='H')
    time_m = pd.date_range(start_date, end_date - timedelta(hours=1), freq='10min')

    as_ds.wind_speed[as_ds.wind_speed > 55] = np.nan
    as_ds.wind_speed[as_ds.wind_speed < 0] = np.nan

    as_ws = pd.Series(as_ds.wind_speed.values, index=as_ds.time.data)
    as_ws = as_ws.reindex(time_h)

    as_ws_m = pd.Series(as_ds.wind_speed.values, index=as_ds.time.data)
    as_ws_m = as_ws_m.reindex(time_m)

    # as_ds.time.data = pd.to_numeric(as_ds.time.data, errors='coerce')
    as_ws_av = []
    for i in range(0, len(as_ws_m), 6):
        as_ws_av.append(np.mean(as_ws_m[i:i + 6]))

    asosb_ws_1hr_avg = pd.Series(as_ws_av, index=as_ws.index)

    return asosb_ws_1hr_avg


def load_ASOSB(start_date, end_date, ASbuoy, height):
    url_erddap = 'https://erddap.maracoos.org/erddap/'
    dataset_id = 'AtlanticShores_ASOW-' + str(ASbuoy) + '_wind'
    start = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    stop = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')

    variables = [
                 'wind_speed',
                 'altitude',
                 'wind_from_direction',
                 'time'
                 ]

    constraints = {
                   'time>=': start,
                   'time<=': stop,
                   'altitude=': height[0],
                   # 'altitude>=': 0,
                   # 'altitude<=': 260,
    }

    e = ERDDAP(server=url_erddap,
               protocol='tabledap',
               response='nc'
               )

    e.dataset_id = dataset_id
    e.constraints = constraints
    e.variables = variables

    df = e.to_pandas(index_col="time (UTC)",
                     parse_dates=True,
                     )

    df.reset_index(inplace=True)

    df.rename(columns={'time (UTC)': 'time',
                       'wind_speed (m/s)': 'wind_speed',
                       'altitude (m)': 'altitude',
                       'wind_from_direction (degree)': 'wind_from_direction'
                       },
              inplace=True
              )

    df['time'] = df['time'].dt.tz_localize(None)
    df.set_index('time', inplace=True)

    ds = df.to_xarray()

    return ds


def load_ASOSB_ws_heights_dir(start_date, end_date, as_dir):
    as_ds = xr.open_dataset(as_dir)

    as_ds.time.data = pd.to_datetime(as_ds.time.data, format='%m-%d-%Y %H:%M')
    time_h = pd.date_range(start_date, end_date - timedelta(hours=1), freq='H')
    time_m = pd.date_range(start_date, end_date - timedelta(hours=1), freq='10min')

    as_ds.wind_speed[as_ds.wind_speed > 55] = np.nan
    as_ds.wind_speed[as_ds.wind_speed < 0] = np.nan

    as_ws = pd.Series(as_ds.wind_speed.values, index=as_ds.time.data)
    as_ws = as_ws.reindex(time_h)

    as_ws_m = pd.Series(as_ds.wind_speed.values, index=as_ds.time.data)
    as_ws_m = as_ws_m.reindex(time_m)

    # as_ds.time.data = pd.to_numeric(as_ds.time.data, errors='coerce')
    as_ws_av = []
    for i in range(0, len(as_ws_m), 6):
        as_ws_av.append(np.mean(as_ws_m[i:i + 6]))

    asosb_ws_1hr_avg = pd.Series(as_ws_av, index=as_ws.index)

    return asosb_ws_1hr_avg


def load_ASOSB_swt(start_date, end_date):
    as_dir = '/Users/JadenD/PycharmProjects/covid19/data/atlantic_shores/AtlanticShores_20200228_20210228_sst.nc'
    as_ds = xr.open_dataset(as_dir)

    as_ds.time.data = pd.to_datetime(as_ds.time.data, format='%m-%d-%Y %H:%M')
    time_h = pd.date_range(start_date, end_date - timedelta(hours=1), freq='H')
    time_m = pd.date_range(start_date, end_date - timedelta(hours=1), freq='10min')

    # as_ds.wind_speed[as_ds.wind_speed > 55] = np.nan
    # as_ds.wind_speed[as_ds.wind_speed < 0] = np.nan

    as_at = pd.Series(as_ds.air_temperature.values, index=as_ds.time.data)
    as_at = as_at.reindex(time_h)

    as_swt_1m = pd.Series(as_ds.sea_water_temperature_at_1m.values, index=as_ds.time.data)
    as_swt_1m = as_swt_1m.reindex(time_h)

    as_swt_2m = pd.Series(as_ds.sea_water_temperature_at_2m.values, index=as_ds.time.data)
    as_swt_2m = as_swt_2m.reindex(time_h)

    as_swt_32m = pd.Series(as_ds.sea_water_temperature_at_32m.values, index=as_ds.time.data)
    as_swt_32m = as_swt_32m.reindex(time_h)

    d = {'air_temp': as_at, 'swt_1m': as_swt_1m, 'swt_2m': as_swt_2m, 'swt_32m': as_swt_32m}

    df = pd.DataFrame(d)

    # as_ws_av = []
    # for i in range(0, len(as_ws_m), 6):
    #     as_ws_av.append(np.mean(as_ws_m[i:i + 6]))
    #
    # asosb_ws_1hr_avg = pd.Series(as_ws_av, index=as_at.index)

    return df


def load_nyserda_rework(buoy, end_date):
    if buoy[0] == 'NYNE05':
        buoy_end_date = end_date-timedelta(days=1)
        # filename = 'E05_Hudson_North_10_min_avg_20190812_' + buoy_end_date.strftime('%Y%m%d') + '.csv'
        filename = 'E05_Hudson_North_10_min_avg_20190812_20200927.csv'
        start_date = datetime(2019, 8, 12)
    elif buoy[0] == 'NYSE06':
        buoy_end_date = end_date-timedelta(days=1)
        # filename = 'E06_Hudson_South_10_min_avg_20190904_' + buoy_end_date.strftime('%Y%m%d') + '.csv'
        filename = 'E06_Hudson_South_10_min_avg_20190904_20200927.csv'
        start_date = datetime(2019, 9, 4)
    else:
        print('Not a correct buoy')

    # dir = '/Users/jadendicopoulos/Downloads/NYSERDA Floating LiDAR Buoy Data/'
    # dir = '/Users/jadendicopoulos/PycharmProjects/rucool/data/nyserda/'
    dir = '/Users/JadenD/PycharmProjects/covid19/data/nyserda/'
    nys_ds = pd.read_csv(dir + filename, error_bad_lines=False, delimiter=', ')  # , delim_whitespace=True)

    nys_ws_1hr_nonav = nys_ds  # data every 6 steps (1 hour)
    nys_ws_1hr_nonav = nys_ws_1hr_nonav.reset_index()

    nys_ws = pd.to_numeric(nys_ds['lidar_lidar158m_Z10_HorizWS'], errors='coerce')

    time = pd.date_range(start_date, end_date-timedelta(hours=1), freq='H')

    nys_ws_1hr_nonav['timestamp'] = pd.to_datetime(nys_ws_1hr_nonav['timestamp'], format='%m-%d-%Y %H:%M')
    nys_ws_1hr_nonav['lidar_lidar158m_Z10_HorizWS'] = pd.to_numeric(nys_ws_1hr_nonav['lidar_lidar158m_Z10_HorizWS']
                                                                    , errors='coerce')

    nys_ws_1hr_nonav = pd.Series(nys_ws_1hr_nonav['lidar_lidar158m_Z10_HorizWS'].values,
                                 index=nys_ws_1hr_nonav['timestamp'])
    nys_ws_1hr_nonav = nys_ws_1hr_nonav.reindex(time)

    return nys_ws_1hr_nonav


def load_nyserda_ws_heights(buoy, height, start_date, end_date):
    # nys_heights = [[18], [38], [58], [78], [98], [118], [138], [158], [178], [198]]
    if buoy[0] == 'NYNE05':
        filename = 'E05_Hudson_North_10_min_avg.csv'
    elif buoy[0] == 'NYSE06':
        filename = 'E06_Hudson_South_10_min_avg.csv'
    else:
        print('Not a correct buoy')

    # dir = '/Users/jadendicopoulos/PycharmProjects/rucool/data/nyserda/'
    dir = '/Users/JadenD/PycharmProjects/covid19/data/nyserda/'
    nys_ds = pd.read_csv(dir + filename, error_bad_lines=False, delimiter=', ', engine='python')  # , delim_whitespace=True)

    nys_ws_1hr_nonav = nys_ds  # data every 6 steps (1 hour)
    nys_ws_1hr_nonav = nys_ws_1hr_nonav.reset_index()

    time = pd.date_range(start_date, end_date-timedelta(hours=1), freq='H')

    nys_ws_1hr_nonav['timestamp'] = pd.to_datetime(nys_ws_1hr_nonav['timestamp'], format='%m-%d-%Y %H:%M')
    nys_ws_1hr_nonav['lidar_lidar' + str(height) + 'm_Z10_HorizWS'] = pd.to_numeric(nys_ws_1hr_nonav['lidar_lidar' + str(height) + 'm_Z10_HorizWS']
                                                                    , errors='coerce')

    nys_ws_1hr_nonav = pd.Series(nys_ws_1hr_nonav['lidar_lidar' + str(height) + 'm_Z10_HorizWS'].values,
                                 index=nys_ws_1hr_nonav['timestamp'])
    nys_ws_1hr_nonav = nys_ws_1hr_nonav.reindex(time)

    return nys_ws_1hr_nonav


def load_nyserda_ws_cup(buoy, start_date, end_date):
    # nys_heights = [[18], [38], [58], [78], [98], [118], [138], [158], [178], [198]]
    if buoy[0] == 'NYNE05':
        filename = 'E05_Hudson_North_10_min_avg.csv'
    elif buoy[0] == 'NYSE06':
        filename = 'E06_Hudson_South_10_min_avg.csv'
    else:
        print('Not a correct buoy')

    # dir = '/Users/jadendicopoulos/PycharmProjects/rucool/data/nyserda/'
    dir = '/Users/JadenD/PycharmProjects/covid19/data/nyserda/'
    nys_ds = pd.read_csv(dir + filename, error_bad_lines=False, delimiter=', ', engine='python')  # , delim_whitespace=True)

    nys_ws_1hr_nonav = nys_ds  # data every 6 steps (1 hour)
    nys_ws_1hr_nonav = nys_ws_1hr_nonav.reset_index()

    time = pd.date_range(start_date, end_date-timedelta(hours=1), freq='H')

    nys_ws_1hr_nonav['timestamp'] = pd.to_datetime(nys_ws_1hr_nonav['timestamp'], format='%m-%d-%Y %H:%M')
    nys_ws_1hr_nonav['meteo_Sm_avg'] = pd.to_numeric(nys_ws_1hr_nonav['meteo_Sm_avg'], errors='coerce')

    nys_ws_1hr_nonav = pd.Series(nys_ws_1hr_nonav['meteo_Sm_avg'].values, index=nys_ws_1hr_nonav['timestamp'])
    nys_ws_1hr_nonav = nys_ws_1hr_nonav.reindex(time)

    return nys_ws_1hr_nonav


def load_nyserda_wd_heights(buoy, height, start_date, end_date):
    # nys_heights = [[18], [38], [58], [78], [98], [118], [138], [158], [178], [198]]
    if buoy[0] == 'NYNE05':
        filename = 'E05_Hudson_North_10_min_avg.csv'
    elif buoy[0] == 'NYSE06':
        filename = 'E06_Hudson_South_10_min_avg.csv'
    else:
        print('Not a correct buoy')

    # dir = '/Users/jadendicopoulos/PycharmProjects/rucool/data/nyserda/'
    dir = '/Users/JadenD/PycharmProjects/covid19/data/nyserda/'
    nys_ds = pd.read_csv(dir + filename, error_bad_lines=False, delimiter=', ')  # , delim_whitespace=True)

    nys_wd_1hr = nys_ds  # data every 6 steps (1 hour)
    nys_wd_1hr = nys_wd_1hr.reset_index()

    time = pd.date_range(start_date, end_date-timedelta(hours=1), freq='H')

    nys_wd_1hr['timestamp'] = pd.to_datetime(nys_wd_1hr['timestamp'], format='%m-%d-%Y %H:%M')
    nys_wd_1hr['lidar_lidar' + str(height) + 'm_WD_alg_03'] = pd.to_numeric(nys_wd_1hr['lidar_lidar' + str(height) + 'm_WD_alg_03']
                                                                    , errors='coerce')

    nys_wd_1hr_nonav = pd.Series(nys_wd_1hr['lidar_lidar' + str(height) + 'm_WD_alg_03'].values,
                                 index=nys_wd_1hr['timestamp'])
    nys_wd_1hr_nonav = nys_wd_1hr_nonav.reindex(time)

    nys_wd_1hr_av = pd.Series(nys_wd_1hr['lidar_lidar' + str(height) + 'm_WD_alg_03'].values,
                                 index=nys_wd_1hr['timestamp'])
    nys_wd_1hr_av = nys_wd_1hr_av.rolling(window=6).mean()
    # nys_wd_1hr_av = np.mean(nys_wd_1hr_av.reshape(-1, 6), axis=1)
    nys_wd_1hr_av = nys_wd_1hr_av.reindex(time)

    return nys_wd_1hr_nonav, nys_wd_1hr_av


# Met Tower and SoDAR functions
def make_met_file(dtime):
    t2 = dtime.replace()
    datestr = '%d%02d%02d' % (t2.year, t2.month, t2.day)
    return 'DAT.data.ftp.wxflow.surface.%s.dat' % datestr, datestr


def met_tower_checker(start_date, end_date):
    times = pd.date_range(start_date, end_date, freq="D")
    dir = '/Volumes/home/coolgroup/MetData/CMOMS/surface/daily/'
    # filename = 'DAT.data.ftp.wxflow.surface.20190324.dat'
    for t in times:
        filename, datestr = make_met_file(t)
        try:
            met_ds = pd.read_csv(dir + filename, error_bad_lines=False)
            f = open("MetTower_checker_output.txt", "a")

            if met_ds.any().all() == True:
                pass
            elif met_ds.any().all() == False:
                print('Checker Tripped on: ' + datestr, file=f)
                print(met_ds.any(), file=f)
                print('\n', file=f)

            f.close()

        except:
            f = open("MetTower_checker_output.txt", "a")
            print('Missing File on: ' + datestr, file=f)
            print('\n', file=f)


# All purpose

def pol2cart(rho, phi):
    u = rho * np.cos(phi*(np.pi/180))
    v = rho * np.sin(phi*(np.pi/180))
    return u, v


def weib(x, n, a):
    return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)


def wind_shear_exp(v1, v2, z2, z1):
    # z1 is the lower height and v1 is the corresponding windspeed at that height
    return np.log(v2/v1)/np.log(z2/z1)


def metrics(observations, forecast):
    ind = np.where((~np.isnan(observations) & ~np.isnan(forecast)))

    obs_mean = np.mean(observations[ind])
    for_mean = np.mean(forecast[ind])
    obs_std = np.std(observations[ind])
    for_std = np.std(forecast[ind])

    rms = np.sqrt(np.mean((forecast[ind]-observations[ind])**2))
    crms = np.sqrt(np.mean(((forecast[ind]-for_mean) - (observations[ind]-obs_mean))**2))
    mb = for_mean - obs_mean
    count = ind[:][0].__len__()

    metrics_frame = [rms, crms, mb, count, for_std, for_mean, obs_std, obs_mean]

    return metrics_frame


# start_date = datetime(2022, 2, 8)
# end_date = datetime(2022, 2, 10)
# buoy = 'NYSE06', b'NYSE06'
# point_location = 'wrf_validation_lite_points_v2.csv'
# height = [160]
# ds = load_wrf(start_date, end_date,
#               forecast_offset=1, version_num='v4.1',
#               point_location=point_location, buoy=buoy,
#               height=[160])

# ds = load_ASOSB(start_date, end_date, ASbuoy=4, height=[160])
# ds = load_nam(start_date, end_date, buoy, point_location, height)
# print(ds)

