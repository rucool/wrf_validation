import xarray as xr
import numpy as np
import dask
import datetime as dt
from scipy import stats
from scipy import interpolate
from erddapy import ERDDAP
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def wind_shear_exp(v1, v2, z2, z1):
    # z1 is the lower height and v1 is the corresponding windspeed at that height
    return np.log(v2/v1)/np.log(z2/z1)


url_erddap = 'https://erddap.maracoos.org/erddap/'
dataset_id = 'AtlanticShores_ASOW-6_wind'

start = '2020-06-01T01:00:00Z'
# start = '2021-08-21T00:00:00Z'
# stop = '2021-08-30T00:00:00Z'
stop = '2020-09-01T01:00:00Z'

variables = [
    'wind_speed',
    'altitude',
    'wind_from_direction',
    'time'
]

constraints = {
    'time>=': start,
    'time<=': stop,
    'altitude>=': 0,
    'altitude<=': 260,
}

e = ERDDAP(
    server=url_erddap,
    protocol='tabledap',
    response='nc'
)

e.dataset_id = dataset_id
e.constraints = constraints
e.variables = variables

df = e.to_pandas(
    index_col="time (UTC)",
    parse_dates=True, )

df.reset_index(inplace=True)

df.rename(columns={'time (UTC)': 'time', 'wind_speed (m/s)': 'wind_speed', 'altitude (m)': 'altitude',
                   'wind_from_direction (degree)': 'wind_from_direction'}, inplace=True)

df['time'] = df['time'].dt.tz_localize(None)

df.set_index('time', inplace=True)

ds = df.to_xarray()

heights = np.unique(ds.altitude)

date_range = pd.date_range(datetime.strptime(start, '%Y-%m-%dT%H:%M:%SZ'),
                           datetime.strptime(stop, '%Y-%m-%dT%H:%M:%SZ') - timedelta(hours=1),
                           freq="H")


for date in date_range:
    try:
        profile_date = date.strftime("%Y-%m-%dT%H:%M:000000000")
        print('Processing: ' + profile_date)

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6, 4))
        plt.style.use(u'seaborn-colorblind')

        # as_ds.wind_speed[as_ds.wind_speed > 55] = np.nan
        # as_ds.wind_speed[as_ds.wind_speed < 0] = np.nan

        u = ds.wind_speed.loc[profile_date]
        z = ds.altitude.loc[profile_date]

        u[u > 55] = np.nan
        u[u < 0] = np.nan

        line1 = ax1.plot(u, z, '-', marker='o')
        line2 = ax0.plot(u, z, '-', marker='o')

        ax1.set_yscale('log')
        ax0.set_ylabel('altitude (m)')
        ax0.set_yticks(heights)
        ax1.set_xlabel('wind speed (m/s)')
        ax0.set_xlabel('wind speed (m/s)')
        fig.suptitle('Wind Speeds at ASOW-4 ' + date.strftime("%Y/%m/%d H%H") + ' (UTC)')

        # plt.legend(loc='best', fontsize='medium')

        # fig.xlim(left=0, right=20)
        ax0.grid(True)
        ax1.grid(True)
        # ax.autoscale(enable=True, axis='x', tight=True)
        plt.savefig('fitted_profiles/profile/ASOSB_profile_' + date.strftime("%Y%m%dT%HH") + '.png', dpi=300)
        plt.clf()
        plt.close()



        # m, b = np.polyfit(u, z, 1)
        # plt.plot(u, m*u + b)

        new_y = []
        new_x = []

        for jj in range(0, len(heights)):
            new_y.append(np.log(u[jj]/u[0]))
            new_x.append(np.log(heights[jj]/heights[0]))

        new_m, new_b = np.polyfit(new_x, new_y, 1)

        C = np.exp(new_b)

        # wind shear calculation for 10m to 80m alpha
        # 0 is the index for 10m and 3 is the index for 80m
        v1 = u[0]
        v2 = u[3]
        z1 = heights[0]
        z2 = heights[3]

        alpha_10m_80m = wind_shear_exp(v1, v2, z2, z1)

        # plt.subplot(4, 4, 2)
        plt.subplots(figsize=(6, 4))
        plt.style.use(u'seaborn-colorblind')

        plt.plot(new_x, new_y, 'o')
        plt.plot(new_x, new_m*np.array(new_x) + new_b, '.-',  color='black')
        ax = plt.gca()
        ax.text(0.1, 0.7,
                'C: ' + str(round(C, 3)) + '\n' +
                'b: ' + str(round(new_b, 3)) + '\n' +
                'fitted alpha: ' + str(round(new_m, 3)) + '\n' +
                '10m and 80m alpha ' + str(alpha_10m_80m.round(3).values),
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8))
                )

        plt.ylabel('$ln(u/u_r)$')
        plt.xlabel('$ln(z/z_r)$')
        # plt.xlim(left=0)
        # plt.ylim(bottom=0)
        plt.title('Fitted Profiles at ASOW-4 ' + date.strftime("%Y/%m/%d H%H") + ' (UTC)')
        plt.grid(True)

        plt.savefig('fitted_profiles/fitted/ASOSB_fitted_profile_' + date.strftime("%Y%m%dT%HH") + '.png', dpi=300)
        # plt.savefig('fitted_profiles/ASOSB_dual_profile_' + date.strftime("%Y%m%dT%HH") + '.png', dpi=300)

        plt.clf()
        plt.close()

        ws_fa = [v1]
        ws_ca = [v1]

        for ii in range(0, len(heights)-1):
            target_height = heights[ii+1]
            ref_height = heights[0]
            ws_fa.append(C * v1 * (target_height / ref_height) ** new_m)
            ws_ca.append(v1 * (target_height / ref_height) ** alpha_10m_80m)

        # for ii in range(0, len(heights)-1):
        #     target_height = heights[ii+1]
        #     ref_height = heights[ii]
        #     ws_fa.append(C * ws_fa[ii] * (target_height / ref_height) ** new_m)
        #     ws_ca.append(ws_ca[ii] * (target_height / ref_height) ** alpha_10m_80m)

        # nam_ws = nam_ws * (160 / 80) ** alpha

        observations_fa = np.array(ws_fa)
        observations_ca = np.array(ws_ca)
        forecast = u
        ind_fa = np.where((~np.isnan(observations_fa) & ~np.isnan(forecast)))
        ind_ca = np.where((~np.isnan(observations_ca) & ~np.isnan(forecast)))

        # obs_mean = np.mean(observations_fa[ind])
        # for_mean = np.mean(forecast[ind])
        # obs_std = np.std(observations_fa[ind])
        # for_std = np.std(forecast[ind])

        rms_fa = np.sqrt(np.mean((forecast[ind_fa]-observations_fa[ind_fa])**2))
        rms_ca = np.sqrt(np.mean((forecast[ind_ca]-observations_ca[ind_ca])**2))

        # label = plt.plot([], [], ' ', label='C = ' + str(round(C, 3)))
        line1 = plt.plot(ws_fa, heights, color='red',
                         label='Fitted \u03B1:           ' +
                               'C=' + str(round(C, 3)) +
                               ', \u03B1=' + str(round(new_m, 3)) +
                               ', rms=' + str(rms_fa.data.round(3))
                         )
        line2 = plt.plot(ws_ca, heights, color='blue',
                         label='10m to 80m \u03B1: ' +
                               'C= 1.000' +
                               ', \u03B1=' + str(alpha_10m_80m.round(3).values) +
                               ', rms=' + str(rms_ca.data.round(3))
                         )
        line3 = plt.plot(u, heights, color='black', label='Atlantic Shores', marker='o')

        plt.ylabel('heights (m)')
        plt.xlabel('wind speeds (m/s)')
        plt.yticks(heights)
        # plt.xlim(left=0)
        # plt.ylim(bottom=0)
        plt.legend(loc='upper center', fontsize='medium', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=1)
        plt.title('Alpha Profiles at ASOW-4 ' + date.strftime("%Y/%m/%d H%H") + ' (UTC)')
        plt.grid(True)
        plt.subplots_adjust(bottom=0.27)

        plt.savefig('fitted_profiles/comp/ASOSB_profile_comp_' + date.strftime("%Y%m%dT%HH") + '.png', dpi=300)

        plt.clf()
        plt.close()
    except:
        print('failed')

# ds.hvplot.line(x='time', y=['wind_speed', 'wind_from_direction'], grid=True, subplots=True,
#                shared_axes=False, width=400, height=400).cols(2)


