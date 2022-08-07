'''
Author: yshi
Date: 2022-08-02 16:41:02
LastEditTime: 2022-08-03 17:09:04
LastEditors: Please set LastEditors
Description: 比较CRA数据和NCEP数据
FilePath: /yshi/oc-smart_localization/CRA_data_process/compare_met.py
'''

# %%
from fileinput import filename
from matplotlib.pyplot import suptitle
import proplot as pplt
import xarray as xr
import numpy as np
import scipy.stats as st
from datetime import datetime
from pyhdf.SD import SD, SDC
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from glob import glob

MET_DIR = '/home/yshi/oc-smart_localization/Python_Linux/anc/'


def get_CRA_O3(datestr):
    ozone = xr.concat([
        xr.open_dataset(_)['tozne']
        for _ in glob(MET_DIR + f'CRA{datestr[:-2]}*.nc')
    ],
                      dim='time').mean('time')
    return ozone


def get_OMI_O3(datestr):
    time_struct = datetime.strptime(datestr, '%Y%m%d%H').timetuple()
    year, doy, hour = time_struct.tm_year, time_struct.tm_yday, time_struct.tm_hour
    ozone_file = MET_DIR + f'N{year:03d}{doy:03d}00' + '_O3_AURAOMI_24h.hdf'
    f = SD(ozone_file, SDC.READ)
    ds = xr.Dataset(
        coords={
            'latitude': (['latitude'], np.arange(89.5, -90.5, -1)),
            'longitude': (['longitude'], np.arange(-179.5, 180.5, 1))
        })
    ds['tozne'] = (['latitude', 'longitude'], f.select('ozone')[:, :])
    return ds['tozne']


def merge_NCEP_hdf4(datestr):
    time_struct = datetime.strptime(datestr, '%Y%m%d%H').timetuple()
    year, doy, hour = time_struct.tm_year, time_struct.tm_yday, time_struct.tm_hour
    time_str = f'N{year:03d}{doy:03d}{hour:02d}'
    met_file = MET_DIR + time_str + '_MET_NCEPR2_6h.hdf'

    f1 = SD(met_file, SDC.READ)
    ds = xr.Dataset(
        coords={
            'latitude': (['latitude'], np.arange(90, -91, -1)),
            'longitude': (['longitude'], np.arange(-179.5, 180.5, 1))
        })

    ds['msl'] = (['latitude', 'longitude'], f1.select('press')[:, :])
    ds['ws'] = (['latitude', 'longitude'],
                np.sqrt(
                    f1.select('z_wind')[:, :]**2,
                    f1.select('m_wind')[:, :]**2))
    ds['r2'] = (['latitude', 'longitude'], f1.select('rel_hum')[:, :])

    return ds


#%%
for datestr in ['2020051800','2020051806','2020051812','2020051818','2020051900']:
    cra_name = f'CRA{datestr}_6h.nc'

    ds_cra = xr.open_dataset(MET_DIR + cra_name)
    ds_ncep = merge_NCEP_hdf4(datestr)

    lat_range = (65, 15)
    lon_range = (70, 135)

    fig, ax = pplt.subplots(ncols=4, nrows=2, proj='cyl')
    for i, variable in enumerate(['msl', 'ws', 'r2']):
        for j, ds in enumerate([ds_cra, ds_ncep]):
            vmax = 10 if i == 1 else None
            p = ax[j, i].pcolormesh(ds[variable].sel(latitude=slice(*lat_range),
                                                    longitude=slice(*lon_range)),
                                    cmap='jet',
                                    vmax=vmax)
            ax[j, i].colorbar(p, loc='b')
    # %
    ozone_cra = get_CRA_O3(datestr)
    ozone_ncep = get_OMI_O3(datestr)
    for j, da in enumerate([ozone_cra, ozone_ncep]):
        p = ax[j, -1].pcolormesh(da.sel(latitude=slice(*lat_range),
                                        longitude=slice(*lon_range)),
                                cmap='jet')
        ax[j, -1].colorbar(p, loc='b')

    ax[0, :].format(title='CRA')
    ax[1, :].format(title='NCEP')
    ax.format(suptitle=datestr,
            lonlim=lon_range,
            latlim=lat_range,
            labels=True,
            coast=True,
            collabels=['Pres', 'WS', 'RH', 'Ozone'])

    fig.savefig(f'/home/yshi/oc-smart_localization/CRA_data_process/figure/MET{datestr}_CRA.vs.NCEP.jpg')