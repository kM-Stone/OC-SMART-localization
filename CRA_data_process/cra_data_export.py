'''
Author: yshi
Date: 2022-07-26 10:37:21
LastEditTime: 2022-08-03 17:06:16
LastEditors: Please set LastEditors
Description: 从原始CRA40数据文件中提取并整合所需变量
FilePath: /yshi/oc-smart_localization/CRA_data_process/cra_data_export.py
'''

#%%
import xarray as xr
import numpy as np
import logging

logging.basicConfig(level=logging.INFO,
                    filename='data_merge.log',
                    format="%(levelname)s:%(message)s",
                    filemode='w')

DATA_PATH = '/home/Public_Data/CRA40/'


class CRADataExport():
    '''
    从指定的CRA文件中提取所需数据并合并
    '''

    def __init__(self, file_time, lat_range=None, lon_range=None) -> None:
        self.file_time = file_time
        self.file_single = DATA_PATH + f'CRA40_SINGLE_{file_time}_GLB_0P25_HOUR_V1_0_0.grib2'
        self.file_singlea = DATA_PATH + f'CRA40_SINGLEA_{file_time}_GLB_0P25_HOUR_V1_1_2.grib2'
        if lat_range == None:
            self.lat_range = (90., -90.)
        else:
            self.lat_range = lat_range

        if lon_range == None:
            self.lon_range = (0, 360.)  # 注意原数据的经度格式
        else:
            self.lon_range = lon_range

    def read_grib_variable(self, file_path, **filter_by_keys):
        # 关于filter_by_keys的设置
        # https://www.heywhale.com/mw/project/627d38e9c08fe7829fee97c3
        dataset = xr.open_dataset(
            file_path,
            engine='cfgrib',
            backend_kwargs={"filter_by_keys": filter_by_keys})
        dataset = dataset.sel(latitude=slice(*self.lat_range),
                              longitude=slice(*self.lon_range))
        return dataset

    def met_merge(self):
        '''变量提取与合并'''
        # surface pressure
        sp = self.read_grib_variable(self.file_single,
                                       typeOfLevel='surface',
                                       paramId=134)['sp']
        sp /= 100
        sp.attrs['units'] = 'hPa'
        sp = sp.drop_vars(['valid_time', 'step', 'surface'])

        # air pressure at meansea level
        msl = self.read_grib_variable(self.file_single,
                                       typeOfLevel='meanSea',
                                       paramId=151)['msl']
        msl /= 100
        msl.attrs['units'] = 'hPa'
        msl = msl.drop_vars(['valid_time', 'step', 'meanSea'])
        
        # total ozone
        ozone = self.read_grib_variable(self.file_single,
                                        typeOfLevel='unknown',
                                        paramId=260130)['tozne']
        ozone = ozone.drop_vars(['valid_time', 'step', 'level'])

        # 2m rh 和 10m wind 的垂直坐标冲突，此处删去 heightAboveGround
        rh2 = self.read_grib_variable(self.file_singlea,
                                      typeOfLevel='heightAboveGround',
                                      level=2)['r2']
        rh2 = rh2.drop_vars(['valid_time', 'step', 'heightAboveGround'])

        wind = self.read_grib_variable(self.file_singlea,
                                       typeOfLevel='heightAboveGround',
                                       level=10)
        wind = wind.drop_vars(['valid_time', 'step', 'heightAboveGround'])
        ws = np.sqrt(wind['u10']**2 + wind['v10']**2)
        ws.name = 'ws'
        ws.attrs = {'long_name': '10 m wind speed', 'units': 'm/s'}

        self.dataset = xr.merge([sp, ozone, rh2, ws, msl])
        self.dataset = longitude_proc(self.dataset)
        self.dataset.attrs = {}

        logging.info(f'{self.file_time} 变量提取合并完成')

    def file_save(self, save_dir):
        self.dataset.to_netcdf(save_dir + f'CRA{self.file_time}_6h.nc')
        logging.info(f'{self.file_time} 文件已保存')


def longitude_proc(ds):
    ds['longitude'] = xr.where(ds['longitude'] > 180, ds['longitude'] - 360,
                               ds['longitude'])
    ds = ds.sel(longitude=sorted(ds['longitude']))
        
    return ds

# %%
if __name__ == '__main__':
    from glob import glob
    import re
    from multiprocessing import Pool
    
    def error_print(error_msg):
        logging.error(error_msg)

    def data_process(timestr):
        data_export = CRADataExport(timestr)
        data_export.met_merge()
        data_export.file_save(
            save_dir='/home/yshi/oc-smart_localization/Python_Linux/anc/')

    timelist = map(lambda x: re.findall(r'\d{10}', x)[0],
                   glob(DATA_PATH + '*SINGLE*.grib2'))
    # log_to_stderr()
    pool = Pool(processes=8)
    
    for timestr in timelist:
        pool.apply_async(data_process, args=(timestr, ), error_callback=error_print)
    pool.close()
    pool.join()