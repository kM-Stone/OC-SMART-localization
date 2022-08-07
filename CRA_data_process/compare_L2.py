'''
Author: yshi
Date: 2022-07-28 10:48:01
LastEditTime: 2022-08-04 11:21:25
LastEditors: Please set LastEditors
Description: 比较替换本地CRA数据前后运行结果的比较
FilePath: /yshi/oc-smart_localization/CRA_data_process/compare_L2.py
'''
# %%
from fileinput import filename
from matplotlib.pyplot import suptitle
import proplot as pplt
import xarray as xr
import numpy as np
import scipy.stats as st


def plot_scatter(xdata, ydata, ax, **plt_kwarg):
    ax.plot(
        (0, 1),
        (0, 1),
        color='k',
        zorder=0,
        transform=ax.transAxes,
    )
    p = ax.scatter(xdata, ydata, zorder=1, **plt_kwarg)
    r = st.pearsonr(xdata, ydata)[0]
    ax.text(0.65, 0.05, f'R = {r:.4f}', transform=ax.transAxes)
    return p


def data_plot(ax, varname,  **plt_kwarg):
    mask = ~np.isnan(ds_cra[varname]) & ~np.isnan(ds_ncep[varname])
    # ind = sample(range(mask.sum().values), 10000)
    xdata = ds_ncep[varname].values[mask]
    ydata = ds_cra[varname].values[mask]
    plot_scatter(xdata, ydata, ax, s=4,  **plt_kwarg)
    vmax = max(xdata.max(), ydata.max())
    ax.format(xlim=(0, vmax),
              ylim=(0, vmax),
              title=varname,
              xlabel='NCEP',
              ylabel='CRA')


OUT_DIR = '/home/yshi/oc-smart_localization/Python_Linux/L2/'
# file_name = 'FY3D_MERSI_GBAL_L1_20200518_0500_1000M_MS_L2_OCSMART.h5'
file_name = 'MYD021KM.A2020138.1200.061.2020139151535_L2_OCSMART.h5'
prefix = 'MYD'
# %%

ds_cra = xr.open_dataset(OUT_DIR + 'CRA_' + file_name,
                         engine='h5netcdf',
                         phony_dims='access')
ds_ncep = xr.open_dataset(OUT_DIR + file_name,
                          engine='h5netcdf',
                          phony_dims='access')
fig, axs = pplt.subplots(
    ncols=3,
    aspect=(1, 1),
    share=1,
)
for i, varname in enumerate(['chlor_a(oci)', 'chlor_a(yoc)', 'tsm(yoc)']):
    data_plot(axs[i], varname)
axs.format(suptitle=file_name)
fig.savefig(
    f'/home/yshi/oc-smart_localization/CRA_data_process/figure/{prefix}_chlor_tsm.jpg')

# %%
for group_name in [
    # 'adg', 'AOD', 'ap', 'aph', 'bbp', 'bp',
     'Rrs'
    ]:
    print(group_name)
    ds_cra = xr.open_dataset(OUT_DIR + 'CRA_' + file_name,
                             engine='h5netcdf',
                             phony_dims='access',
                             group=f'/{group_name}')
    ds_ncep = xr.open_dataset(OUT_DIR + file_name,
                              engine='h5netcdf',
                              phony_dims='access',
                              group=f'/{group_name}')
    n_variables = len(ds_cra.keys())
    fig, axs = pplt.subplots(
        ncols=3,
        nrows=int(np.ceil(n_variables / 3)),
        aspect=(1, 1),
        share=1,
    )
    for i, varname in enumerate(ds_cra.keys()):
        data_plot(axs[i], varname)
    axs.format(suptitle=file_name)
    fig.savefig(
        f'/home/yshi/oc-smart_localization/CRA_data_process/figure/{prefix}_{group_name}.jpg'
    )
