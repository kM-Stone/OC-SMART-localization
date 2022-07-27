<!--
 * @Author: yshi
 * @Date: 2022-07-26 10:49:21
 * @LastEditTime: 2022-07-26 10:49:21
 * @LastEditors: 
 * @Description:
 * @FilePath: /yshi/internship_work/oc_smart/Python_Linux/record.md
-->
## 可能的问题

1. ImportError: libpoppler.so.71: cannot open shared object file: No such file or directory
    主要是poppler库的版本不匹配
2. ImportError: libgdal.so.20: undefined symbol: _ZN6libdap3DDSC1EPNS_15BaseTypeFactoryERKSs
    原因未知

guide上的python3.6.6版本有点老, 可以尝试建立环境时只指定3.6版本
import gdal 改为 from osgeo import gdal

3. 为了支持CRA40数据， 还需安装 xarray 和 cfgrib库

## 框架
1. 参数设置: I/O路径, 传感器识别
2. auxilary data:
   + land_mask设置
3. sensorinfo.py: 根据L1数据识别并从sensorinfo文件夹读取传感器信息（7列）
4. 读取geo数据(经纬度信息)
5. 读取l1b数据，主要是计算可见各波段的reflectance
6. 读取ancillary数据
   + 根据L1B数据解码时间信息, 构造该数据起止时刻的中间时刻
   + 下载该时刻前后两个时刻可获得的数据， 然后插值到该时刻上；如下载不到数据则使用气候态
   + 读取辅助数据，整合到等经纬度网格上，注意经纬度的方向和正负号
7. 数据初始化
   + 神经网络数据加载
   + 算法初始化
   + 各种矩阵的初始化
   + 掩膜矩阵的计算， 用于计算符合条件的水体数据
     + 每个波段的反射率应该>0: `mask_valid = (np.sum(l1b.reflectance <= 0.0, 2) == 0)` 
     + 太阳高度角和仪器高度角限制：`mask_solz = l1b.solz < solz_limit`, `mask_senz = l1b.senz < senz_limit`
     + 水体： `water_portion > water_subpixl_limit`
     + 上述条件取交集即构成最终的掩膜矩阵mask_valid_geo_water，注意使用掩膜矩阵索引后得到的均为一维数组
8. 气体透射率计算
