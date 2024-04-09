# -*- coding: utf-8 -*-
# %%
import os, sys
# from pathlib import Path
# import os.path
# from mjoindices.olr_handling import OLRData
# import mjoindices.olr_handling as olr
# import mjoindices.olr_handling as olr_handling
# import mjoindices.omi.omi_calculator as omi
# import mjoindices.empirical_orthogonal_functions as eof
# import mjoindices.omi.wheeler_kiladis_mjo_filter as wkfilter
import scipy
import numpy as np

import xarray as xr
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta
import calendar
import matplotlib.pyplot as plt
import logging
import torch
import random

np.random.seed(2024)
random.seed(2024)
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
torch.cuda.manual_seed_all(2024)  # 为所有GPU设置种子


sys.path.append('/home/jermy/l-python/')
# import lib_data as lib_data
import lib_math as lib_math


# ====================================================
# functions
# function: get_fn
# define filename based on varriable name and station name
# ====================================================
def get_fn(var, station):
    if var in ["max_temp", "min_temp"]:  # max min temp
        var_fn = "maxmin"
        var_fn2 = "maxmin_temp"
        if station in ["HKO"]:
            fn = "./data/%s_daily/hko_d_temp_maxmin.txt" % (var_fn)
        elif station in ["HKA"]:
            fn = "./data/%s_daily/hkatemp_d_max_min1.txt" % (var_fn)
        else:
            fn = "./data/%s_daily/%s%s.txt" % (var_fn, var_fn2, station)
    elif var == "mean_temp":  # mean temp
        var_fn = "meantemp"
        if station in ["HKO"]:
            fn = "./data/%s_daily/hkotemp_d_mean.txt" % (var)
        elif station in ["HKA"]:
            fn = "./data/%s_daily/hkatemp_d_mean_1.txt" % (var)
        else:
            fn = "./data/%s_daily/%s%s.txt" % (var, var_fn, station)
    elif var == "mslp":  # mslp
        fn = "./data/%s_daily/%s%s.txt" % (var, var, station)
    elif var == "rf":  # rainfall
        if station in ["HKO"]:
            fn = "./data/%s_daily/%s%s_d.txt" % (var, "hko", var)
        elif station in ["HKA"]:
            fn = "./data/%s_daily/%s%s_d.txt" % (var, "hka", var)
        else:
            fn = "./data/%s_daily/%s_%s_daily.txt" % (var, var, station)
    elif var == "rh":  # relative humidity
        if station in ["HKO"]:
            fn = "./data/%s_daily/hkorh_d.txt" % (var)
        elif station in ["HKA"]:
            fn = "./data/%s_daily/hkarh_d2.txt" % (var)
        else:
            fn = "./data/%s_daily/%s%s.txt" % (var, var, station)
    elif var == "wind":  # wind
        var_fn = "wind"
        asd = asd
    return (fn)


# ====================================================
# function: read_station_data(var, station)
# read hko station data
# ====================================================
def read_station_data(var, station):
    fn = get_fn(var, station)
    # print(fn)
    # print(os.path.exists(fn))

    # ----- read txt -----
    # if var in ["max_temp", "min_temp"]:
    #     print(">>> reading " + fn)
    #     if station in ["HKO"]:
    #         fdata = pd.read_csv(fn, skiprows=3, header=None, sep="\s+")
    #     else:
    #         fdata = pd.read_csv(fn, skiprows=4, header=None, sep="\s+")
    #     fdata.columns = ["yyyymmdd", "data_max", "data_min"]
    #     fdata["data_max"][fdata["data_max"]>30000] = np.nan
    #     fdata["data_min"][fdata["data_min"]>30000] = np.nan
    #     fdata["ano_max"] = fdata["data_max"] + 0
    #     fdata["ano_min"] = fdata["data_min"] + 0
    if var == "max_temp":
        print(">>> reading " + fn)
        if station in ["HKO"]:
            fdata = pd.read_csv(fn, skiprows=3, header=None, sep="\s+")
        else:
            fdata = pd.read_csv(fn, skiprows=4, header=None, sep="\s+")
        fdata = fdata.iloc[:, [0, 1]]
        fdata.columns = ["yyyymmdd", "data"]
        fdata["data"][fdata["data"] > 30000] = np.nan
        # -- convert unit: degC
        if station in ["HKO"]:
            pass
        else:
            fdata["data"] = fdata["data"] * 0.1
        # -- create empty list
        fdata["ano"] = fdata["data"] + 0

    elif var == "min_temp":
        print(">>> reading " + fn)
        if station in ["HKO"]:
            fdata = pd.read_csv(fn, skiprows=3, header=None, sep="\s+")
        else:
            fdata = pd.read_csv(fn, skiprows=4, header=None, sep="\s+")
        fdata = fdata.iloc[:, [0, 2]]
        fdata.columns = ["yyyymmdd", "data"]
        fdata["data"][fdata["data"] > 30000] = np.nan
        # -- convert unit: degC
        if station in ["HKO"]:
            pass
        else:
            fdata["data"] = fdata["data"] * 0.1
        # -- create empty list
        fdata["ano"] = fdata["data"] + 0
    elif var == "mean_temp":
        if station in ["HKA"]:
            fdata = pd.read_csv(fn, skiprows=4, header=None, sep="\s+")
        else:
            fdata = pd.read_csv(fn, skiprows=2, header=None, sep="\s+")
        fdata.columns = ["yyyymmdd", "data"]
        fdata["data"][fdata["data"] > 30000] = np.nan
        # -- convert unit: degC
        if station in ["HKO", "HKA"]:
            fdata["data"] = fdata["data"] * 0.1
        else:
            pass
        # -- create empty list
        fdata["ano"] = fdata["data"] + 0
    else:
        asd = sad
    fdata["year"] = np.floor(fdata.yyyymmdd / 10000).astype(int)
    fdata["month"] = np.floor((fdata.yyyymmdd - fdata["year"] * 10000) / 100).astype(int)
    fdata["day"] = np.floor((fdata.yyyymmdd - fdata["year"] * 10000 - fdata["month"] * 100) / 1).astype(int)

    return fdata


# ====================================================
# function: check_if_there_is_missing_dates
# check if there is missing dates
# check whether the time series is continuous
# ====================================================
def check_if_there_is_missing_dates(fdata):
    tmp = []
    for i in range(len(fdata)):
        # ftime   = (datetime(2019,2,3) - np.datetime64('1800-01-01T00:00:00Z')) / np.timedelta64(1, 'h')
        tmp.append(nc.date2num(datetime(fdata["year"].iloc[i], fdata["month"].iloc[i], fdata["day"].iloc[i]),
                               units="days since 1800-01-01 00:00"))
    fdata["time"] = np.array(tmp)

    # ----- check ----
    fdata["time_difference"] = np.nan
    fdata["time_difference"][1:] = fdata["time"].values[1:] - fdata["time"].values[:-1]

    discont_ind = (fdata["time_difference"] > 1)
    if discont_ind.sum() >= 1:
        print(">>> Missing date exists --> the time series is not continuous")
        print(fdata[discont_ind])
        # asd=asd
    else:
        print(">>> No Missing date --> the time series is continuous")
    return (fdata)


# ====================================================
# remove climatology
# ====================================================
def calc_ano_by_rm_daily_ltm_runavg(fdata_tmp, var, window=31, year1=2006, year2=2022):
    fdata = fdata_tmp + 0
    fdata["ltm"] = fdata["ano"] + np.nan
    for month in range(1, 13):
        for day in range(1, calendar.monthrange(2004, month)[1] + 1):
            ltm_sample = np.array([])
            for year in range(year1, year2 + 1):
                if (month == 2) & (day == 29) & (year % 4 != 0):
                    date_tmp0 = datetime(year, 2, 28)
                else:
                    date_tmp0 = datetime(year, month, day)
                date_tmp1 = date_tmp0 - timedelta(days=int(window / 2))
                date_tmp2 = date_tmp0 + timedelta(days=int(window / 2))
                time_tmp1 = nc.date2num(date_tmp1, units="days since 1800-01-01 00:00")
                time_tmp2 = nc.date2num(date_tmp2, units="days since 1800-01-01 00:00")

                ltm_sample_tmp = np.array(fdata[(fdata.time >= time_tmp1) & (fdata.time <= time_tmp2)].data)
                ltm_sample = np.concatenate((ltm_sample, ltm_sample_tmp))
                # if np.isnan(ltm_sample_tmp).sum()==0:
                #     ltm_sample.append(ltm_sample_tmp)

            # ----- Calculate std -----
            ltm_sample = np.array(ltm_sample)
            # print(">>> %0.2i-%0.2i (ltm_sample.shape=%r)" % (month, day, ltm_sample.shape))
            # print(ltm_sample.shape)
            # print(ltm_sample.reshape(-1).shape)
            ltm_tmp = np.nanmean(ltm_sample.reshape(-1))

            # ----- Normalize anomaly -----
            fdata.ano[(fdata.month == month) & (fdata.day == day)] = fdata.data[(fdata.month == month) & (
                        fdata.day <= day)] - ltm_tmp
            fdata.ltm[(fdata.month == month) & (fdata.day == day)] = ltm_tmp + 0
    return (fdata)


# ====================================================
# function: filtwghts_lanczos
# Calculates the Lanczos filter weights.
# ====================================================
def filtwghts_lanczos(nwt, filt_type, fca, fcb):
    """
    Calculates the Lanczos filter weights.

    Parameters
    ----------
    nwt : int
        The number of weights.
    filt_type : str
        The type of filter. Must be one of 'low', 'high', or 'band'.
    fca : float
        The cutoff frequency for the low or band filter.
    fcb : float
        The cutoff frequency for the high or band filter.

    Returns
    -------
    w : ndarray
        The Lanczos filter weights.

    Notes
    -----
    The Lanczos filter is a type of sinc filter that is truncated at a specified frequency.
    This function implements a Lanczos filter in the time domain.
    """

    pi = np.pi
    k = np.arange(-nwt, nwt + 1)

    if filt_type == 'low':
        w = np.zeros(nwt * 2 + 1)
        w[:nwt] = ((np.sin(2 * pi * fca * k[:nwt]) / (pi * k[:nwt])) * np.sin(pi * k[:nwt] / nwt) / (
                    pi * k[:nwt] / nwt))
        w[nwt + 1:] = ((np.sin(2 * pi * fca * k[nwt + 1:]) / (pi * k[nwt + 1:])) * np.sin(pi * k[nwt + 1:] / nwt) / (
                    pi * k[nwt + 1:] / nwt))
        w[nwt] = 2 * fca
    elif filt_type == 'high':
        w = np.zeros(nwt * 2 + 1)
        w[:nwt] = -1 * (np.sin(2 * pi * fcb * k[:nwt]) / (pi * k[:nwt])) * np.sin(pi * k[:nwt] / nwt) / (
                    pi * k[:nwt] / nwt)
        w[nwt + 1:] = -1 * (np.sin(2 * pi * fcb * k[nwt + 1:]) / (pi * k[nwt + 1:])) * np.sin(
            pi * k[nwt + 1:] / nwt) / (pi * k[nwt + 1:] / nwt)
        w[nwt] = 1 - 2 * fcb
    else:
        w1 = np.zeros(nwt * 2 + 1)
        w1[:nwt] = (np.sin(2 * pi * fca * k[:nwt]) / (pi * k[:nwt])) * np.sin(pi * k[:nwt] / nwt) / (pi * k[:nwt] / nwt)
        w1[nwt + 1:] = (np.sin(2 * pi * fca * k[nwt + 1:]) / (pi * k[nwt + 1:])) * np.sin(pi * k[nwt + 1:] / nwt) / (
                    pi * k[nwt + 1:] / nwt)
        w1[nwt] = 2 * fca
        w2 = np.zeros(nwt * 2 + 1)
        w2[:nwt] = (np.sin(2 * pi * fcb * k[:nwt]) / (pi * k[:nwt])) * np.sin(pi * k[:nwt] / nwt) / (pi * k[:nwt] / nwt)
        w2[nwt + 1:] = (np.sin(2 * pi * fcb * k[nwt + 1:]) / (pi * k[nwt + 1:])) * np.sin(pi * k[nwt + 1:] / nwt) / (
                    pi * k[nwt + 1:] / nwt)
        w2[nwt] = 2 * fcb
        w = w2 - w1

    return w


# ====================================================
# function: calc_lanczos_filter(fdata_tmp, filt_type, fcd_low, fcd_high, nwt, timestep, min_periods)
# Calculates the Lanczos filtering.
# ====================================================
def calc_lanczos_filter(fdata_tmp, filt_type, fcd_low, fcd_high, nwt, timestep, min_periods):
    # ----- data -----
    data_tmp = np.array(fdata_tmp["data"]) + 0
    ano_tmp = np.array(fdata_tmp["ano"]) + 0
    # --
    index_tmp = np.arange(0, len(fdata_tmp))
    year_tmp = fdata_tmp.year + 0
    month_tmp = fdata_tmp.month + 0
    day_tmp = fdata_tmp.day + 0

    # ----- check nan -----
    # nan_ind = np.isnan(data_tmp)
    nan_ind = np.isnan(ano_tmp)
    if nan_ind.sum() > 0:
        print(fdata_tmp[nan_ind])
        print(">>> missing values exist!! nan values exists in the time series!!")
        # asd=asd
        # print(">>> missing values exist!! filling nan by linear interpolation!!")
        # sr_tmp = pd.Series(data_tmp)
        # sr_tmp = sr_tmp.interpolate(method='linear')
        # data_tmp = np.array(sr_tmp)
        # # --
        # sr_tmp = pd.Series(ano_tmp)
        # sr_tmp = sr_tmp.interpolate(method='linear')
        # ano_tmp = np.array(sr_tmp)

    # ====================================================
    # filter
    # ====================================================
    # ----- Lanczos parameters -----
    # timestep = 1 # time step: 1 day
    # fcd_low = 10 # low frequency cutoff: 10 days
    # fcd_high = 60 # high frequency cutoff: 60 days

    # nwt = 60 # number of weights: The filter uses 121 (nwgths*2+1) weights
    # filt_type = 'band' # 'low', 'high', 'band'
    fca = timestep / fcd_low
    fcb = timestep / fcd_high

    # ----- Lanczos bandpass filter -----
    wgt = filtwghts_lanczos(nwt, filt_type, fcb, fca)
    wgt = xr.DataArray(wgt, dims=['window'])
    # fil_tmp = xr.DataArray(data_tmp,dims='x')
    fil_tmp = xr.DataArray(ano_tmp, dims='x')
    # --
    # fil_tmp = fil_tmp.rolling(x=len(wgt),center=True).construct('window').dot(wgt)
    # # fil_tmp = (fil_tmp.rolling(x=len(wgt),center=True).construct('window')*wgt).sum(axis=1)
    # --
    # min_periods = 10 #nwt/2 #nwt + 0 # nwt/2
    fil_tmp_window = (fil_tmp.rolling(x=len(wgt), center=True).construct('window') * wgt)
    ind_nan = np.isnan(fil_tmp_window)
    ind_nan = ind_nan.sum(axis=1)
    fil_tmp = fil_tmp_window.sum(axis=1)
    fil_tmp[ind_nan > min_periods] = np.nan
    # --
    fil_tmp = fil_tmp.data
    print(np.isnan(fil_tmp).sum())
    print(np.isnan(ano_tmp).sum())
    print(np.isnan(data_tmp).sum())

    # -- set nan manually
    # data_tmp[nan_ind] = np.nan
    # ano_tmp[nan_ind] = np.nan
    # fil_tmp[nan_ind] = np.nan

    # -- append
    fdata_tmp["fil"] = fil_tmp
    fdata_tmp["fil_norm"] = fil_tmp / np.nanstd(fil_tmp)

    # --df
    # df = pd.DataFrame(np.transpose([year_tmp,month_tmp,day_tmp,data_tmp,ano_tmp,fil_tmp]), columns=["year","month","day","obs","ano","fil"])
    # stdd = np.nanstd(fil_tmp)*1
    # df["fil_norm"] = df.fil / stdd
    # df["qbie"] = df.fil_norm*0
    # df["qbie"][(df.fil_norm>=2)] = 2
    # df["qbie"][(df.fil_norm>=1) & (df.fil_norm<2)] = 1
    # df["qbie"][(df.fil_norm>-1) & (df.fil_norm<1)] = 0
    # df["qbie"][(df.fil_norm>-2) & (df.fil_norm<=-1)] = -1
    # df["qbie"][(df.fil_norm<=-2)] = -2
    # print(df[abs(df.fil_norm)>=2])
    # print(df[df.fil==df.fil.max()])
    # print(df[df.fil==df.fil.min()])
    # df.iloc[4420:4450]
    # print(df.iloc[np.isnan(fil_tmp)])

    return (fdata_tmp)


# ====================================================
# setting
# ====================================================
event_type = "positive"
event_type = "negative"
# --

# ====================================================
# stations
# ====================================================
# mean_temp && maxmin stations
station_list1 = ["HKO", "HKA", "BHD", "BR1", "CCH", "CWB", "EPC", "HKP", "HKS", "HPV", "JKB", "KAT", "KFB", "KLT", "KP",
                 "KSC", "KTG", "LFS", "NGP", "NLS", "PEN", "PLC", "SE1", "SEK", "SHA", "SKG", "SKW", "SLW", "SSH",
                 "SSP", "STY", "TAP", "TC", "TKL", "TLS", "TMS", "TP", "TU1", "TWN", "TW", "TY1", "TYW", "VP1", "WB1",
                 "WB2", "WB4", "WB8", "WGL", "WLP", "WTS", "YCT", "YLP"]
station_list2 = ["HKO", "HKA", "BHD", "BR1", "CCH", "CWB", "EPC", "HKP", "HKS", "HPV", "JKB", "KAT", "KFB", "KLT", "KP",
                 "KSC", "KTG", "LFS", "NGP", "NLS", "PEN", "PLC", "SE1", "SEK", "SHA", "SKG", "SKW", "SLW", "SSH",
                 "SSP", "STY", "TAP", "TC", "TKL", "TLS", "TMS", "TP", "TU1", "TWN", "TW", "TY1", "TYW", "VP1", "WB1",
                 "WB2", "WB4", "WB8", "WGL", "WLP", "WTS", "YCT", "YLP"]
# mslp stations
station_list3 = ["HKO", "HKA", "CCH", "KP", "LFS", "NLS", "PEN", "SEK", "SHA", "SLW", "SSH", "TC", "TKL", "TMS", "TP",
                 "WB1", "WB2", "WB4", "WB8", "WGL", "WLP", "YCT"]
# rf stations
station_list4 = ["HKO", "HKA", "BR1", "CCH", "CPH", "EPC", "GI", "HPV", "JKB", "KAT", "KFB", "KP", "KSC", "LAM", "LFS",
                 "PEN", "PLC", "PPC", "QU1", "R11", "R12", "R14", "R18", "R21", "R22", "R23", "R24", "R28", "R29",
                 "R31", "SE", "SEK", "SHA", "SKW", "SLW", "SSH", "SSP", "TAP", "TC", "TKL", "TMR", "TMS", "TTC", "TU1",
                 "TWN", "TYW", "VP1", "WGL", "WLP"]
# rh stations
station_list5 = ["HKO", "HKA", "BR1", "CCH", "HKS", "JKB", "KP", "KSC", "LFS", "NLS", "PEN", "SEK", "SHA", "SKG", "SLW",
                 "SSH", "TC", "TKL", "TMS", "TP", "TU1", "TWN", "TW", "TY1", "TYW", "WB1", "WB2", "WB4", "WB8", "WGL",
                 "WLP", "YCT"]
# wind stations
station_list6 = ["HKO", "HKA", "BHD", "CCB", "CCH", "CP1", "EPC", "GI", "HKS", "JKB", "KP", "LAM", "LFS", "NGP", "NLS",
                 "NP", "PEN", "PLC", "SC", "SEK", "SE", "SF", "SHA", "SHL", "SHW", "SKG", "SLW", "TC", "TKL", "TME",
                 "TMS", "TMT", "TO", "TPK", "TUN", "WB1", "WB2", "WB4", "WB8", "WGL", "WLP", "YTS"]
# ----- find duplicate stations -----
station_common = np.intersect1d(station_list1, station_list2)
station_common = np.intersect1d(station_common, station_list3)
station_common = np.intersect1d(station_common, station_list4)
station_common = np.intersect1d(station_common, station_list5)
station_common = np.intersect1d(station_common, station_list6)
# -- df
station_df = pd.DataFrame([station_list1, station_list2, station_list3, station_list4, station_list5, station_common]).T

# ====================================================
# read
# ====================================================
# for station in station_common:
# for station in ["HKO","WGL"]:
for station in ["HKO"]:
    # for station in ["SLW"]:
    # for station in ["WGL"]:
    # for station in ["SLW"]:
    # for station in ["HKO","LFS","SEK","SLW","WGL"]:

    # station = "HKO"
    # station = "HKA"
    # station = "CCH"
    # station = "KP"
    # station = "TKL"
    # ====================================================
    # read station data
    # ====================================================
    fdata_maxtemp = read_station_data("max_temp", station)
    fdata_mintemp = read_station_data("min_temp", station)
    fdata_meantemp = read_station_data("mean_temp", station)

    actual_fdata_meantemp = read_station_data("mean_temp", station)  #add

    # ----- extract time
    # discont_ind = (fdata["year"]>=0)
    # discont_ind = (fdata["year"]>=1989) & (fdata["year"]<=1991)
    # discont_ind = (fdata["year"]>=1950)
    # discont_ind = (fdata["year"]>=2010)
    fdata_maxtemp = fdata_maxtemp[(fdata_maxtemp["year"] >= 1950)].reset_index(drop=True)
    fdata_mintemp = fdata_mintemp[(fdata_mintemp["year"] >= 1950)].reset_index(drop=True)
    fdata_meantemp = fdata_meantemp[(fdata_meantemp["year"] >= 1950)].reset_index(drop=True)

    actual_fdata_meantemp = actual_fdata_meantemp[(actual_fdata_meantemp["year"] >= 1950)].reset_index(drop=True)  #add

    # ====================================================
    # check if there is missing dates
    # ====================================================
    fdata_maxtemp = check_if_there_is_missing_dates(fdata_maxtemp)
    fdata_mintemp = check_if_there_is_missing_dates(fdata_mintemp)
    fdata_meantemp = check_if_there_is_missing_dates(fdata_meantemp)

    actual_fdata_meantemp = check_if_there_is_missing_dates(actual_fdata_meantemp)   #add

    # ====================================================
    # remove climatology
    # ====================================================
    # ---------------
    # -- rm climatology by daily ltm (31-day running mean)
    # ---------------
    fdata_maxtemp = calc_ano_by_rm_daily_ltm_runavg(fdata_maxtemp, "max_temp", 31, 2006, 2022)
    fdata_mintemp = calc_ano_by_rm_daily_ltm_runavg(fdata_mintemp, "min_temp", 31, 2006, 2022)
    fdata_meantemp = calc_ano_by_rm_daily_ltm_runavg(fdata_meantemp, "mean_temp", 31, 2006, 2022)

    actual_fdata_meantemp = calc_ano_by_rm_daily_ltm_runavg(actual_fdata_meantemp, "mean_temp", 31, 2006, 2022)   #add
    # 对 actual_fdata_meantemp 应用正确的年份筛选，确保范围是2009至2021年
    actual_fdata_meantemp = actual_fdata_meantemp[
        (actual_fdata_meantemp['year'] >= 2009) & (actual_fdata_meantemp['year'] <= 2021)]                        #add
    actual_fdata_meantemp_path = r'2009-2021data\actual_data.csv'       #add
    actual_fdata_meantemp.to_csv(actual_fdata_meantemp_path, index=False, encoding='utf-8-sig')                   #add
    print(f"实际数据已保存至 {actual_fdata_meantemp_path}")

    # ====================================================
    # extract data for filtering
    # ====================================================
    # ----- Lanczos parameters -----
    timestep = 1  # time step: 1 day
    fcd_low = 10  # low frequency cutoff: 10 days
    fcd_high = 60  # high frequency cutoff: 60 days
    nwt = 60  # number of weights: The filter uses 121 (nwgths*2+1) weights
    filt_type = 'band'  # 'low', 'high', 'band'
    # -- allow max of 10 nan values in the window
    min_periods = 1  # 10 #nwt/2 #nwt + 0 # nwt/2

    # ----- Lanczos filter -----
    fdata_maxtemp = calc_lanczos_filter(fdata_maxtemp, filt_type, fcd_low, fcd_high, nwt, timestep, min_periods)
    fdata_mintemp = calc_lanczos_filter(fdata_mintemp, filt_type, fcd_low, fcd_high, nwt, timestep, min_periods)
    fdata_meantemp = calc_lanczos_filter(fdata_meantemp, filt_type, fcd_low, fcd_high, nwt, timestep, min_periods)

    filed_fdata_meantemp = calc_lanczos_filter(actual_fdata_meantemp, filt_type, fcd_low, fcd_high, nwt, timestep, min_periods)  #add
    # 对 filed_fdata_meantemp 应用相同的年份筛选和保存步骤
    filed_fdata_meantemp = filed_fdata_meantemp[
        (filed_fdata_meantemp['year'] >= 2009) & (filed_fdata_meantemp['year'] <= 2021)]                      #add
    filed_fdata_meantemp_path = r'2009-2021data\lanczos_filtered_data.csv'    #add
    filed_fdata_meantemp.to_csv(filed_fdata_meantemp_path, index=False, encoding='utf-8-sig')                   #add
    print(f"经过Lanczos滤波后的数据已保存至 {filed_fdata_meantemp_path}")

    print(fdata_maxtemp.iloc[150:155])
    print(fdata_mintemp.iloc[150:155])
    print(fdata_meantemp.iloc[150:155])

    # ====================================================
    # prepare data for 1D CNN
    # ====================================================
    # ----- Remove the partial year data -----
    ### ----- Split the dataset to training, validation, and testing -----
    # -- training data
    # year_ind = (fdata_meantemp.year>=1980) & (fdata_meantemp.year<=2017)
    year_ind = (fdata_meantemp.year >= 2006) & (fdata_meantemp.year <= 2019)
    # year_ind = (fdata_meantemp.year>=1994) & (fdata_meantemp.year<=2019)
    x_train = torch.tensor(fdata_meantemp.ano[year_ind][None, None, ...], dtype=torch.float64)  # x for training
    y_train = torch.tensor(fdata_meantemp.fil[year_ind][None, None, ...], dtype=torch.float64)  # y for training

    # -- validation data
    # year_ind = (fdata_meantemp.year>=2018) & (fdata_meantemp.year<=2019)
    year_ind = (fdata_meantemp.year >= 2020) & (fdata_meantemp.year <= 2021)
    x_val = torch.tensor(fdata_meantemp.ano[year_ind][None, None, ...], dtype=torch.float64)  # x for validation
    y_val = torch.tensor(fdata_meantemp.fil[year_ind][None, None, ...], dtype=torch.float64)  # y for validation

    # -- testing data
    # year_ind = (fdata_meantemp.year>=2020) & (fdata_meantemp.year<=2021)
    # year_ind = (fdata_meantemp.year>=2021) & (fdata_meantemp.year<=2021)
    year_ind = (fdata_meantemp.year >= 2022) & (fdata_meantemp.year <= 2022)
    # year_ind = (fdata_meantemp.year>=2022) & (fdata_meantemp.year<=2022)& (fdata_meantemp.month<=6)
    x_test = torch.tensor(fdata_meantemp.ano[year_ind][None, None, ...], dtype=torch.float64)  # x for testing
    y_test = torch.tensor(fdata_meantemp.fil[year_ind][None, None, ...], dtype=torch.float64)  # y for testing
    df_test = fdata_meantemp[year_ind] + 0

    # ----- deal with nans -----
    check_nan = np.isnan(x_train) | np.isnan(y_train)
    x_train_filled_zeros, y_train_filled_zeros = x_train + 0, y_train + 0
    x_train_filled_zeros[check_nan] = 0
    y_train_filled_zeros[check_nan] = 0
    check_nan = np.isnan(x_val) | np.isnan(y_val)
    x_val_filled_zeros, y_val_filled_zeros = x_val + 0, y_val + 0
    x_val_filled_zeros[check_nan] = 0
    y_val_filled_zeros[check_nan] = 0
    check_nan = np.isnan(x_test)
    x_test_filled_zeros = x_test + 0
    x_test_filled_zeros[check_nan] = 0

    # # ----- deal with nans -----
    # x_train_filled_zeros = torch.tensor(fdata_meantemp.ano[year_ind][None, None, ...], dtype=torch.float64)  # x for training
    # y_train_filled_zeros = torch.tensor(fdata_meantemp.fil[year_ind][None, None, ...], dtype=torch.float64)  # y for training
    # x_val_filled_zeros = torch.tensor(fdata_meantemp.ano[year_ind][None, None, ...], dtype=torch.float64)  # x for validation
    # y_val_filled_zeros = torch.tensor(fdata_meantemp.fil[year_ind][None, None, ...], dtype=torch.float64)  # y for validation
    # x_test_filled_zeros = torch.tensor(fdata_meantemp.ano[year_ind][None, None, ...], dtype=torch.float64)  # x for testing
    #
    # # Find NaN values in tensors and fill them with zeros
    # x_train_filled_zeros[x_train_filled_zeros.isnan()] = 0
    # y_train_filled_zeros[y_train_filled_zeros.isnan()] = 0
    # x_val_filled_zeros[x_val_filled_zeros.isnan()] = 0
    # y_val_filled_zeros[y_val_filled_zeros.isnan()] = 0
    # x_test_filled_zeros[x_test_filled_zeros.isnan()] = 0

    # ====================================================
    # 1D CNN functions
    # ====================================================
    ### Train the CNN model, get the CNN band pass filtered data, and save the weights to a text file
    # Set the output path to store the model weights
    # outPath = './02-try_by_pytorch/'

    kernel1 = 60  # 60
    kernel2 = 10  # 10
    no_epochs = 10000  # 10000 #500
    learning_rate = 0.001  # 0.01 # default=0.001
    epsilon = 1e-7  # default=1e-8
    var_name = 'mean_temp'  # 'olr'

    # ----- Define the model -----
    torch.set_default_dtype(torch.float64)


    class cnn1d_model(torch.nn.Module):
        def __init__(self, kernel_size_1=60, kernel_size_2=10):
            super(cnn1d_model, self).__init__()
            self.conv1 = torch.nn.Conv1d(1, 1, kernel_size=kernel_size_1, groups=1, bias=False,
                                         padding='same')  # Depthwise convolution
            self.conv2 = torch.nn.Conv1d(1, 1, kernel_size=kernel_size_2, groups=1, bias=False,
                                         padding='same')  # Depthwise convolution

        def forward(self, x):
            x1 = self.conv1(x)
            x2 = x - x1  # Subtract layer
            x3 = self.conv2(x2)
            return x3


    # 定义模型权重文件的路径
    model_weights_path = r'model weight\cnn1d_model_weights.pth'


    # ====================================================
    # train the model
    # ====================================================
    # ----- Create the model -----
    model = cnn1d_model(kernel1, kernel2)
    model.double()  # 确保模型与数据类型匹配

    # 在条件检查之前预定义变量
    loss_train_list = []
    loss_val_list = []
    corr_train_list = []
    corr_val_list = []
    aaaaaa = 0
    # 检查是否存在预训练的模型权重
    if os.path.isfile(model_weights_path):
    # if aaaaaa == 1:
        # 加载预训练的权重
        model.load_state_dict(torch.load(model_weights_path))
        print("Loaded pretrained weights.")
    else:
        # ----- 定义损失函数和优化器 -----
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=epsilon)

        # ----- Training loop -----
        for epoch in range(no_epochs):

            # -- Forward pass
            # outputs_train   = model(x_train) # by x_train
            # outputs_val     = model(x_val) # by x_val
            outputs_train = model(x_train_filled_zeros)  # by x_train
            outputs_val = model(x_val_filled_zeros)  # by x_val

            # -- Compute loss （计算该epoch的loss，计算loss时，忽略有nan的点）
            check_finite = np.isfinite(y_train)
            loss_train = criterion(outputs_train[check_finite], y_train[check_finite])  # by x_train
            # loss_train   = np.nanmean((outputs_train.data[check_finite] - y_train[check_finite])**2) # by x_train
            # --
            check_finite = np.isfinite(y_val)
            loss_val = np.nanmean((outputs_val.data[check_finite] - y_val[check_finite]) ** 2)  # by x_val
            # --
            loss_train_list.append(loss_train.data)
            loss_val_list.append(loss_val)

            # -- Backward and optimize
            optimizer.zero_grad()
            loss_train.backward()  # compute the gradients of the loss function with respect to the model's parameters
            optimizer.step()  # performs a parameter update based on the computed gradients

            # -- Compute correlation （计算该epoch的相关系数，类似与计算loss，同样忽略有nan的点）
            check_finite = np.isfinite(y_train)
            corr_train_list.append(
                lib_math.corr_n(outputs_train.data[check_finite], y_train[check_finite])[0])  # by x_train
            check_finite = np.isfinite(y_val)
            corr_val_list.append(lib_math.corr_n(outputs_val.data[check_finite], y_val[check_finite])[0])  # by x_val

            # -- Print progress
            if no_epochs >= 1001:
                if epoch % 100 == 0:
                    print(f"Epoch {epoch + 1}/{no_epochs}, Loss_train: {loss_train.item()}, Loss_val: {loss_val}")
                    # for (i, param) in enumerate(model.parameters()):
                    #     # print(param.sum())
                    #     print(param)
                    #     # if i==1:
                    #     #     print(param)
                    #     #     print(param.sum())
            else:
                if epoch % 10 == 0:
                    print(f"Epoch {epoch + 1}/{no_epochs}, Loss_train: {loss_train.item()}, Loss_val: {loss_val}")
        # --
        loss_train_list = np.array(loss_train_list)
        loss_val_list = np.array(loss_val_list)
        corr_train_list = np.array(corr_train_list)
        corr_val_list = np.array(corr_val_list)
        print("corr_train_list-----------",corr_train_list)
        print("corr_val_list-----------", corr_val_list)

        # 保存模型权重
        torch.save(model.state_dict(), model_weights_path)
        print(f"Model weights saved to {model_weights_path}.")


    # ====================================================
    # prediction
    # ====================================================
    # Make predictions
    # pred = model(x_test)
    pred = model(x_test_filled_zeros)
    # --
    check_nan = np.isnan(x_test)
    pred[check_nan] = np.nan

    criterion = torch.nn.MSELoss()
    # ----- 测试数据的预测 -----
    pred_test = model(x_test_filled_zeros)
    # 计算测试集上的损失
    loss_test = criterion(pred_test, y_test)

    # 打印测试损失

    print(f"Test Loss: {loss_test.item()}")
    print(f"Test Loss: {loss_test}")


    # 在需要调用 numpy() 的地方使用 detach() 方法
    df_pred = pd.DataFrame(pred.detach().numpy().flatten(), columns=['after_data'])

    # 定义输出文件路径1
    output_file_path = r'prediction\mean_temp\HKO.csv'

    # 将DataFrame保存为CSV文件
    df_pred.to_csv(output_file_path, index=False, encoding='utf-8-sig')  # 使用utf-8编码保存中文字符

    print(f"输出已保存至 {output_file_path}")


    # =====================================================
    # plots
    # ====================================================
    df_test["cnn_pred"] = np.squeeze(np.array(pred.data))
    plot_year = np.array(df_test.year)
    plot_month = np.array(df_test.month)
    plot_day = np.array(df_test.day)
    plot_obs = np.array(df_test.data)
    plot_ano = np.array(df_test.ano)
    plot_ltm = np.array(df_test.ltm)
    plot_fil = np.array(df_test.fil)
    plot_cnn = np.array(df_test.cnn_pred)

    # # 将plot_fil保存到CSV文件
    # plot_fil_path = 'D:\\PythonProject\\olr.1dcnn_pytorch\\mape2010-2020\\plot_fil\\plot_fil.csv'
    # np.savetxt(plot_fil_path, plot_fil, delimiter=',')
    #
    # # 将plot_cnn保存到CSV文件
    # plot_cnn_path = 'D:\\PythonProject\\olr.1dcnn_pytorch\\mape2010-2020\\plot_cnn\\plot_cnn.csv'
    # np.savetxt(plot_cnn_path, plot_cnn, delimiter=',')

    # ==============================================
    # plot setting (time series)
    # ==============================================
    var = "meantemp"
    dir_out = "./image_mean_temp/"
    os.system("mkdir -p %s" % dir_out)
    fon_prefix = "%s/02-try_%s_%s" % (dir_out, var, station)
    title = "(%s, %s, kernel1=%d, kernel2=%d, no_epochs=%d)" % (station, var, kernel1, kernel2, no_epochs)

    x = np.arange(0, len(plot_fil))
    xmin = x.min()
    xmax = x.max()
    # xticks = np.arange(xmin, xmax+1, 5)
    # # -- y
    # ymax1   = np.nanmax(abs(plot_ano)) + 0.5
    # ymin1   = ymax1 * -1
    if var == "mean_temp":
        ymax1 = 38
        ymin1 = 3
        ymax2 = 8
        ymin2 = ymax2 * -1
    elif var == "max_temp":
        ymax1 = 40
        ymin1 = 5
        ymax2 = 8
        ymin2 = ymax2 * -1
    elif var == "min_temp":
        ymax1 = 35
        ymin1 = 0
        ymax2 = 8
        ymin2 = ymax2 * -1
    else:
        ymax1 = np.nanmax(plot_obs) + 4
        ymin1 = np.nanmin(plot_obs) - 4
        ymax2 = np.nanmax(abs(plot_fil)) + 1
        ymin2 = ymax2 * -1

    title_size = 48  # 32
    tick_size = 40  # 28
    axis_title_size = 44  # title_size

    # ==============================================
    # plot data
    # ==============================================
    # fig, ax = plt.subplots(figsize=(22,10), sharex=True)
    fig, ax = plt.subplots(figsize=(40, 10), sharex=True)
    # fig.subplots_adjust(left=0.1,right=0.82)
    # fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    # fig, ax = plt.subplots(figsize=(30,10), sharex=True)
    ax2 = ax.twinx()

    # ----- line -----
    linewidth = 6
    alpha = 0.7
    # --
    # ax.plot(x, y1*0., color="black", alpha=0.5, linewidth=2)
    # ax2.plot(x, y1*0., color="black", alpha=0.5, linewidth=2)
    # --
    # p1, = ax.plot(x, plot_ano, color="b", alpha=alpha, linewidth=linewidth, label="obs")
    p1, = ax.plot(x, plot_obs, "-", color="b", alpha=alpha, linewidth=linewidth - 2, label="obs")
    p2, = ax.plot(x, plot_ltm, "--", color="b", alpha=alpha - 0.2, linewidth=linewidth - 2, label="ltm")
    p3, = ax2.plot(x, plot_ano, "o-", color="r", alpha=alpha, linewidth=linewidth - 2, label="ano")
    p4, = ax2.plot(x, plot_fil, "o-", color="m", alpha=alpha, linewidth=linewidth, label="fil")
    p5, = ax2.plot(x, plot_cnn, "o-", color="k", alpha=alpha - 0.2, linewidth=linewidth - 3, label="cnn")

    # ----- Title -----
    ax.set_title(title, fontsize=title_size, loc="left", weight="bold")
    ax.set_xlabel(" ", fontsize=axis_title_size, weight="bold")
    ax.set_ylabel("%s (°C)" % var, fontsize=axis_title_size, color=p1.get_color(), weight="bold")
    ax2.set_ylabel("Anomaly (°C)", fontsize=axis_title_size, color=p3.get_color(), weight="bold")

    # ----- axes -----
    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='x', labelsize=tick_size, **tkw)
    ax.tick_params(axis='y', colors=p1.get_color(), labelsize=tick_size, **tkw)
    ax2.tick_params(axis='y', colors=p3.get_color(), labelsize=tick_size, **tkw)
    # -- ticks
    # ax.tick_params(bottom=True, top=False, left=True, right=True)
    # ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=True)
    ax.tick_params(which='both', width=4)
    ax.tick_params(which='major', length=20, color='k')
    ax.tick_params(which='minor', length=10, color='k')
    ax2.tick_params(which='both', width=4)
    ax2.tick_params(which='major', length=20, color='k')
    ax2.tick_params(which='minor', length=10, color='k')
    # -- Hide the right and top spines
    ax.spines['right'].set_linewidth(4)
    ax.spines['top'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.spines['bottom'].set_linewidth(4)

    # ----- tick labels
    if (xmax - xmin) <= 730:
        tick_ind = (plot_day == 1)
        xticks = x[tick_ind]
        xticklabels = []
        for i in xticks:
            if plot_month[i] == 1:
                xticklabels.append("%0.4i-%0.2i-%0.2i" % (plot_year[i], plot_month[i], plot_day[i]))
            else:
                xticklabels.append("%0.2i-%0.2i" % (plot_month[i], plot_day[i]))
    else:
        tick_ind = (plot_month == 1) & (plot_day == 1)
        xticks = x[tick_ind]
        xticklabels = plot_year[tick_ind]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=tick_size)
    # ax.set_yticks(yticks1)
    # ax.set_yticklabels(yticks1, fontsize=tick_size)
    # --
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin1, ymax1)
    ax2.set_ylim(ymin2, ymax2)

    # lines = [p1, p2, p3, p4, p5]
    lines = [p2, p1, p4, p3, p5]
    lines = [p1, p3, p2, p4, p5]
    # loc_legend    = "best"
    loc_legend = "lower center"
    # loc_legend  = "upper left"
    # loc_legend    = "lower left"
    # ax.legend(lines, [l.get_label() for l in lines], fontsize=tick_size, loc=loc_legend)
    # ax.legend(lines, [l.get_label() for l in lines], fontsize=(tick_size-2), loc=loc_legend, ncol=2, borderaxespad=0., handlelength=0.7)
    ax.legend(lines, [l.get_label() for l in lines], loc=loc_legend, ncol=5, handlelength=1.5, labelspacing=0.3,
              handletextpad=0.5, columnspacing=1.0, borderaxespad=0.2, shadow=True, fontsize=(tick_size - 4))

    # ax.legend(fontsize=(tick_size-2), loc=loc_legend, ncol=1)

    plt.savefig(fon_prefix + ".png", bbox_inches='tight', dpi=100)
    print(">>> done plotting " + fon_prefix + ".png")
    # plt.savefig(fon_prefix+".eps", dpi=300)
    # print(">>> done plotting " + fon_prefix + ".eps")
    plt.close()

    # ==============================================
    # plot setting (loss)
    # ==============================================
    dir_out = "./image_mean_temp/"
    os.system("mkdir -p %s" % dir_out)
    fon_prefix = "%s/02-try_loss_%s_%s" % (dir_out, var, station)
    title = "(%s, %s, kernel1=%d, kernel2=%d, no_epochs=%d)" % (station, var, kernel1, kernel2, no_epochs)

    xmin = 0
    xmax = len(loss_train_list)

    title_size = 48  # 32
    tick_size = 40  # 28
    axis_title_size = 44  # title_size

    # ==============================================
    # plot data
    # ==============================================
    fig, ax = plt.subplots(figsize=(40, 10), sharex=True)
    ax2 = ax.twinx()

    # ----- line -----
    linewidth = 6
    alpha = 0.7
    # --
    # ax.plot(x, y1*0., color="black", alpha=0.5, linewidth=2)
    # ax2.plot(x, y1*0., color="black", alpha=0.5, linewidth=2)
    # --
    # p1, = ax.plot(x, plot_ano, color="b", alpha=alpha, linewidth=linewidth, label="obs")
    p1, = ax.plot(loss_train_list, "-", color="b", alpha=alpha, linewidth=linewidth, label="loss_train")
    p2, = ax.plot(loss_val_list, "-", color="r", alpha=alpha, linewidth=linewidth, label="loss_val")
    p3, = ax.plot(corr_train_list, "--", color="b", alpha=alpha, linewidth=linewidth, label="corr_train")
    p4, = ax.plot(corr_val_list, "--", color="r", alpha=alpha, linewidth=linewidth, label="corr_val")

    # ----- Title -----
    ax.set_title(title, fontsize=title_size, loc="left", weight="bold")
    ax.set_xlabel("Epochs", fontsize=axis_title_size, weight="bold")
    ax.set_ylabel("Loss (MSE)", fontsize=axis_title_size, weight="bold")
    ax2.set_ylabel("Correlation", fontsize=axis_title_size, weight="bold")

    # ----- axes -----
    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='x', labelsize=tick_size, **tkw)
    ax.tick_params(axis='y', labelsize=tick_size, **tkw)
    ax2.tick_params(axis='y', labelsize=tick_size, **tkw)
    # ax.tick_params(axis='y', colors=p1.get_color(), labelsize=tick_size, **tkw)
    # -- ticks
    # ax.tick_params(bottom=True, top=False, left=True, right=True)
    # ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=True)
    ax.tick_params(which='both', width=4)
    ax.tick_params(which='major', length=20, color='k')
    ax.tick_params(which='minor', length=10, color='k')
    ax2.tick_params(which='both', width=4)
    ax2.tick_params(which='major', length=20, color='k')
    ax2.tick_params(which='minor', length=10, color='k')
    # -- Hide the right and top spines
    ax.spines['right'].set_linewidth(4)
    ax.spines['top'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.spines['bottom'].set_linewidth(4)

    # ax.set_xticks(xticks)
    # ax.set_xticklabels(xticklabels, fontsize=tick_size)
    # ax.set_yticks(yticks1)
    # ax.set_yticklabels(yticks1, fontsize=tick_size)
    # --
    ax.set_xlim(xmin, xmax)
    # ax.set_ylim(ymin1, ymax1)
    ax.set_yscale('log', base=10)
    ax2.set_yscale('log', base=10)

    lines = [p1, p2, p3, p4]
    loc_legend = "best"
    # loc_legend    = "lower center"
    # loc_legend  = "upper right"
    # loc_legend    = "lower left"
    # ax.legend(lines, [l.get_label() for l in lines], fontsize=tick_size, loc=loc_legend)
    # ax.legend(lines, [l.get_label() for l in lines], fontsize=(tick_size-2), loc=loc_legend, ncol=2, borderaxespad=0., handlelength=0.7)
    ax.legend(lines, [l.get_label() for l in lines], loc=loc_legend, ncol=5, handlelength=1.5, labelspacing=0.3,
              handletextpad=0.5, columnspacing=1.0, borderaxespad=0.2, shadow=True, fontsize=(tick_size - 4))

    # ax.legend(fontsize=(tick_size-2), loc=loc_legend, ncol=1)

    plt.savefig(fon_prefix + ".png", bbox_inches='tight', dpi=100)
    print(">>> done plotting " + fon_prefix + ".png")
    # plt.savefig(fon_prefix+".eps", dpi=300)
    # print(">>> done plotting " + fon_prefix + ".eps")
    plt.close()

    asd = asd


