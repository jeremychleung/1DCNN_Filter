# -*- coding: utf-8 -*-
"""
@author: jer
"""
import sys, os
import numpy as np
import pandas as pd
import xarray as xr
import math as mt
# import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats, interpolate
# from scipy.optimize import curve_fit#做曲线拟合
import metpy.calc as mpcalc
import pymannkendall as mk
import random

# ---------------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# /////////////////////////////////////////////////////////
# ---------------------------------------------------------
# This file (lib_math.py) includes the following functions:
# *****
# 1. general functions 
#	- calc_nansum: same as np.nansum 
#	- calc_nanmean: same as np.nanmean
#	- check_ascending:check whether an array is ascending
# 	- 
# *****
# 2. functions for interpolation or regridding 
# 	- linint_2d: interpolate 2d array
# 	- linint_3d: interpolate 3d array
# 	- nearest_int_1d: Nearest-neighbor interpolation (1d array)
# 	- nearest_int_2d: Nearest-neighbor interpolation (2d array)
# *****
# 3. functions for time series analyses
# 	- linear_trend_2d: calculate linear trend of a 2-d array
# 	- linear_trend_3d: calculate linear trend of a 3-d array
# 	- calc_trend: calculate trend
# 	- calc_nd_run_avg: smoothing - running mean
# 	- calc_nd_run_avg_2d: smoothing - running mean (2d array)
# 	- corr_n: correlation (for n dimension array)
# 	- corr_pattern: pattern correlation (for 2 dimension array only)
# 	- pval_student_t_test: perform student's t-test for linear regression / pearson correlation / arithmetic mean (return p value)
# *****
# 4. functions for grid / gridded data manipulation
# 	- calc_grid_area: calculate area of each grid 
# 	- calc_wgts_mean: calculate 'mean' weighting acording to grid area (for lat-lon grid)
# 	- calc_wgts_eof: calculate 'eof' weighting acording to grid area (for lat-lon grid)
# 	- regrid_latlon_to_equalarea: regirdding data from lat-lon grid to equal-area grid
# *****
# 5. functions for PDF calculations
#   - calc_pdf: calculate pdf of equal-weighted data
# 	- bootstrap_resampling: Doing resampling based on bootstrap method
#   - smooth_bootstrap_resampling: doing resmpaling based on smooth bootsrap method
#   - resampling_kernal_bootstrap_sym: Doing resampling based on kernal bootstrap smoothing and mirroring (symmetrical input)
# *****
# 6. functions for smoothing / filtering
#   - smth9_ncl: spatial smoothing (9-point)
# *****
# 7. functions for numerical simulation
#   - dfdt_cdf: 2nd order central finite difference method: solving d(f)/dt
#   - dfdx_cdf: 2nd order central finite difference method: solving d(f)/dx 
#   - dfdy_cdf: 2nd order central finite difference method: solving d(f)/dy
# *****
# ---------------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# /////////////////////////////////////////////////////////
# ---------------------------------------------------------


#********************************************************
#********************************************************
#********************************************************
# ----- general functions -----
#********************************************************
#********************************************************
#********************************************************

#==================================================
# function: same as np.nansum 
# input: calc_nansum(data, **kwargs)
# output: return(data)
# -----
# (np.nansum -> 当数组所有数值都是nan的时候，返回0)
# (func.calc_nansum -> 当数组所有数值都是nan的时候，返回nan)
# ***** 当数组所有数值都是nan的时候，返回nan *****
#==================================================
def calc_nansum(data, **kwargs):
	# return(np.nanmean(data, **kwargs) * np.isfinite(data).sum(**kwargs))
	return((np.nanmean(data, **kwargs) * np.isfinite(data).sum(**kwargs)).astype("float32"))
	# if np.isnan(data).all():
	#   return np.nan
	# else:
	#   return np.nansum(a, **kwargs) 

#==================================================
# function: same as np.nanmean
# (the function indeed does not ignore nan values, so you will still get nan when there are nan values in the input)
# input: calc_nanmean(data, axis)
# output: return(data)
# -----
#==================================================
def calc_nanmean(data, axis=0):
	dims_data 	= data.shape
	dims_tmp 	= np.delete(dims_data, axis)
	tmp 		= np.zeros(dims_tmp)
	for i in range(0,dims_data[axis]):
		if axis==0:
			# tmp = tmp + data[i,...]
			tmp = calc_nansum([tmp,data[i,...]],axis=0) 
		elif axis==1:
			# tmp = tmp + data[:,i,...]
			tmp = calc_nansum([tmp,data[:,i,...]],axis=0) 
		elif axis==2:
			# tmp = tmp + data[:,:,i,...]
			tmp = calc_nansum([tmp,data[:,:,i,...]],axis=0) 
		elif axis==3:
			# tmp = tmp + data[:,:,:,i,...]
			tmp = calc_nansum([tmp,data[:,:,:,i,...]],axis=0) 
		elif axis==4:
			# tmp = tmp + data[:,:,:,:,i,...]
			tmp = calc_nansum([tmp,data[:,:,:,:,i,...]],axis=0) 
	tmp = tmp / dims_data[axis]
	return(tmp)

#==================================================
# function: check whether an array is ascending
# input: linint_2d(data_old, lat_old, lon_old, lat_new, lon_new, kind)
# output: return(area_list)
# -----
#==================================================
def check_ascending(data_check):
	data_check = list(data_check)
	if data_check == sorted(data_check,reverse=False):
		return True
	elif data_check == sorted(data_check,reverse=True):
		return False


#********************************************************
#********************************************************
#********************************************************
# ----- functions for interpolation or regridding -----
#********************************************************
#********************************************************
#********************************************************

#==================================================
# functions: interpolate 2d array
# input: linint_2d(data_old, lat_old, lon_old, lat_new, lon_new, kind)
# output: return(data_new)
# -----
# ** lat_old, lon_old, lat_new, lon_new MUST BE in ascending order
#==================================================
def linint_2d(data_old, lat_old, lon_old, lat_new, lon_new, opt_kind="linear"):
	if check_ascending(lat_old)==False:
		# raise ValueError('[ERROR]: order of lat_old is not ascending (please check)') 
		f = interpolate.interp2d(lon_old, lat_old[::-1], data_old[::-1,:], kind=opt_kind)
		data_new = f(lon_new, lat_new)
	if check_ascending(lon_old)==False:
		# raise ValueError('[ERROR]: order of lon_old is not ascending (please check)') 
		f = interpolate.interp2d(lon_old[::-1], lat_old, data_old[:,::-1], kind=opt_kind)
		data_new = f(lon_new, lat_new)
	if check_ascending(lat_new)==False:
		raise ValueError('[ERROR]: order of lat_new is not ascending (please check)') 
	if check_ascending(lon_new)==False:
		raise ValueError('[ERROR]: order of lon_new is not ascending (please check)') 
	else:
		f = interpolate.interp2d(lon_old, lat_old, data_old, kind=opt_kind)
		data_new = f(lon_new, lat_new)
	return(data_new)

#==================================================
# functions: interpolate 3d array
# input: linint_3d(data_old, lat_old, lon_old, lat_new, lon_new, opt_kind)
# output: return(data_new)
# -----
# ** 1. lat_old, lon_old, lat_new, lon_new MUST BE in ascending order
# ** 2. first dimension of data_old MUST BE time dimension
#==================================================
def linint_3d(data_old, lat_old, lon_old, lat_new, lon_new, opt_kind='linear'):
	data_new = np.zeros([data_old.shape[0], len(lat_new), len(lon_new)]) + np.nan
	for i in range(0, data_new.shape[0]):
		data_new[i,:,:] = linint_2d(data_old[i,:,:], lat_old, lon_old, lat_new, lon_new, opt_kind)
	return(data_new)


# ************************************************
# function: Nearest-neighbor interpolation (1d array)
# input: nearest_int_1d(data_old, lat_old, lon_old, lat_new, lon_new)
# output: return(data_new)
# ************************************************
def nearest_int_1d(data_old, lat_old, lon_old, lat_new, lon_new):
	if 1==0:
		asd=asd
	# if check_ascending(lat_old)==False:
	# 	# raise ValueError('[ERROR]: order of lat_old is not ascending (please check)') 
	# 	f = interpolate.NearestNDInterpolator(list(zip(lon_old, lat_old[::-1])), data_old[::-1,:])
	# 	data_new = f(lon_new, lat_new)
	# if check_ascending(lon_old)==False:
	# 	# raise ValueError('[ERROR]: order of lon_old is not ascending (please check)') 
	# 	f = interpolate.NearestNDInterpolator(list(zip(lon_old[::-1], lat_old)), data_old[:,::-1])
	# 	data_new = f(lon_new, lat_new)
	# if check_ascending(lat_new)==False:
	# 	raise ValueError('[ERROR]: order of lat_new is not ascending (please check)') 
	# if check_ascending(lon_new)==False:
	# 	raise ValueError('[ERROR]: order of lon_new is not ascending (please check)') 
	else:
		# print(lon_old.shape)
		# print(lat_old.shape)
		# print(data_old.shape)
		f = interpolate.NearestNDInterpolator(list(zip(lon_old, lat_old)), data_old)
		# print(lon_new.shape)
		# print(lat_new.shape)
		# X, Y = np.meshgrid(lon_new, lat_new)
		# print(X.shape)
		# print(Y.shape)
		# data_new = f(X, Y)
		data_new = f(lon_new, lat_new)
		# data_new = f(lat_new, lon_new)
	return(data_new)

# ************************************************
# function: Nearest-neighbor interpolation (2d array)
# !! first dimension must be time !!
# --
# input: nearest_int_2d(data_old, lat_old, lon_old, lat_new, lon_new)
# output: return(data_new)
# ************************************************
def nearest_int_2d(data_old, lat_old, lon_old, lat_new, lon_new):
	if len(data_old.shape)==2:
		pass
	else:
		print(">>> only support 2-dimensional array")
		asd=asd

	data_new = np.zeros([data_old.shape[0], len(lat_new)]) + np.nan
	for i in range(0, data_old.shape[0]):
		print(">>> nearest_int_2d on i=%d/%d" % (i, data_old.shape[0]))
		data_old_tmp = data_old[i,:] + 0
		data_new_tmp = nearest_int_1d(data_old_tmp, lat_old, lon_old, lat_new, lon_new)
		data_new[i,:] = data_new_tmp + 0

	return(data_new)

#********************************************************
#********************************************************
#********************************************************
# ----- functions for time series analyses -----
#********************************************************
#********************************************************
#********************************************************

#==================================================
# functions: calculate linear trend of a 2-d array
# input: linear_trend_2d(fdata, time)
# output: return(slope_list, intercept_list, pval_list)
# -----
# fdata = 2d array (time x grid)
# t = 1d array (time)
#==================================================
def linear_trend_2d(fdata, time):
	slope_list 	= []
	intercept_list 	= []
	pval_list 	= []
	for i in range(0,fdata.shape[1]):
		ind 	= np.isfinite(fdata[:,i])
		x_tmp 	= time[ind] + 0
		y_tmp 	= fdata[ind,i] + 0
		# --
		# slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(time,fdata[:,i])
		if len(x_tmp)>1:
			slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x_tmp,y_tmp)
			slope_list.append(slope1)
			intercept_list.append(intercept1)
			pval_list.append(p_value1)
		else:
			slope_list.append(np.nan)
			intercept_list.append(np.nan)
			pval_list.append(np.nan)
	slope_list = np.array(slope_list)
	intercept_list = np.array(intercept_list)
	pval_list = np.array(pval_list)
	return(slope_list, intercept_list, pval_list)

#==============================================
# functions: calculate linear trend of a 3-d array
# input: linear_trend_2d(fdata, time)
# output: return(slope_list, pval_list)
# -----
# fdata = 3d array (time x lat x lon)
# time = 1d array (time)
#==============================================
def linear_trend_3d(fdata, time):
	slope_list  = np.zeros([fdata.shape[1],fdata.shape[2]]) + np.nan
	intercept_list   = np.zeros([fdata.shape[1],fdata.shape[2]]) + np.nan
	pval_list   = np.zeros([fdata.shape[1],fdata.shape[2]]) + np.nan
	for i in range(0,fdata.shape[1]):
		for j in range(0,fdata.shape[2]):
			ind 	= np.isfinite(fdata[:,i,j])
			x_tmp 	= time[ind] + 0
			y_tmp 	= fdata[ind,i,j] + 0
			# --
			if len(x_tmp)>1:
				slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x_tmp,y_tmp)
				slope_list[i,j] = slope1 + 0
				intercept_list[i,j] = intercept1 + 0
				pval_list[i,j] = p_value1 + 0
			else:
				slope_list[i,j] = np.nan
				intercept_list[i,j] = np.nan
				pval_list[i,j] = np.nan
	return(slope_list, intercept_list, pval_list)

#==================================================
# functions: calculate Theil-Sen trend of a 2-d array
# input: theil_sen_trend_2d(fdata, time)
# output: return(slope_list, pval_list)
# -----
# fdata = 2d array (time x grid)
# t = 1d array (time)
#==================================================
def theil_sen_trend_2d(fdata, time, alpha=0.05):
	slope_list 	= []
	intercept_list 	= []
	pval_list 	= []
	istrend_list 	= []
	for i in range(0,fdata.shape[1]):
		# ind 	= np.isfinite(fdata[:,i])
		# x_tmp 	= time[ind] + 0
		# y_tmp 	= fdata[ind,i] + 0
		# --
		x_tmp 	= time + 0
		y_tmp 	= fdata[:,i] + 0
		# --
		if len(x_tmp)>1:
			# slope1, intercept1, conf90_1, conf90_2 = stats.theilslopes(y_tmp, x_tmp, 0.90) #, method='separate')
			mk_test1 = mk.original_test(y_tmp, alpha)
			slope_list.append(mk_test1.slope)
			intercept_list.append(mk_test1.intercept)
			pval_list.append(mk_test1.p)
			istrend_list.append(mk_test1.trend)
		else:
			slope_list.append(np.nan)
			intercept_list.append(np.nan)
			pval_list.append(np.nan)
			istrend_list.append(np.nan)
	slope_list = np.array(slope_list)
	intercept_list = np.array(intercept_list)
	pval_list = np.array(pval_list)
	istrend_list = np.array(istrend_list)
	return(slope_list, intercept_list, pval_list, istrend_list)

#==============================================
# functions: calculate Theil-Sen trend of a 3-d array
# input: theil_sen_trend_3d(fdata, time)
# output: return(slope_list, pval_list)
# -----
# fdata = 3d array (time x lat x lon)
# time = 1d array (time)
#==============================================
def theil_sen_trend_3d(fdata, time, alpha=0.05):
	slope_list  = np.zeros([fdata.shape[1],fdata.shape[2]]) + np.nan
	intercept_list   = np.zeros([fdata.shape[1],fdata.shape[2]]) + np.nan
	pval_list   = np.zeros([fdata.shape[1],fdata.shape[2]]) + np.nan
	istrend_list 	= np.zeros([fdata.shape[1],fdata.shape[2]]).astype(str)
	for i in range(0,fdata.shape[1]):
		for j in range(0,fdata.shape[2]):
			# ind 	= np.isfinite(fdata[:,i,j])
			# x_tmp 	= time[ind] + 0
			# y_tmp 	= fdata[ind,i,j] + 0
			# --
			x_tmp 	= time[:] + 0
			y_tmp 	= fdata[:,i,j] + 0
			# --
			if len(x_tmp)>1:
				# slope1, intercept1, conf90_1, conf90_2 = stats.theilslopes(y_tmp, x_tmp, 0.90) #, method='separate')
				mk_test1 = mk.original_test(y_tmp, alpha)
				slope_list[i,j] = mk_test1.slope + 0
				intercept_list[i,j] = mk_test1.intercept + 0
				pval_list[i,j] = mk_test1.p + 0
				istrend_list[i,j] = mk_test1.trend 
			else:
				slope_list[i,j] = np.nan
				intercept_list[i,j] = np.nan
				pval_list[i,j] = np.nan
				istrend_list[i,j] = np.nan
	return(slope_list, intercept_list, pval_list, istrend_list)

#==================================================
# function: calculate trend
# input: linint_2d(data_old, lat_old, lon_old, lat_new, lon_new, kind)
# output: return(area_list)
# -----
#==================================================
def calc_trend(sst_pfd, ftime):
	trend, pval = linear_trend_2d(sst_pfd, ftime) # unit: % / year
	trend_norm 	= trend / sst_pfd.mean(axis=0) * 100 # normalized trend

	# convert unit
	# trend 		= trend * 10 # unit: % / decade
	# trend_norm 	= trend_norm * 10 # unit: % / decade

	# set 0
	trend[sst_pfd.sum(axis=0)==0] = 0
	trend_norm[sst_pfd.sum(axis=0)==0] = 0
	return(trend, trend_norm, pval)

#==============================================
# smoothing: running mean
#==============================================
def calc_nd_run_avg(data, npoint=3, axis=0, opt_keep_begin_end=False):
	nday_half = int((npoint-1)/2)

	# dim_data_tmp = [data.shape[i] if (i>-1) else nday for i in range(-1,len(data.shape))]
	# data_tmp = np.zeros(dim_data_tmp) + np.nan
	data_tmp = np.zeros([npoint,len(data)]) + np.nan
	k=0
	for i in range(0,nday_half+1):
		if i==0:
			data_tmp[k,:] = data[:]
			k=k+1
		else:
			data_tmp[k,:-1*i] = data[i:]
			k=k+1
			data_tmp[k,i:] = data[:-1*i]
			k=k+1

	if opt_keep_begin_end==True:
		data_smth = np.nanmean(data_tmp, axis=0)
	elif opt_keep_begin_end==False:
		data_smth = np.mean(data_tmp, axis=0)
	return(data_smth)

#==============================================
# smoothing: running mean
#==============================================
def calc_nd_run_avg_2d(data, npoint=3, axis=0, opt_keep_begin_end=False):
	nday_half = int((npoint-1)/2)

	# dim_data_tmp = [data.shape[i] if (i>-1) else nday for i in range(-1,len(data.shape))]
	# data_tmp = np.zeros(dim_data_tmp) + np.nan
	dims = [npoint]
	for i in range(0,len(data.shape)):
		dims.append(data.shape[i])
	data_tmp = np.zeros(dims) + np.nan

	if axis!=0:
		asd=asd

	k=0
	for i in range(0,nday_half+1):
		if i==0:
			data_tmp[k,:,...] = data[:,...]
			k=k+1
		else:
			data_tmp[k,:-1*i,...] = data[i:,...]
			k=k+1
			data_tmp[k,i:,...] = data[:-1*i,...]
			k=k+1

	if opt_keep_begin_end==True:
		data_smth = np.nanmean(data_tmp, axis=0)
	elif opt_keep_begin_end==False:
		data_smth = np.mean(data_tmp, axis=0)
	return(data_smth)

#==============================================
# correlation (for n dimension array)
# parameter:
# a = input data 1
# b = input data 2
# axis = time dimension
#==============================================
def corr_n(a, b, axis=0, opt_save_memory=False):
	# -------------------------------
	# remove nan
	# -------------------------------
	if len(a.shape)==1:
		ind_isfinite = np.isfinite(a) & np.isfinite(b)
		a = a[ind_isfinite]
		b = b[ind_isfinite]
	else:
		if (np.isnan(a).sum()>0) | (np.isnan(b).sum()>0):
			print("!! ERROR: missing values !!")
			asd=asd

	# -------------------------------
	# 计算相关系数
	# -------------------------------
	if opt_save_memory==False:
		# 分子
		corr_num_a  = a - np.expand_dims(np.nanmean(a, axis=axis), axis=axis)
		corr_num_b  = b - np.expand_dims(np.nanmean(b, axis=axis), axis=axis)

		# 分母
		corr_den_a  = np.sqrt(np.nansum((a - np.expand_dims(np.nanmean(a, axis=axis), axis=axis))**2, axis=axis)) 
		corr_den_b  = np.sqrt(np.nansum((b - np.expand_dims(np.nanmean(b, axis=axis), axis=axis))**2, axis=axis)) 

		# 相关系数
		corr    = np.nansum(corr_num_a*corr_num_b, axis=axis) / ( corr_den_a*corr_den_b )
	elif opt_save_memory==True:
		# 分子
		corr_num_a  = a - np.expand_dims(calc_nanmean(a, axis=axis), axis=axis)
		corr_num_b  = b - np.expand_dims(calc_nanmean(b, axis=axis), axis=axis)

		# 分母
		corr_den_a  = np.sqrt(np.nansum((a - np.expand_dims(calc_nanmean(a, axis=axis), axis=axis))**2, axis=axis)) 
		corr_den_b  = np.sqrt(np.nansum((b - np.expand_dims(calc_nanmean(b, axis=axis), axis=axis))**2, axis=axis)) 

		# 相关系数
		corr    = calc_nansum(corr_num_a*corr_num_b, axis=axis) / ( corr_den_a*corr_den_b )

	# -------------------------------
	# 检验
	# two-sided student t test
	# -------------------------------
	n       = a.shape[axis]
	tval    = corr * np.sqrt( (n-2) / (1 - corr**2) )
	pval    = stats.t.sf(np.abs(tval), (n-2))*2  # two-sided pvalue = Prob(abs(t)>tt)
	return(corr, pval)

#==============================================
# Pattern correlation (for 2 dimension array)
# parameter:
# a = input data 1 (2-dimensional)
# b = input data 2 (2-dimensional)
# axis = time dimension
#==============================================
def corr_pattern(a, b, lat, lon, opt_save_memory=False):
	if len(a.shape)!=2:
		asd=sad
	if len(b.shape)!=2:
		asd=sad
	# -------------------------------
	# weights
	# -------------------------------
	wgts_mean = calc_wgts_mean(lat, a[:,:])
	# -------------------------------
	# reshape
	# -------------------------------
	a = a.reshape(-1)
	b = b.reshape(-1)
	wgts_mean = wgts_mean.reshape(-1)

	# -------------------------------
	# remove nan
	# -------------------------------
	if len(a.shape)==1:
		ind_isfinite = np.isfinite(a) & np.isfinite(b)
		a = a[ind_isfinite]
		b = b[ind_isfinite]
		wgts_mean = wgts_mean[ind_isfinite]
	else:
		if (np.isnan(a).sum()>0) | (np.isnan(b).sum()>0):
			print("!! ERROR: missing values !!")
			asd=asd

	# -------------------------------
	# 计算相关系数
	# -------------------------------
	if opt_save_memory==False:
		# 分子
		corr_num_a  = a - np.nansum(a*wgts_mean) / np.nansum(wgts_mean)
		corr_num_b  = b - np.nansum(b*wgts_mean) / np.nansum(wgts_mean)
		# corr_num_a  = a - np.expand_dims(np.nanmean(a, axis=axis), axis=axis)
		# corr_num_b  = b - np.expand_dims(np.nanmean(b, axis=axis), axis=axis)

		# 分母
		corr_den_a  = np.sqrt(np.nansum(wgts_mean * (a - np.nansum(a*wgts_mean) / np.nansum(wgts_mean))**2)) 
		corr_den_b 	= np.sqrt(np.nansum(wgts_mean * (b - np.nansum(b*wgts_mean) / np.nansum(wgts_mean))**2)) 
		# corr_den_a  = np.sqrt(np.nansum((a - np.expand_dims(np.nanmean(a, axis=axis), axis=axis))**2, axis=axis)) 
		# corr_den_b  = np.sqrt(np.nansum((b - np.expand_dims(np.nanmean(b, axis=axis), axis=axis))**2, axis=axis)) 

		# 相关系数
		corr    = np.nansum(wgts_mean*corr_num_a*corr_num_b) / ( corr_den_a*corr_den_b )
	elif opt_save_memory==True:
		asd=asd
		# 分子
		corr_num_a  = a - np.expand_dims(calc_nanmean(a, axis=axis), axis=axis)
		corr_num_b  = b - np.expand_dims(calc_nanmean(b, axis=axis), axis=axis)

		# 分母
		corr_den_a  = np.sqrt(np.nansum((a - np.expand_dims(calc_nanmean(a, axis=axis), axis=axis))**2, axis=axis)) 
		corr_den_b  = np.sqrt(np.nansum((b - np.expand_dims(calc_nanmean(b, axis=axis), axis=axis))**2, axis=axis)) 

		# 相关系数
		corr    = calc_nansum(corr_num_a*corr_num_b, axis=axis) / ( corr_den_a*corr_den_b )

	# -------------------------------
	# 检验
	# two-sided student t test
	# -------------------------------
	n       = len(a)
	tval    = corr * np.sqrt( (n-2) / (1 - corr**2) )
	pval    = stats.t.sf(np.abs(tval), (n-2))*2  # two-sided pvalue = Prob(abs(t)>tt)
	return(corr, pval)


#==============================================
# perform student's t-test for linear regression / pearson correlation / arithmetic mean (return p value)
# parameter:
# opt_coefficient = linregress / pearson_r
#==============================================
def pval_student_t_test(x, y, opt_coefficient="linregress"):

	if opt_coefficient in ["linregress"]:
		# -- linear regression
		slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x,y)
		# -- calc degree of freedom
		n 		= len(y)
		dof 	= n - 2
		# -- calc pval
		tval 	= (slope1 - 0) / std_err1
		pval 	= stats.t.sf(np.abs(tval), dof)*2  # two-sided pvalue = Prob(abs(t)>tt)
		print(pval)
		print(p_value1)

	elif opt_coefficient in ["pearson_r"]:
		# -- calc pearson correlation
		corr1, pval1 = corr_n(x,y)
		print(corr_n(x,y))
		# -- calc degree of freedom
		n 		= len(y)
		dof 	= (n - 2) 
		# -- calc pval
		tval    = corr1 * np.sqrt( (n-2) / (1 - corr1**2) )
		pval    = stats.t.sf(np.abs(tval), dof)*2  # two-sided pvalue = Prob(abs(t)>tt)
		print(pval)
		print(stats.pearsonr(x, y))

	return(pval)

#********************************************************
#********************************************************
#********************************************************
# ----- functions for grid / gridded data manipulation -----
#********************************************************
#********************************************************
#********************************************************

#==================================================
# functions: calculate area of each grid 
# input: calc_grid_area(lon,lat)
# output: return(area_list) (unit: km**2)
# -----
#==================================================
def calc_grid_area(lon,lat):    
	#----- delta x/y (in terms of distance, unit=m)
	dy,dx = mpcalc.lat_lon_grid_deltas(lon,lat) #dy为纬度，dx为经度
	dy,dx = abs(dy.m),abs(dx.m) #.m 代表单位取米
	dx = np.concatenate((dx,dx[0,:][None,:]),axis=0)
	dy = np.concatenate((dy,dy[:,0][:,None]),axis=1)
	#----- area (unit=km**2)
	area_list = dx*dy/1000./1000.
	return(area_list)

#==================================================
# functions: calculate 'mean' weighting according to grid area (for lat-lon grid)
# input: calc_wgts_mean(lat,data)
# output: return(wgts_mean)
# -----
#==================================================
def calc_wgts_mean(lat,data):    
	#----- weighting for eof / mean -----
	# -- cosine
	wgts_lat = np.cos(lat / 180 * np.pi)
	wgts_lat[wgts_lat<0] = 0
	# # -- square-root
	# wgts_lat = np.sqrt(wgts_lat)
	# -- weights-mean
	wgts_mean = data*0 + wgts_lat[:,None]
	return(wgts_mean)

#==================================================
# functions: calculate eof weighting according to grid area (for lat-lon grid)
# input: calc_wgts_eof(lat,data)
# output: return(wgts_eof)
# -----
#==================================================
def calc_wgts_eof(lat,data):    
	#----- weighting for eof / mean -----
	# -- cosine
	wgts_lat = np.cos(lat / 180 * np.pi)
	wgts_lat[wgts_lat<0] = 0
	# # -- square-root
	# wgts_lat = np.sqrt(wgts_lat)
	# -- weights-mean
	wgts_mean = data*0 + wgts_lat[:,None]
	# -- weights-eof
	wgts_eof = np.sqrt(wgts_mean)
	return(wgts_eof)

# ============================================================
# function: define new grid (equal-area grid)
# input: define_EA_grid_area(Dlat, Dlon, ref_lat=0)
# Dlat = latitude spacing
# Dlon = longitude spacing
# ref_lat = the latitude where we define the new grid box 
# (ref_lat=0 --> equator --> larger grid box)
# (ref_lat=90 --> north/south pole --> smallest grid box)
# ============================================================
def define_EA_grid_area(Dlat, Dlon, ref_lat=0):
	R0      = 6377.83 # 地球赤道半径，小圆台下表面的半径
	R1      = 6371.39 # 地球平均半径
	R2      = 6356.91 # 极地半径

	pi      = mt.pi
	Rad0    = (pi * Dlat) / 180.0 # 角度转弧度制
	Rad1    = mt.atan(R2**2 / (R0**2) * mt.tan(Rad0)) # 地理坐标转为地心坐标 

	h0      = R1 * mt.sin(Rad1) # 求小圆台的高
	r0      = R1 * mt.cos(Rad1) # 小台柱上表面的半径
	L0      = mt.sqrt(h0**2 + (R1 - r0)**2) # 小圆台的母线

	s0 	= pi * L0 * (R1+r0) # 小圆台的侧面积
	ds0 = (s0) * Dlon / 360.0 # 等面积切分 (area of each grid cell)
	return(s0, ds0)

# ============================================================
# function: define new grid (equal-area grid)
# 计算全球的格子数
# ============================================================
def define_EA_grid_number(flat, flon, Dlat, Dlon, ds0, fgrid_type="lat-lon"):
	R0 = 6377.83 # 地球赤道半径，小圆台下表面的半径
	R1 = 6371.39 # 地球平均半径
	R2 = 6356.91 # 极地半径

	# ----- 确定每层的格子数 ----- 
	if fgrid_type=="gaussian":
		# ----------------------
		# check whether 0 degree north exists in flat
		if (flat==0).sum()>0:
			print("!! 0 degree north exists in flat !! (check 'flat' and 'fgrid_type')")
			asd=asd
		# ----------------------

		Mlon 	= 360.0 / Dlon
		N 		= []
		NN 		= []
		NS 		= []
		for i in range(0,int(len(flat)/2)): #纬带切割
			# Mlata 	= flat[i] # 上
			# Mlatb 	= flat[i+1] # 下
			# Mlata 	= flat[i] - Dlat/2. # flat[i] # 上
			# Mlatb 	= flat[i] + Dlat/2. # flat[i+1] # 下
			if i==0:
				Mlata   = flat[i] - (Dlat/2.) #-90 # 上
				Mlatb   = (flat[i] + flat[i+1])/2 # 下
			# elif i==int(len(lat)/2-1):
			# 	Mlata   = (flat[i] + flat[i-1])/2 # 下
			# 	Mlatb   = 0 # 下
			else:
				Mlata   = (flat[i] + flat[i-1])/2 # 上
				Mlatb   = (flat[i] + flat[i+1])/2 # 下
				if i == int(len(flat)/2-1):
					Mdiff = Mlata - Mlatb
					print(">>> %d. Mlata=%0.2f, Mlatb=%0.2f, diff=%0.2f" % (i, Mlata, Mlatb, Mdiff))
					if Mlatb>1e-10:
						asd=asd

			# ----- 弧度转换 -----
			Rada    = (2.0 * mt.pi * Mlata) / 360.0 
			Radb    = (2.0 * mt.pi * Mlatb) / 360.0

			# ----- 地理纬度修正成地心纬度 -----
			Radah   = mt.atan(mt.tan(Rada) * R2**2 / R0**2) 
			Radbh   = mt.atan(mt.tan(Radb) * R2**2 / R0**2)

			ha      = R1 * mt.sin(Radah) #较长
			hb      = R1 * mt.sin(Radbh) #较短

			ra      = R1 * mt.cos(Radah) # 计算上底半径
			rb      = R1 * mt.cos(Radbh) # 计算底半径

			L       = mt.sqrt((hb - ha)**2 + (rb - ra)**2) # 母线

			s1      = mt.pi * L * (ra + rb) # 计算纬带的面积
			n       = s1 / ds0 # 计算该层有多少个格子
			#print(s1)

			NS.append(mt.ceil(n)) # 把格子数信息储存进列表
			NN.append(mt.ceil(n))

		NS.reverse()
		N = NN + NS#一个包含每个纬带格子数的列表

		# ----- added by jer -----
		N = np.array(N) # 把上面算好的 N，从 list换成array
		N_df = pd.DataFrame(np.transpose([N, flat]), columns=["N", "lat"]) 
		# ------------------------

	# ====================================
	# 等经纬度网格
	# ====================================
	elif fgrid_type=="lat-lon":
		# ----------------------
		# check whether 0 degree north exists in flat
		if (flat==0).sum()==0:
			print("!! 0 degree north does exist in flat !! (check 'flat' and 'fgrid_type')")
			asd=asd
		# ----------------------

		Mlat 	= 180.0 / Dlat
		Mlon 	= 360.0 / Dlon
		Mlat2 	= np.round(Mlat / 2)
		N 		= []
		NN 		= []
		NS 		= []
		MlatN 	= []
		MlatS 	= []
		for i in range(0,int(Mlat2),1):
			# ----- 切割纬带 -----
			# Mlata   = (lat - 1) * Dlat 
			# Mlatb   = Mlata + Dlat
			Mlata   = flat[i] + 0 # 上
			Mlatb   = flat[i+1] + 0 # 下
			if i == int(Mlat2-1):
				Mdiff = Mlata - Mlatb
				print(">>> %d. Mlata=%0.2f, Mlatb=%0.2f, diff=%0.2f" % (i, Mlata, Mlatb, Mdiff))
				# if Mlatb!=0:
				# 	asd=asd
			# print(">>> %d. Mlata=%0.2f, Mlatb=%0.2f, diff=%0.2f" % (i, Mlata, Mlatb, Mdiff))

			# ----- 弧度转换 -----
			Rada 	= (2.0 * mt.pi * Mlata) / 360.0 
			Radb 	= (2.0 * mt.pi * Mlatb) / 360.0

			# ----- 地理纬度修正成地心纬度 -----
			Radah 	= mt.atan(mt.tan(Rada) * R2**2 / R0**2) 
			Radbh 	= mt.atan(mt.tan(Radb) * R2**2 / R0**2)

			ha 		= R1 * mt.sin(Radah)
			hb 		= R1 * mt.sin(Radbh)
			    
			ra 		= R1 * mt.cos(Radah) # 计算下底半径
			rb 		= R1 * mt.cos(Radbh) # 计算上底半径

			L 		= mt.sqrt((hb - ha)**2 + (rb - ra)**2) # 母线

			s1 		= mt.pi * L * (ra + rb) # 计算纬带的面积
			n 		= s1 / ds0 # 计算该层有多少个格子
			#print(s1)

			NS.append(mt.ceil(n)) # 把格子数信息储存进列表
			NN.append(mt.ceil(n))
			MlatS.append( (Mlata+Mlatb)/-2 )
			MlatN.append( (Mlata+Mlatb)/2 )

		NS.reverse()
		N = NN + NS#一个包含每个纬带格子数的列表
		MlatS.reverse()
		MlatN = MlatN + MlatS #一个包含每个纬带格子数的列表

		# ----- added by jer -----
		N = np.array(N) # 把上面算好的 N，从 list换成array
		MlatN = np.array(MlatN) # 把上面算好的 N，从 list换成array
		N_df = pd.DataFrame(np.transpose([N, MlatN]), columns=["N", "lat"]) 
		# ------------------------

	return(N, N_df)

# ============================================================
# function: 转换数据
# transfrom data from "gaussian grid" / "lat-lon grid" to "equal-area grid"
# ============================================================
def transform_data_to_equal_area_grid(fdata_new, ftime, flat_new, flon_new, N, Dlat, Dlon, opt_allow_nan=False):
	# -------------------------------------------------
	# 建立空的数组，for输出数据,每进入一个时间（k）的循环清空一次 
	# -------------------------------------------------
	data_EA = np.zeros([len(ftime), N.sum()]) + np.nan # 建立新的数组，预备放入equal-area 的sst 数据 
	lat_EA = np.zeros([N.sum()]) + np.nan
	lon_EA = np.zeros([N.sum()]) + np.nan

	# -------------------------------------------------
	#----- 开始循环计算每个格子的数据值 -----
	# -------------------------------------------------
	d = 0
	for i_lat in range(0, len(flat_new)): # lat序列，分辨率为0.25时，总长度为720
		print(">>> regridding data: ilat=%d(%0.2f)" % (i_lat, flat_new[i_lat]))
		step = 360 / N[i_lat]   # 每层的步长 (amended by jer)
		elat = flat_new[i_lat]
		# print("%d. lat=%0.3f N=%d step=%0.3f " % (d, flat_new[i_lat], N[i_lat], step))

		for i_N in range(1, N[i_lat]+1): #每个纬带上的N序列，随i的改变而改变
			elon1 = (i_N - 1) * step # 格子左边界
			elon2 = (i_N) * step  # 格子右边界
			print("%d. lat=%0.2f lon1=%0.2f lon2=%0.2f lon=%0.2f " % (d, flat_new[i_lat], elon1, elon2, lon_EA[d]))

			if elon2 > (360.0 + Dlon): 
				raise ValueError("[ERROR] elon2 is larger than %0.3fE" % (360.+Dlon))  # 报错

			# -------------------------------------------------
			# 对 elon1 和 elon2 之间的所有数值做平均
			# 赋值到sst_EA 里
			# ------------------------------------------------
			lon_ind = (flon_new>=elon1) & (flon_new<=elon2)
			lon_ind1 = np.argmin(abs(flon_new-elon1))
			lon_ind2 = np.argmin(abs(flon_new-elon2))
			# --
			# data_EA[:, d] 	= np.nanmean(fdata_new[:,i_lat,:][:,lon_ind], axis=1)
			data_EA[:, d] 	= np.nanmean(fdata_new[:,i_lat,:][:,lon_ind1:lon_ind2+1], axis=1)
			# --
			lat_EA[d] 	= elat
			lon_EA[d] 	= (elon1 + elon2) / 2
			# --
			if (np.isnan(data_EA[:, d]).sum()>0):
				print("!! missing value !!")
				print("%d. lat=%0.2f lon1=%0.2f lon2=%0.2f lon=%0.2f " % (d, flat_new[i_lat], elon1, elon2, lon_EA[d]))
				if opt_allow_nan==False:
					asdas=dasd
			# print("%d. lat=%0.2f lon1=%0.2f lon2=%0.2f lon=%0.2f " % (d, flat_new[i_lat], elon1, elon2, lon_EA[d]))
			d   = d + 1

	# -------------------------------------------------
	# check if any missing value
	# -------------------------------------------------
	if (np.isnan(data_EA).sum()>0) | (np.isnan(lat_EA).sum()>0) | (np.isnan(lon_EA).sum()>0):
		if opt_allow_nan==False:
			print("!! missing value found !!")
			asdas=dasd

	return(data_EA, lat_EA, lon_EA)

#==================================================
# functions: regirdding data from lat-lon grid to equal-area grid
# input: regrid_latlon_to_equalarea(data)
# output: return(data_ea, lat_ea, lon_ea)
# -----
#==================================================
def regrid_latlon_to_equalarea(fdata, flat, flon, ftime, fgrid_type="lat-lon", opt_grid_size="max"):    
	# ============================================================
	# define new grid (equal-area grid)
	# ============================================================
	# ----- 计算每个格子的面积 ----- 
	Dlat    = (flat[1:]-flat[:-1]).max() # 分辨率
	Dlon    = (flon[1:]-flon[:-1]).max() # + 0.03
	# --
	# s0, ds0 = define_EA_grid_area(Dlat, Dlon) #s0=小圆台的侧面积, ds0=等面积切分 (area of each grid cell)
	# --
	grid_area = calc_grid_area(flon, flat)
	# ds0 = grid_area.max() # 定义格子面积为赤道格点的面积
	if opt_grid_size=="mean":
		ds0 = grid_area.mean() # 定义格子面积为极地格点的面积
	elif opt_grid_size=="max":
		ds0 = grid_area.max() # 定义格子面积为极地格点的面积
	elif opt_grid_size=="min":
		ds0 = grid_area.min() # 定义格子面积为极地格点的面积
	elif opt_grid_size=="q25":
		ds0 = np.quantile(grid_area, 0.25)
	elif opt_grid_size=="q50":
		ds0 = np.quantile(grid_area, 0.50)
	elif opt_grid_size=="q75":
		ds0 = np.quantile(grid_area, 0.75)
	else:
		asd=asd
	# print(ds0) # 等面积切分 (area of each grid cell)

	# ----- 确定每层的格子数 ----- 
	N, N_df = define_EA_grid_number(flat, flon, Dlat, Dlon, ds0, fgrid_type=fgrid_type) # 格子数
	print(">>> total area = %0.3e km**2, total number of grid = %d, unit cell area = %0.3e km**2, Dlat=%0.3f, Dlon=%0.3f" % (N.sum()*ds0, N.sum(), ds0, Dlat, Dlon))

	# ============================================================
	# 数据处理 (amended by jer)
	# 在纬向方向上错开一个纬度，将整个纬向方向上经度上的数据进行平均处理
	# ============================================================
	if fgrid_type=="gaussian":
		# -- 数据处理 (amended by jer) --
		fdata_new = fdata.data #将sst_new中被mask覆盖的值换成fillvalue数值 (-32767)
		fdata_new[fdata_new==fdata.fill_value] = np.nan # 把fillvalue 的值都变成nan
		flat_new 	= flat.data
		flon_new 	= flon.data

	elif fgrid_type=="lat-lon":
		# -- 数据处理 (amended by jer) --
		fdata_new = (fdata.data[:,:-1,:] + fdata.data[:,1:,:]) / 2#在纬向方向上错开一个纬度，将整个纬向方向上经度上的数据进行平均处理
		fdata_new[fdata_new==fdata.fill_value] = np.nan # 把fillvalue 的值都变成nan
		flat_new = (flat[:-1] + flat[1:]) / 2 # (定义新的纬度数列，方便检查循环有没有出错、和后面保存到数据里)
		flon_new = flon + 0. # (定义新的经度数列，与原来的定义是一样的)

	# ----- added by jer -----
	# -- 附上coordinate (非必要)这样在读取变量的时候就可以显示该数值对应的coordinate --
	fdata_new = xr.DataArray(fdata_new, dims=["time","lat","lon"])
	fdata_new = fdata_new.assign_coords(time=ftime)
	fdata_new = fdata_new.assign_coords(lat=flat_new)
	fdata_new = fdata_new.assign_coords(lon=flon_new)
	# ------------------------

	# -------------------------------------------------
	# 转换数据 (amended by jer)
	# transfrom data from "gaussian grid" / "lat-lon grid" to "equal-area grid"
	# -------------------------------------------------
	if np.isnan(fdata_new).sum()>0:
		opt_allow_nan=True
	else:
		opt_allow_nan=False
	# --
	data_EA, lat_EA, lon_EA = transform_data_to_equal_area_grid(fdata_new, ftime, flat_new, flon_new, N, Dlat, Dlon, opt_allow_nan=opt_allow_nan)

	return(data_EA, lat_EA, lon_EA)


#********************************************************
#********************************************************
#********************************************************
# ----- functions for PDF calculation -----
#********************************************************
#********************************************************
#********************************************************
#==================================================
# functions: calculate pdf of equal-weighted data
# input: calc_pdf(data, pfd_bin1, pfd_bin2)
# output: return(pfd_list)
# -----
#==================================================
def calc_pdf(data, pfd_bin1, pfd_bin2, grid_area): 

	for i_bin in range(0, len(pfd_bin1)):
		sst_tmp1    = pfd_bin1[i_bin]
		sst_tmp2    = pfd_bin2[i_bin]
		fskt_tmp    = data + 0
		# -
		# -- pfd
		pfd_list = np.nansum((fskt_tmp>=sst_tmp1) & (fskt_tmp<sst_tmp2)) * grid_area * 1e-6 # unit: 10^6 km**2 
		# -- cfd
		# cfd_list   = ((fsst>=sst_tmp1)).sum(axis=1) # unit: cell

	return(pfd_list)

#==================================================
# functions: doing resmpaling based on bootsrap method
# input: bootstrap_resampling(data,n_resample,n_size)
# output: return(area_list)
# -----
# data = input data
# n_resample = the number of times of resampling
# size_ratio = the size of each time of resampling (size = len(data)*size_ratio)
# -----
# https://web.stanford.edu/class/archive/cs/cs109/cs109.1216/lectures/19_sampling_bootstrap_annotated.pdf
# https://www.journaldev.com/45580/bootstrap-sampling-in-python
#==================================================
# def bootstrap_resampling(data, pfd_bin, n_resample=100, size_ratio=1): 
def bootstrap_resampling(data, n_resample=100, size_ratio=1): 
	# x = np.random.normal(loc=300.0, size=1000)
	# print(np.mean(x))

	# -- the size of each time of resampling (n_size = len(data)*size_ratio) 
	if size_ratio==1:
		# n_size = len(data)
		n_size = np.isfinite(data).sum()
	else:
		# n_size = int(len(data) * size_ratio)
		n_size = int(np.isfinite(data).sum() * size_ratio)

	# resample_pfd_list = []
	resample_list = []
	data_cut = data[np.isfinite(data)]
	data_cut = pd.Series(data_cut)
	for i in range(n_resample):
		# -- resampling 
		# resample_tmp = random.sample(data_cut.tolist(), n_size) # without replacement
		resample_tmp = data_cut.sample(n_size,replace=True).values # with replacement
		resample_list.append(resample_tmp)
		# -- calculate pdf of the resampled data
		# resample_pfd_tmp = calc_pdf(resample_tmp, pfd_bin1, pfd_bin2, grid_area)
		# -- append to list
		# resample_pfd_list.append(resample_pfd_tmp)
	 
	# resample_pfd_mean = np.nanmean(resample_pfd_list, axis=1)
	# return(resample_pfd_mean)
	return(np.array(resample_list))

#==================================================
# functions: doing resmpaling based on smooth bootsrap method
# input: smooth_bootstrap_resampling(data, n_resample=100, size_ratio=1, opt_kernel="uniform", kernel_bandwidth=0.05)
# output: return(area_list)
# -----
# data = input data
# n_resample = the number of times of resampling
# size_ratio = the size of each time of resampling (size = len(data)*size_ratio)
# opt_kernel = "uniform" or "kernel" # Uniform kernel: A recorded measurement of 30 mm means that the true value of the sepal was in the interval [29.5, 30.5). In general, a recorded value of x means that the true value is in [x-0.5, x+0.5). Assuming that any point in that interval is equally likely leads to a uniform kernel with bandwidth 0.5. # Normal kernel: You can assume that the true measurement is normally distributed, centered on the measured value, and is very likely to be within [x-0.5, x+0.5). For a normal distribution, 95% of the probability is contained in ±2σ of the mean, so you could choose σ=0.25 and assume that the true value for the measurement x is in the distribution N(x, 0.25).
# -----
# https://blogs.sas.com/content/iml/2016/08/17/smooth-bootstrap-sas.html
#==================================================
def smooth_bootstrap_resampling(data, n_resample=100, size_ratio=1, opt_kernel="uniform", kernel_bandwidth=0.05): 
	# ----- kernel -----
	# opt_kernel = "uniform" # Uniform kernel: A recorded measurement of 30 mm means that the true value of the sepal was in the interval [29.5, 30.5). In general, a recorded value of x means that the true value is in [x-0.5, x+0.5). Assuming that any point in that interval is equally likely leads to a uniform kernel with bandwidth 0.5.
	# opt_kernel = "normal" # Normal kernel: You can assume that the true measurement is normally distributed, centered on the measured value, and is very likely to be within [x-0.5, x+0.5). For a normal distribution, 95% of the probability is contained in ±2σ of the mean, so you could choose σ=0.25 and assume that the true value for the measurement x is in the distribution N(x, 0.25).

	# -- the size of each time of resampling (n_size = len(data)*size_ratio) 
	if size_ratio==1:
		n_size = np.isfinite(data).sum()
	else:
		n_size = int(np.isfinite(data).sum() * size_ratio)

	resample_list = []
	data_cut = data[np.isfinite(data)]
	data_cut = pd.Series(data_cut)
	for i in range(n_resample):
		# -- resampling 
		resample_tmp = data_cut.sample(n_size,replace=True).values # with replacement
		# -- kernel
		if opt_kernel=="uniform":
			random_number = np.random.uniform(kernel_bandwidth*-1, kernel_bandwidth, n_size)
		elif opt_kernel in ["normal", "gaussian"]:
			random_number = np.random.normal(0, kernel_bandwidth/2, n_size)
		resample_tmp = resample_tmp + random_number
		# -- append
		resample_list.append(resample_tmp)

	return(np.array(resample_list))

#===============================================
# function: Doing resampling based on kernal bootstrap smoothing and mirroring (symmetrical input)
# input: resampling_kernal_bootstrap_sym(data, weights, n_resample=1000, size_ratio=1, opt_kernel="uniform", kernel_bandwidth=1)
# output: return(resample_arr)
# -----
# ref: 
#=============================================== 
def resampling_kernal_bootstrap_sym(data, weights, n_resample=1000, size_ratio=1, opt_kernel="uniform", kernel_bandwidth=1, opt_sym=False): 
	# ----- kernel -----
	# opt_kernel = "uniform" # Uniform kernel: A recorded measurement of 30 mm means that the true value of the sepal was in the interval [29.5, 30.5). In general, a recorded value of x means that the true value is in [x-0.5, x+0.5). Assuming that any point in that interval is equally likely leads to a uniform kernel with bandwidth 0.5.
	# opt_kernel = "normal" # Normal kernel: You can assume that the true measurement is normally distributed, centered on the measured value, and is very likely to be within [x-0.5, x+0.5). For a normal distribution, 95% of the probability is contained in ±2σ of the mean, so you could choose σ=0.25 and assume that the true value for the measurement x is in the distribution N(x, 0.25).

	# ----- size_ratio -----
	# -- the size of each time of resampling (n_size = len(data)*size_ratio) 
	if size_ratio==1:
		n_size = np.isfinite(data).sum()
	else:
		n_size = int(np.isfinite(data).sum() * size_ratio)

	# ----- create a symmetrical 1-dimensional data -----
	data_1d = data.reshape(-1) # reshape from 2D to 1D
	data_cut = data_1d[np.isfinite(data_1d)] # remove nan values
	if opt_sym==True: # mirror data along zero axis (symmetrical)
		data_cut_sym = np.concatenate([-data_cut,data_cut]) 
	elif opt_sym==False:
		data_cut_sym = data_cut + 0

	# ----- create a symmetrical 1-dimensional weighting array -----
	weights_1d = weights.reshape(-1) # reshape from 2D to 1D
	weights_cut = weights_1d[np.isfinite(data_1d)] # remove nan values
	if opt_sym==True: # mirror data along zero axis (symmetrical)
		weights_cut_sym = np.concatenate([weights_cut,weights_cut])
	elif opt_sym==False:
		weights_cut_sym = weights_cut + 0

	# ----- bootstrap for n_resample times (based on area-weighting function) -----
	resample_arr = np.zeros((n_resample, n_size),dtype=np.float64) * np.nan
	for i in range(n_resample):
		# -- resampling 
		# resample_tmp = data_cut.sample(n_size,replace=True,weights=weights_cut).values # with replacement #use pandas
		# resample_tmp = np.random.choice(data_cut,size=n_size,replace=True,p=weights_cut/(weights_cut.sum()))  # with replacement #use numpy
		resample_tmp = np.random.choice(data_cut_sym, size=n_size, replace=True, p=weights_cut_sym/(weights_cut_sym.sum()))

		# -- kernel
		if opt_kernel=="uniform":
			# random_number = np.random.uniform(kernel_bandwidth*-0.5, kernel_bandwidth*0.5, len(data_cut_sym))
			random_number = np.random.uniform(kernel_bandwidth, kernel_bandwidth, n_size)
		elif opt_kernel in ["normal", "gaussian"]:
			# random_number = np.random.normal(0, kernel_bandwidth/2, n_size)
			random_number = np.random.normal(0, kernel_bandwidth, n_size)
		elif opt_kernel=="none":
			random_number = resample_tmp * 0

		# -- append
		if opt_sym==True: # mirror data along zero axis (symmetrical)
			resample_tmp = resample_tmp + random_number
			resample_tmp[resample_tmp<0] = resample_tmp[resample_tmp<0] * -1
			resample_arr[i,:] = resample_tmp + 0
		elif opt_sym==False:
			resample_arr[i,:] = resample_tmp + random_number

	return(resample_arr)

#********************************************************
#********************************************************
#********************************************************
# ----- 6. functions for smoothing / filtering -----
#********************************************************
#********************************************************
#********************************************************
#==============================================
# spatial smoothing (9-point)
# ---------------
# https://www.ncl.ucar.edu/Document/Functions/Built-in/smth9.shtml
# -----
# Two scalars which affect the degree of smoothing. 
# In general, for good results, 
# set p = 0.50. With p = 0.50, a value of q = -0.25 results in "light" smoothing; 
# and q = 0.25 results in "heavy" smoothing. 
# A value of q = 0.0 results in a 5-point local smoother.
# -----
# This function performs 9-point smoothing using the equation:

# f0 = f0 + (p / 4) * (f2 + f4 + f6 + f8 - 4 * f0) + (q / 4) * (f1 + f3 + f5 + f7 - 4 * f0)
# where f0, f1 (and so on) are as indicated:
#       1-------------8---------------7
#       |             |               |
#       |             |               |
#       |             |               |
#       |             |               |
#       2-------------0---------------6
#       |             |               |
#       |             |               |
#       |             |               |
#       |             |               |
#       3-------------4---------------5
#==============================================
def smth9_ncl(data_in, p=0.5, q=-0.25):
	# -- f0
	f0 = data_in + 0
	# -- f1
	f1 = f0 * 0
	f1[1:,1:] = f0[:-1,:-1]
	# -- f2
	f2 = f0 * 0
	f2[:,1:] = f0[:,:-1]
	# -- f3
	f3 = f0 * 0
	f3[:-1,1:] = f0[1:,:-1]
	# -- f4
	f4 = f0 * 0
	f4[:-1,:] = f0[1:,:]
	# -- f5
	f5 = f0 * 0
	f5[:-1,:-1] = f0[1:,1:]
	# -- f6
	f6 = f0 * 0
	f6[:,:-1] = f0[:,1:]
	# -- f7
	f7 = f0 * 0
	f7[1:,:-1] = f0[:-1,1:]
	# -- f8
	f8 = f0 * 0
	f8[1:,:] = f0[:-1,:]
	# ----- filter -----
	data_out = f0 + (p / 4) * (f2 + f4 + f6 + f8 - 4 * f0) + (q / 4) * (f1 + f3 + f5 + f7 - 4 * f0)
	return(data_out)



#********************************************************
#********************************************************
#********************************************************
# ----- 7. functions for numerical simulation -----
#********************************************************
#********************************************************
#********************************************************
#=================================
# central 2nd order finite difference method: solving d(f)/dt
# f(lat,lon)
# d(f)/dt = (f(:,i+1) - f(:,i-1)) / 2dt
#=================================
def dfdt_cdf(f,t):
	# -- dt
	dt 		= t + np.nan
	dt[0]	= t[1] - t[0]
	dt[1:-1] = (t[2:] - t[:-2]) / 2
	dt[-1] 	= t[-1] - t[-2]

	# -- df/dt
	result  = f[:]*0 + np.nan
	result[1:-1]  = (f[2:] - f[:-2]) / (t[2:] - t[:-2])
	result[0]  = (f[1] - f[0]) / (t[1] - t[0])
	result[-1]  = (f[-1] - f[-2]) / (t[-1] - t[-2])
	return(result)

#=================================
# central 2nd order finite difference method: solving d(f)/dx
# f(lat,lon)
# d(f)/dx = (f(:,i+1) - f(:,i-1)) / 2dx
#=================================
def dfdx_cdf(f,dx):
	result  = f[:,:]*0
	result[:,1:-1]  = (f[:,2:] - f[:,:-2]) / (dx[:,1:] + dx[:,:-1])
	# -- cyclic
	result[:,0]  = (f[:,1] - f[:,-1]) / (dx[:,1] + dx[:,-1])
	result[:,-1]  = (f[:,0] - f[:,-2]) / (dx[:,0] + dx[:,-2])
	# -- non-cyclic
	# result[:,0]  = (f[:,1] - f[:,0]) / (dx[:,0])
	# result[:,-1]  = (f[:,-1] - f[:,-2]) / (dx[:,-1])
	return(result)

#=================================
# central finite difference method: solving d(f)/dy
# f(lat,lon)
# d(f)/dy = (f(j+1,:) - f(j-1,:)) / 2dy
#=================================
def dfdy_cdf(f,dy):
# def dfdy_cdf(f,dy,lat):
	result  = f[:,:]*0
	result[1:-1,:]  = (f[2:,:] - f[:-2,:]) / (dy[1:,:] + dy[:-1,:])
	# result[0,:]  = (f[1,:]) / (2*dy[0,:])
	# result[-1,:]  = (-1*f[-2,:]) / (2*dy[-1:,:])
	result[0,:]  = (f[1,:] - f[0,:]) / (dy[0,:])
	result[-1,:]  = (f[-1,:] - f[-2,:]) / (dy[-1,:])

	# a_earth = 6371*1000.
	# latt  = np.tile((lat)[:,np.newaxis],(1,len(result[0,:])))
	# latt  = np.abs(latt)
	# result = result - (f/a_earth)*np.tan(latt/180.*np.pi)
	return(result)
