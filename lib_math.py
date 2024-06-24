# -*- coding: utf-8 -*-
"""
@author: jer
"""
import sys, os
import numpy as np
import pandas as pd
import xarray as xr
import math as mt
from datetime import datetime, timedelta
from scipy import stats, interpolate
import metpy.calc as mpcalc
import pymannkendall as mk
import random

#==================================================
# function: same as np.nansum 
# input: calc_nansum(data, **kwargs)
# output: return(data)
# -----
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
