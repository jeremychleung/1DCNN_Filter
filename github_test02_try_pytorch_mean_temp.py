import os, sys
import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta
import calendar
import matplotlib.pyplot as plt
import torch
import random

# Set seeds for reproducibility across numpy, random, and torch (including CUDA)
np.random.seed(2024)
torch.manual_seed(2024)
torch.cuda.manual_seed_all(2024)

# Add a custom path to the module search paths; this allows importing custom modules not in the default search paths
sys.path.append('/home/jermy/l-python/')
import lib_math as lib_math

# Function to read mean temperature data from the HKO station
def read_hko_mean_temp_data():
    # Directly specifying the file path for HKO station's mean temperature data
    fn = "./data/mean_temp_daily/hkotemp_d_mean.txt"
    # Reading the data, assuming the first two lines of the HKO data file are headers or explanations to be skipped
    fdata = pd.read_csv(fn, skiprows=2, header=None, sep="\s+")
    # Naming the data columns
    fdata.columns = ["yyyymmdd", "data"]
    # Data preprocessing: treating values greater than 30000 as missing values
    fdata["data"][fdata["data"] > 30000] = np.nan
    # Unit conversion: converting data to Celsius
    fdata["data"] = fdata["data"] * 0.1
    # Creating an empty list for anomalies
    fdata["ano"] = fdata["data"] + 0
    # Adding columns for year, month, day
    fdata["year"] = np.floor(fdata.yyyymmdd / 10000).astype(int)
    fdata["month"] = np.floor((fdata.yyyymmdd - fdata["year"] * 10000) / 100).astype(int)
    fdata["day"] = np.floor((fdata.yyyymmdd - fdata["year"] * 10000 - fdata["month"] * 100)).astype(int)

    return fdata

# Function to check for missing dates and assess continuity of the time series
def check_if_there_is_missing_dates(fdata):
    tmp = []
    # Convert dates to numeric form for easy comparison
    for i in range(len(fdata)):
        tmp.append(nc.date2num(datetime(fdata["year"].iloc[i], fdata["month"].iloc[i], fdata["day"].iloc[i]),
                               units="days since 1800-01-01 00:00"))
    fdata["time"] = np.array(tmp)
    # Check for time differences to find discontinuities
    fdata["time_difference"] = np.nan
    fdata["time_difference"][1:] = fdata["time"].values[1:] - fdata["time"].values[:-1]
    # Identify and report discontinuities
    discont_ind = (fdata["time_difference"] > 1)
    if discont_ind.sum() >= 1:
        print(">>> Missing date exists --> the time series is not continuous")
        print(fdata[discont_ind])
    else:
        print(">>> No Missing date --> the time series is continuous")
    return (fdata)


# Function to remove climatology and calculate anomalies by a running mean approach
def calc_ano_by_rm_daily_ltm_runavg(fdata_tmp, var, window=31, year1=2006, year2=2022):
    fdata = fdata_tmp + 0
    fdata["ltm"] = fdata["ano"] + np.nan    # Initialize Long Term Mean (LTM) column
    # Loop through each month and day to calculate daily LTM
    for month in range(1, 13):
        for day in range(1, calendar.monthrange(2004, month)[1] + 1):
            ltm_sample = np.array([])
            # Aggregate samples for LTM calculation
            for year in range(year1, year2 + 1):
                if (month == 2) & (day == 29) & (year % 4 != 0):
                    date_tmp0 = datetime(year, 2, 28)
                else:
                    date_tmp0 = datetime(year, month, day)
                date_tmp1 = date_tmp0 - timedelta(days=int(window / 2))
                date_tmp2 = date_tmp0 + timedelta(days=int(window / 2))
                time_tmp1 = nc.date2num(date_tmp1, units="days since 1800-01-01 00:00")
                time_tmp2 = nc.date2num(date_tmp2, units="days since 1800-01-01 00:00")
                # Extract samples within the window for LTM calculation
                ltm_sample_tmp = np.array(fdata[(fdata.time >= time_tmp1) & (fdata.time <= time_tmp2)].data)
                ltm_sample = np.concatenate((ltm_sample, ltm_sample_tmp))

            # ----- Calculate std -----
            ltm_sample = np.array(ltm_sample)
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

    # ----- check nan -----
    nan_ind = np.isnan(ano_tmp)
    if nan_ind.sum() > 0:
        print(fdata_tmp[nan_ind])
        print(">>> missing values exist!! nan values exists in the time series!!")

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

    fil_tmp = xr.DataArray(ano_tmp, dims='x')

    fil_tmp_window = (fil_tmp.rolling(x=len(wgt), center=True).construct('window') * wgt)
    ind_nan = np.isnan(fil_tmp_window)
    ind_nan = ind_nan.sum(axis=1)
    fil_tmp = fil_tmp_window.sum(axis=1)
    fil_tmp[ind_nan > min_periods] = np.nan

    fil_tmp = fil_tmp.data
    print(np.isnan(fil_tmp).sum())
    print(np.isnan(ano_tmp).sum())
    print(np.isnan(data_tmp).sum())

    # -- append
    fdata_tmp["fil"] = fil_tmp
    fdata_tmp["fil_norm"] = fil_tmp / np.nanstd(fil_tmp)

    return (fdata_tmp)

# read
# ====================================================
station = "HKO"  # Only process data from the HKO site

# read station data
# ====================================================
# Use a function to read the average temperature data of the HKO site
fdata_meantemp = read_hko_mean_temp_data()
actual_fdata_meantemp = read_hko_mean_temp_data()

# ----- extract time
fdata_meantemp = fdata_meantemp[(fdata_meantemp["year"] >= 1950)].reset_index(drop=True)
actual_fdata_meantemp = actual_fdata_meantemp[(actual_fdata_meantemp["year"] >= 1950)].reset_index(drop=True)

# ====================================================
# check if there is missing dates
fdata_meantemp = check_if_there_is_missing_dates(fdata_meantemp)
actual_fdata_meantemp = check_if_there_is_missing_dates(actual_fdata_meantemp)

# ====================================================
# remove climatology
# -- rm climatology by daily ltm (31-day running mean)
fdata_meantemp = calc_ano_by_rm_daily_ltm_runavg(fdata_meantemp, "mean_temp", 31, 2006, 2022)

actual_fdata_meantemp = calc_ano_by_rm_daily_ltm_runavg(actual_fdata_meantemp, "mean_temp", 31, 2006, 2022)
# Apply the correct year filter to actual_fdata_meantemp to ensure the range is 2009 to 2021
actual_fdata_meantemp = actual_fdata_meantemp[
    (actual_fdata_meantemp['year'] >= 2009) & (actual_fdata_meantemp['year'] <= 2021)]
actual_fdata_meantemp_path = r'2009-2021data\actual_data.csv'
actual_fdata_meantemp.to_csv(actual_fdata_meantemp_path, index=False, encoding='utf-8-sig')
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
fdata_meantemp = calc_lanczos_filter(fdata_meantemp, filt_type, fcd_low, fcd_high, nwt, timestep, min_periods)

filed_fdata_meantemp = calc_lanczos_filter(actual_fdata_meantemp, filt_type, fcd_low, fcd_high, nwt, timestep, min_periods)  #add
# Apply the same year filtering and saving steps to filed_fdata_meantemp
filed_fdata_meantemp = filed_fdata_meantemp[
    (filed_fdata_meantemp['year'] >= 2009) & (filed_fdata_meantemp['year'] <= 2021)]
filed_fdata_meantemp_path = r'2009-2021data\lanczos_filtered_data.csv'
filed_fdata_meantemp.to_csv(filed_fdata_meantemp_path, index=False, encoding='utf-8-sig')
print(f"The data after Lanczos filtering has been saved to: {filed_fdata_meantemp_path}")

print(fdata_meantemp.iloc[150:155])

# prepare data for 1D CNN
# ====================================================
# ----- Remove the partial year data -----
### ----- Split the dataset to training, validation, and testing -----
# -- training data
year_ind = (fdata_meantemp.year >= 2006) & (fdata_meantemp.year <= 2019)
x_train = torch.tensor(fdata_meantemp.ano[year_ind][None, None, ...], dtype=torch.float64)  # x for training
y_train = torch.tensor(fdata_meantemp.fil[year_ind][None, None, ...], dtype=torch.float64)  # y for training

# -- validation data
year_ind = (fdata_meantemp.year >= 2020) & (fdata_meantemp.year <= 2021)
x_val = torch.tensor(fdata_meantemp.ano[year_ind][None, None, ...], dtype=torch.float64)  # x for validation
y_val = torch.tensor(fdata_meantemp.fil[year_ind][None, None, ...], dtype=torch.float64)  # y for validation

# -- testing data
year_ind = (fdata_meantemp.year >= 2022) & (fdata_meantemp.year <= 2022)
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

# ====================================================
# 1D CNN functions
# ====================================================
### Train the CNN model, get the CNN band pass filtered data, and save the weights to a text file
# Set the output path to store the model weights
kernel1 = 60  # 60
kernel2 = 10  # 10
no_epochs = 10000  # 10000 #500
learning_rate = 0.001  # 0.01 # default=0.001
epsilon = 1e-7  # default=1e-8
var_name = 'mean_temp'

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

# Defines the path to the model weight file
model_weights_path = r'model weight\cnn1d_model_weights.pth'

# train the model
# ----- Create the model -----
model = cnn1d_model(kernel1, kernel2)
model.double()  # Ensure that the model matches the data type

# Predefine variables before condition checking
loss_train_list = []
loss_val_list = []
corr_train_list = []
corr_val_list = []
aaaaaa = 0

# Check if there are pre-trained model weights
# if os.path.isfile(model_weights_path):     # If a weight file for the model exists, load the weight file
if aaaaaa == 1:     # If aaaaaa equals 1, retrain the model
    model.load_state_dict(torch.load(model_weights_path))     # Load pretrained weights
    print("Loaded pretrained weights.")
else:
    # ----- Define loss function and optimizer -----
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=epsilon)

    # ----- Training loop -----
    for epoch in range(no_epochs):

        # -- Forward pass
        outputs_train = model(x_train_filled_zeros)  # by x_train
        outputs_val = model(x_val_filled_zeros)  # by x_val

        # -- Compute loss （Calculate the loss of the epoch. When calculating the loss, ignore points with nan）
        check_finite = np.isfinite(y_train)
        loss_train = criterion(outputs_train[check_finite], y_train[check_finite])  # by x_train

        check_finite = np.isfinite(y_val)
        loss_val = np.nanmean((outputs_val.data[check_finite] - y_val[check_finite]) ** 2)  # by x_val

        loss_train_list.append(loss_train.data)
        loss_val_list.append(loss_val)

        # -- Backward and optimize
        optimizer.zero_grad()
        loss_train.backward()  # compute the gradients of the loss function with respect to the model's parameters
        optimizer.step()  # performs a parameter update based on the computed gradients

        # -- Compute correlation （Calculate the correlation coefficient of the epoch, similar to calculating loss, and also ignore points with nan）
        check_finite = np.isfinite(y_train)
        corr_train_list.append(
            lib_math.corr_n(outputs_train.data[check_finite], y_train[check_finite])[0])  # by x_train
        check_finite = np.isfinite(y_val)
        corr_val_list.append(lib_math.corr_n(outputs_val.data[check_finite], y_val[check_finite])[0])  # by x_val

        # -- Print progress
        if no_epochs >= 1001:
            if epoch % 100 == 0:
                print(f"Epoch {epoch + 1}/{no_epochs}, Loss_train: {loss_train.item()}, Loss_val: {loss_val}")
        else:
            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{no_epochs}, Loss_train: {loss_train.item()}, Loss_val: {loss_val}")

    loss_train_list = np.array(loss_train_list)
    loss_val_list = np.array(loss_val_list)
    corr_train_list = np.array(corr_train_list)
    corr_val_list = np.array(corr_val_list)
    print("corr_train_list-----------",corr_train_list)
    print("corr_val_list-----------", corr_val_list)

    # 保存模型权重
    torch.save(model.state_dict(), model_weights_path)
    print(f"Model weights saved to {model_weights_path}.")

# Make predictions
pred = model(x_test_filled_zeros)

check_nan = np.isnan(x_test)
pred[check_nan] = np.nan

criterion = torch.nn.MSELoss()
# ----- Predictions on test data -----
pred_test = model(x_test_filled_zeros)
# Calculate the loss on the test set
loss_test = criterion(pred_test, y_test)

# Print test loss
print(f"Test Loss: {loss_test.item()}")
print(f"Test Loss: {loss_test}")

# Use detach() method where numpy() needs to be called
df_pred = pd.DataFrame(pred.detach().numpy().flatten(), columns=['after_data'])

# Define output file path 1
output_file_path = r'prediction\mean_temp\HKO.csv'

# Save DataFrame as CSV file
df_pred.to_csv(output_file_path, index=False, encoding='utf-8-sig')  # Save Chinese characters using utf-8 encoding
print(f"The output has been saved to: {output_file_path}")

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

# plot setting (time series)
var = "meantemp"
dir_out = "./image_mean_temp/"
os.system("mkdir -p %s" % dir_out)
fon_prefix = "%s/02-try_%s_%s" % (dir_out, var, station)
title = "(%s, %s, kernel1=%d, kernel2=%d, no_epochs=%d)" % (station, var, kernel1, kernel2, no_epochs)

x = np.arange(0, len(plot_fil))
xmin = x.min()
xmax = x.max()

ymax1 = 38
ymin1 = 3
ymax2 = 8
ymin2 = ymax2 * -1

title_size = 48  # 32
tick_size = 40  # 28
axis_title_size = 44  # title_size

# plot data
fig, ax = plt.subplots(figsize=(40, 10), sharex=True)
ax2 = ax.twinx()

# ----- line -----
linewidth = 6
alpha = 0.7

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

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin1, ymax1)
ax2.set_ylim(ymin2, ymax2)

lines = [p2, p1, p4, p3, p5]
lines = [p1, p3, p2, p4, p5]
loc_legend = "lower center"
ax.legend(lines, [l.get_label() for l in lines], loc=loc_legend, ncol=5, handlelength=1.5, labelspacing=0.3,
          handletextpad=0.5, columnspacing=1.0, borderaxespad=0.2, shadow=True, fontsize=(tick_size - 4))

plt.savefig(fon_prefix + ".png", bbox_inches='tight', dpi=100)
print(">>> done plotting " + fon_prefix + ".png")
plt.close()

# plot setting (loss)
dir_out = "./image_mean_temp/"
os.system("mkdir -p %s" % dir_out)
fon_prefix = "%s/02-try_loss_%s_%s" % (dir_out, var, station)
title = "(%s, %s, kernel1=%d, kernel2=%d, no_epochs=%d)" % (station, var, kernel1, kernel2, no_epochs)

xmin = 0
xmax = len(loss_train_list)

title_size = 48  # 32
tick_size = 40  # 28
axis_title_size = 44  # title_size

# plot data
fig, ax = plt.subplots(figsize=(40, 10), sharex=True)
ax2 = ax.twinx()

# ----- line -----
linewidth = 6
alpha = 0.7

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
# -- ticks
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

ax.set_xlim(xmin, xmax)
ax.set_yscale('log', base=10)
ax2.set_yscale('log', base=10)

lines = [p1, p2, p3, p4]
loc_legend = "best"
ax.legend(lines, [l.get_label() for l in lines], loc=loc_legend, ncol=5, handlelength=1.5, labelspacing=0.3,
          handletextpad=0.5, columnspacing=1.0, borderaxespad=0.2, shadow=True, fontsize=(tick_size - 4))

plt.savefig(fon_prefix + ".png", bbox_inches='tight', dpi=100)
print(">>> done plotting " + fon_prefix + ".png")
plt.close()
