import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, r2_score

# 更新字体大小设置
title_fontsize = 30  # 标题字体大小增加至30
label_fontsize = 28  # 标签字体大小增加至28
legend_fontsize = 26  # 图例字体大小增加至26
ticks_fontsize = 24  # 刻度字体大小增加至24

# 定义按特定规则替换数据为NaN值的函数
def replace_with_nan_interval(data, interval=60, days_to_replace=1):
    total_count = data.shape[0]
    for start_day in range(0, total_count, interval):
        end_day = min(start_day + days_to_replace, total_count)
        data[start_day:end_day] = np.nan
    return data

# 定义CNN模型
class cnn1d_model(torch.nn.Module):
    def __init__(self, kernel_size_1=60, kernel_size_2=10):
        super(cnn1d_model, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 1, kernel_size=kernel_size_1, bias=False, padding='same')
        self.conv2 = torch.nn.Conv1d(1, 1, kernel_size=kernel_size_2, bias=False, padding='same')

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = x - x1
        x3 = self.conv2(x2)
        return x3

# 加载数据，读取指定列
file_path = '2009-2021data/lanczos_filtered_data.csv'
data = pd.read_csv(file_path, usecols=['yyyymmdd', 'ano', 'fil'])
data_selected = data['ano']

# 将日期列转换为日期时间格式以便过滤
data['yyyymmdd'] = pd.to_datetime(data['yyyymmdd'], format='%Y%m%d')
# 过滤2018年的数据
data_filtered = data[(data['yyyymmdd'].dt.year == 2018)]
# 获取过滤后数据在原始数据集中的索引
filtered_indices = data_filtered.index

# 提取过滤后的fil列作为需要的数据
fil_2018 = data_filtered['fil'].values

# 加载模型权重
model_weights_path = 'model weight/cnn1d_model_weights.pth'
model = cnn1d_model(kernel_size_1=60, kernel_size_2=10)
model.double()  # 设置模型为双精度
model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))

# 定义统计量计算函数
def calculate_mape_each(actual, predicted):
    return np.where(actual != 0, np.abs((actual - predicted) / actual) * 100, np.nan)

def calculate_mae_each(actual, predicted):
    return np.abs(actual - predicted)

def calculate_overall_mape_mae(mape_each, mae_each):
    mape_mean = np.nanmean(mape_each)
    mae_mean = np.mean(mae_each)
    return mape_mean, mae_mean

# 定义绘图函数，使用更新的字体大小
def plot_metric_each(values, metric_name, save_path):
    plt.figure(figsize=(14, 7))  # 调整画布大小以适应更大的字体
    plt.plot(values, label=metric_name)
    plt.xlabel('Observations', fontsize=label_fontsize)
    plt.ylabel(metric_name, fontsize=label_fontsize)
    plt.title(f'{metric_name} over Observations', fontsize=title_fontsize)
    plt.legend(fontsize=legend_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.grid(True)
    plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
    plt.savefig(save_path)
    plt.close()

# 在循环外初始化收集评价指标的列表
mape_values = []
mae_values = []
mse_values = []
rmse_values = []
r2_values = []
days_to_replace_list = list(range(1, 61))  # days_to_replace的值从1到60

actual_values_2018 = fil_2018

# 对2009-2021年的数据进行预测和评价指标的计算
for days_to_replace in range(1, 61):
    data_modified = replace_with_nan_interval(data_selected.copy().values, 60, days_to_replace)
    data_selected_no_nan = np.nan_to_num(data_modified)

    # 进行预测
    x_test = torch.tensor(data_selected_no_nan[None, None, :], dtype=torch.float64)
    pred = model(x_test).detach().numpy().squeeze()

    # 从预测结果中提取2018年数据进行评价
    predicted_values_2018 = pred[filtered_indices]

    # 在实际值和预测值对比图的绘制部分也应用更新的字体大小
    plt.figure(figsize=(18, 9))  # 调整画布大小以适应更大的字体
    plt.plot(actual_values_2018, alpha=0.7, label='Actual fil', color='m')
    plt.plot(predicted_values_2018, alpha=0.5, label='Predicted cnn', color='k')
    plt.xlabel('Day of the Year', fontsize=label_fontsize)
    plt.ylabel('Temperature', fontsize=label_fontsize)
    plt.legend(fontsize=legend_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.grid(True)
    plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域

    # 保存图表到指定路径
    save_path = f'mape2018/picture_mape2018_missing/fil_vs_cnn_2018_missing/fil_vs_cnn_2018_missing_days_to_replace={days_to_replace}.png'
    plt.savefig(save_path)

    # 计算统计量
    mape_each = calculate_mape_each(actual_values_2018, predicted_values_2018)
    mae_each = calculate_mae_each(actual_values_2018, predicted_values_2018)
    overall_mape, overall_mae = calculate_overall_mape_mae(mape_each, mae_each)
    mse = mean_squared_error(actual_values_2018, predicted_values_2018)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_values_2018, predicted_values_2018)

    # 收集每个评价指标的值
    mape_values.append(overall_mape)
    mae_values.append(overall_mae)
    mse_values.append(mse)
    rmse_values.append(rmse)
    r2_values.append(r2)

    # 创建目录和保存统计量
    dir_path = f'mape2018/picture_mape2018_missing/days_to_replace={days_to_replace}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    stats_path = os.path.join(dir_path, f'days_to_replace={days_to_replace}.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(f"Overall MAPE: {overall_mape}%\n")
        f.write(f"Overall MAE: {overall_mae}\n")
        f.write(f"MSE: {mse}\n")
        f.write(f"RMSE: {rmse}\n")
        f.write(f"R²: {r2}\n")

    print(f"完成 days_to_replace={days_to_replace} 的处理和保存")

# 绘制评价指标随days_to_replace变化的图表并保存，使用更新的字体大小
def plot_evaluation_index_vs_days(index_values, index_name, save_path):
    plt.figure(figsize=(14, 7))  # 调整画布大小以适应更大的字体
    plt.plot(days_to_replace_list, index_values, marker='o', linestyle='-')
    plt.xlabel('Days to Replace', fontsize=label_fontsize)
    plt.ylabel(index_name, fontsize=label_fontsize)
    plt.title(f'{index_name} vs. Days to Replace', fontsize=title_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.grid(True)
    plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
    plt.savefig(save_path)
    plt.close()

# 创建保存图表的目录
evaluation_index_dir = 'mape2018/picture_mape2018_missing/2018Evaluation index'
if not os.path.exists(evaluation_index_dir):
    os.makedirs(evaluation_index_dir)

# 绘制并保存图表
plot_evaluation_index_vs_days(mae_values, 'MAE', os.path.join(evaluation_index_dir, 'MAE_vs_Days_to_Replace.png'))
plot_evaluation_index_vs_days(mape_values, 'MAPE', os.path.join(evaluation_index_dir, 'MAPE_vs_Days_to_Replace.png'))
plot_evaluation_index_vs_days(mse_values, 'MSE', os.path.join(evaluation_index_dir, 'MSE_vs_Days_to_Replace.png'))
plot_evaluation_index_vs_days(rmse_values, 'RMSE', os.path.join(evaluation_index_dir, 'RMSE_vs_Days_to_Replace.png'))
plot_evaluation_index_vs_days(r2_values, 'R2', os.path.join(evaluation_index_dir, 'R2_vs_Days_to_Replace.png'))

print("完成所有评价指标随days_to_replace变化的图表绘制和保存")
