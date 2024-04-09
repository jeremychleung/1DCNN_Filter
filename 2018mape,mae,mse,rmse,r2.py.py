import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

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

# 将日期列转换为日期时间格式以便过滤
data['yyyymmdd'] = pd.to_datetime(data['yyyymmdd'], format='%Y%m%d')
# 过滤2018年的数据
data_filtered = data[(data['yyyymmdd'].dt.year == 2018)]

# 提取过滤后的fil列作为需要的数据
fil_2018 = data_filtered['fil'].values
data_selected = data_filtered['ano'].values

# 加载模型权重
model_weights_path = 'model weight/cnn1d_model_weights.pth'
model = cnn1d_model(kernel_size_1=60, kernel_size_2=10)
model.double()  # 设置模型为双精度
model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))

# 进行预测
x_test = torch.tensor(data_selected[None, None, :], dtype=torch.float64)
pred = model(x_test).detach().numpy().squeeze()

# 更新字体大小设置
title_fontsize = 30  # 标题字体大小增加至30
label_fontsize = 28  # 标签字体大小增加至28
legend_fontsize = 26  # 图例字体大小增加至26
ticks_fontsize = 24  # 刻度字体大小增加至24

# 绘制实际值和预测值对比图，增大了字体大小
plt.figure(figsize=(14, 7))
plt.plot(fil_2018, alpha=0.7, label='Actual fil', color="m")
plt.plot(pred, alpha=0.5, label='Predicted cnn', color="k")
plt.xlabel('Day of the Year', fontsize=label_fontsize)
plt.ylabel('Temperature', fontsize=label_fontsize)
plt.legend(fontsize=legend_fontsize)
plt.xticks(fontsize=ticks_fontsize)
plt.yticks(fontsize=ticks_fontsize)
plt.grid(True)
plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域

# 保存图表到文件
plt.savefig('mape2018/picture_mape2018_no missing/fil_vs_cnn_2018.png')

# 计算并打印统计量
def calculate_mape_each(actual, predicted):
    return np.where(actual != 0, np.abs((actual - predicted) / actual) * 100, np.nan)

def calculate_mae_each(actual, predicted):
    return np.abs(actual - predicted)

mape_each = calculate_mape_each(fil_2018, pred)
mae_each = calculate_mae_each(fil_2018, pred)
mape_mean = np.nanmean(mape_each)
mae_mean = np.nanmean(mae_each)

valid_indices = ~np.isnan(fil_2018) & ~np.isnan(pred)
mse = mean_squared_error(fil_2018[valid_indices], pred[valid_indices])
rmse = np.sqrt(mse)
r2 = r2_score(fil_2018[valid_indices], pred[valid_indices])

print(f"Overall MAPE: {mape_mean}%")
print(f"Overall MAE: {mae_mean}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")
