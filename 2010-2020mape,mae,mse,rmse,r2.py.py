import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score

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
# 过滤2010-2020年的数据
data_filtered = data[(data['yyyymmdd'].dt.year >= 2010) & (data['yyyymmdd'].dt.year <= 2020)]

# 提取过滤后的fil列作为需要的数据
fil_2010_2020 = data_filtered['fil'].values
data_selected = data_filtered['ano'].values

# 加载模型权重
model_weights_path = 'model weight/cnn1d_model_weights.pth'
model = cnn1d_model(kernel_size_1=60, kernel_size_2=10)
model.double()  # 设置模型为双精度
model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))

# 进行预测
x_test = torch.tensor(data_selected[None, None, :], dtype=torch.float64)
pred = model(x_test).detach().numpy().squeeze()

# 定义统计量计算函数
def calculate_mape_each(actual, predicted):
    return np.where(actual != 0, np.abs((actual - predicted) / actual) * 100, np.nan)

def calculate_mae_each(actual, predicted):
    return np.abs(actual - predicted)

def calculate_overall_mape_mae(mape_each, mae_each):
    mape_mean = np.nanmean(mape_each)
    mae_mean = np.nanmean(mae_each)
    return mape_mean, mae_mean

# 计算统计量
mape_each = calculate_mape_each(fil_2010_2020, pred)
mae_each = calculate_mae_each(fil_2010_2020, pred)
overall_mape, overall_mae = calculate_overall_mape_mae(mape_each, mae_each)
# 找出非NaN的索引
valid_indices = ~np.isnan(fil_2010_2020) & ~np.isnan(pred)
# 只使用非NaN的索引来计算MSE
mse = mean_squared_error(fil_2010_2020[valid_indices], pred[valid_indices])
rmse = np.sqrt(mse)
# 使用相同的非NaN索引来计算R²
r2 = r2_score(fil_2010_2020[valid_indices], pred[valid_indices])

# print(f"Overall MAPE: {overall_mape}%")
print(f"Overall MAE: {overall_mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")
