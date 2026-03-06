import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 必须在 import plt 之前设置
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import glob
import os 
file_list = glob.glob("../../output/solar_wave000*_T.h5")
def extract_number(filename):
    basename = os.path.basename(filename)
    num_str = ''.join([c for c in basename if c.isdigit()])    
    return int(num_str) if num_str else 0

file_list_sorted = sorted(file_list, key=extract_number)

signal_list = [] 
time_list = []    

for i, file in enumerate(file_list_sorted):
    with h5py.File(file, "r") as f:
        vz_3d = f["vz"][:]  
        signal = np.mean(vz_3d, axis=(0,1,2))
        signal_list.append(signal)
        time_list.append(i)  
    del vz_3d 

# 转成pd

df = pd.DataFrame({
    "time": time_list,
    "signal": signal_list  
})
print("数据读取+降维完成，前5行：")
print(df.head())
print("时序数据长度（时刻数）：", len(df))

# -------------------- 数据清洗 --------------------

df = df[df["signal"].between(df["signal"].quantile(0.01), df["signal"].quantile(0.99))]
df = df.dropna()
print("清洗后数据量：", len(df))

# -------------------- 特征工程（滑动窗口/差分/傅里叶/自回归） --------------------
df["diff_1"] = df["signal"].diff()  # 差分
df["window_10_mean"] = df["signal"].rolling(window=10).mean()  # 滑动窗口均值
#傅里叶
t = np.arange(len(df))
cycle_length = 50  
df["fourier_sin"] = np.sin(2 * np.pi * t / cycle_length)
df["fourier_cos"] = np.cos(2 * np.pi * t / cycle_length)
# 自回归：前1个时刻值
df["signal_shift_1"] = df["signal"].shift(1)
df = df.dropna()  # 去掉空

# -------------------- 构造训练数据 --------------------

X = df[["diff_1", "window_10_mean", "fourier_sin", "fourier_cos", "signal_shift_1"]]
y = df["signal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
# -------------------- 训练模型 --------------------
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# 评估误差
mse = mean_squared_error(y_test, y_pred)
print("预测误差 MSE =", mse)
# -------------------- 画图 --------------------
plt.figure(figsize=(12,4))
plt.plot(y_test.values, label="True Value (3D mean time series)", color="blue")
plt.plot(y_pred, label="Predicted Value", color="red")
plt.xlabel("Time Step (Test Set)")
plt.ylabel("Mean of Physical Quantity")
plt.title("Time Series Prediction of Solar 3D Simulation Data")
plt.legend()
plt.savefig("timeseries_result.png")
