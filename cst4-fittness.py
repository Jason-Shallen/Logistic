import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# 定义四参数逻辑函数
def four_parameter_logistic(x, a, b, c, d):
    return ((a - d) / (1.0 + ((x / c) ** b))) + d

# 数据集
x_data = np.array([5,50, 100, 200, 400, 800, 1600])
y_data = np.array([0.085,0.729, 1.186, 1.566, 1.943, 2.226, 2.462])

# 无权重拟合
params_no_weight, _ = curve_fit(four_parameter_logistic, x_data, y_data, maxfev=10000)
y_fit_no_weight = four_parameter_logistic(x_data, *params_no_weight)

# 有权重拟合（权重为 1/(y+0.1)^2，避免除零问题）
weights = 1 / ((y_data + 0.1) ** 2)
params_weight, _ = curve_fit(four_parameter_logistic, x_data, y_data, sigma=weights, maxfev=10000)
y_fit_weight = four_parameter_logistic(x_data, *params_weight)

# 计算残差和相关系数
residuals_no_weight = y_data - y_fit_no_weight
residuals_weight = y_data - y_fit_weight
r2_no_weight = r2_score(y_data, y_fit_no_weight)
r2_weight = r2_score(y_data, y_fit_weight)

# 计算浓度和偏差（使用拟合方程求反函数）
def inverse_four_parameter_logistic(y, a, b, c, d):
    try:
        return c * ((a - y) / (y - d + 1e-10)) ** (1 / b)  # 加入1e-10避免除以零问题
    except ValueError:
        return np.nan  # 如果出现计算错误，返回NaN

calculated_concentration_no_weight = inverse_four_parameter_logistic(y_data, *params_no_weight)
deviation_no_weight = calculated_concentration_no_weight - x_data

calculated_concentration_weight = inverse_four_parameter_logistic(y_data, *params_weight)
deviation_weight = calculated_concentration_weight - x_data

# 构建结果表格
data = {
    '浓度 (X)': x_data,
    '吸光值 (OD450)': y_data,
    '计算浓度 (无权重)': calculated_concentration_no_weight,
    '偏差 (无权重)': deviation_no_weight,
    '计算浓度 (有权重)': calculated_concentration_weight,
    '偏差 (有权重)': deviation_weight
}
result_df = pd.DataFrame(data)

# 输出结果
print("无权重拟合参数: a = {:.4f}, b = {:.4f}, c = {:.4f}, d = {:.4f}".format(*params_no_weight))
print("无权重拟合相关系数: {:.4f}, 相关系数的平方: {:.4f}".format(np.sqrt(r2_no_weight), r2_no_weight))

print("\n有权重拟合参数: a = {:.4f}, b = {:.4f}, c = {:.4f}, d = {:.4f}".format(*params_weight))
print("有权重拟合相关系数: {:.4f}, 相关系数的平方: {:.4f}".format(np.sqrt(r2_weight), r2_weight))

# 打印结果表格
print("\n拟合结果表格:")
print(result_df)

# 绘制拟合曲线
x_plot = np.linspace(0, 1600, 1000)
y_plot_no_weight = four_parameter_logistic(x_plot, *params_no_weight)
y_plot_weight = four_parameter_logistic(x_plot, *params_weight)

plt.scatter(x_data, y_data, label='原始数据', color='black')
plt.plot(x_plot, y_plot_no_weight, label='无权重拟合', color='blue')
plt.plot(x_plot, y_plot_weight, label='有权重拟合', color='red')
plt.xlabel('浓度 (X)', fontproperties='SimHei')
plt.ylabel('吸光值 (OD450)', fontproperties='SimHei')
plt.legend(prop={'family': 'SimHei'})
plt.title('4参数逻辑曲线拟合', fontproperties='SimHei')
plt.show()

# 用户输入多个y值，计算对应的X浓度
def calculate_x_from_y(y_values, params_no_weight, params_weight):
    x_no_weight = [inverse_four_parameter_logistic(y, *params_no_weight) for y in y_values]
    x_weight = [inverse_four_parameter_logistic(y, *params_weight) for y in y_values]
    return np.array(x_no_weight), np.array(x_weight)

# 用户输入y值
user_y_values = np.array([0.5, 1.0, 1.5, 2.0,2.2,2.3,2.40])  # 可以替换为用户输入的y值列表
x_no_weight, x_weight = calculate_x_from_y(user_y_values, params_no_weight, params_weight)

# 输出计算结果
for i, y in enumerate(user_y_values):
    print("\n对于吸光值 y = {:.3f}:".format(y))
    print("无权重拟合对应的浓度 X = {:.3f}".format(x_no_weight[i]))
    print("有权重拟合对应的浓度 X = {:.3f}".format(x_weight[i]))
