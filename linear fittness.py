import numpy as np
import matplotlib.pyplot as plt

# 示例数据
x = np.array([0, 6.2, 12.3, 37, 111, 1000])
y = np.array([2479, 8116, 13913, 36069, 103345, 846981])

# 计算权重，权重为1/y^2
weights = 1 / (y ** 2)

# 非加权线性拟合
coefficients_unweighted = np.polyfit(x, y, deg=1)
slope_unweighted, intercept_unweighted = coefficients_unweighted

# 计算非加权的预测值
y_pred_unweighted = slope_unweighted * x + intercept_unweighted

# 计算非加权的相关系数R^2
ss_res_unweighted = np.sum((y - y_pred_unweighted) ** 2)
ss_tot_unweighted = np.sum((y - np.mean(y)) ** 2)
r2_unweighted = 1 - (ss_res_unweighted / ss_tot_unweighted)

# 加权线性拟合
coefficients_weighted = np.polyfit(x, y, deg=1, w=weights)
slope_weighted, intercept_weighted = coefficients_weighted

# 计算加权的预测值
y_pred_weighted = slope_weighted * x + intercept_weighted

# 计算加权的相关系数R^2
ss_res_weighted = np.sum(weights * (y - y_pred_weighted) ** 2)
ss_tot_weighted = np.sum(weights * (y - np.average(y, weights=weights)) ** 2)
r2_weighted = 1 - (ss_res_weighted / ss_tot_weighted)

# 打印拟合结果
print(f"Non-weighted Fit: Slope = {slope_unweighted:.4f}, Intercept = {intercept_unweighted:.4f}, R^2 = {r2_unweighted:.4f}")
print(f"Weighted Fit: Slope = {slope_weighted:.4f}, Intercept = {intercept_weighted:.4f}, R^2 = {r2_weighted:.4f}")

# 绘制数据点和拟合直线
plt.scatter(x, y, label='Data')

# 非加权拟合直线
x_fit = np.linspace(min(x), max(x), 100)
y_fit_unweighted = slope_unweighted * x_fit + intercept_unweighted
plt.plot(x_fit, y_fit_unweighted, color='blue', label=f'Non-weighted Fit: y = {slope_unweighted:.4f}x + {intercept_unweighted:.4f}\n$R^2$ = {r2_unweighted:.4f}')

# 加权拟合直线
y_fit_weighted = slope_weighted * x_fit + intercept_weighted
plt.plot(x_fit, y_fit_weighted, color='red', label=f'Weighted Fit: y = {slope_weighted:.4f}x + {intercept_weighted:.4f}\n$R^2$ = {r2_weighted:.4f}')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
