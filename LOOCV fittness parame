import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# 假设我们有一些样本数据 X 和标签 y
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([0, 0, 1, 1, 1])

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体字体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 定义逻辑回归模型
model = LogisticRegression(penalty='l2', solver='liblinear')  # 使用 L2 正则化

# 定义需要调优的超参数范围（C 是正则化强度的反比）
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

# 初始化 LOOCV
loo = LeaveOneOut()

# 使用 GridSearchCV 进行超参数调优，并使用 LOOCV 作为交叉验证方法
grid_search = GridSearchCV(model, param_grid, cv=loo)
grid_search.fit(X, y)

# 输出最佳的超参数
print(f"最优的正则化强度 C: {grid_search.best_params_['C']}")

# 获取最优模型
best_model = grid_search.best_estimator_

# 输出最优模型的参数 θ
theta_final = np.append(best_model.intercept_, best_model.coef_)
print(f"最优模型的参数 θ: {theta_final}")

# 对每个样本进行预测，并输出实际结果和预测结果
y_pred = best_model.predict_proba(X)[:, 1]  # 获取属于类别 1 的预测概率
for i, (actual, predicted) in enumerate(zip(y, y_pred)):
    print(f"样本 {i+1}: 实际结果 = {actual}, 预测结果 = {predicted:.4f}")

# 计算整体的损失值
loss = log_loss(y, y_pred)
print(f"模型的总体损失值（Log Loss）: {loss:.4f}")

# 可视化
plt.figure(figsize=(10, 6))
x_range = np.arange(len(y))  # 样本索引

# 绘制实际值和预测概率的条形图
plt.bar(x_range - 0.2, y, width=0.4, label='实际值', color='blue', align='center')
plt.bar(x_range + 0.2, y_pred, width=0.4, label='预测概率值', color='orange', align='center')

# 添加图形细节
plt.xlabel('样本索引')
plt.ylabel('分类')
plt.title('LOOCV 超参数调优后的逻辑回归预测结果')
plt.xticks(x_range)
plt.legend()

# 显示图形
plt.show()
