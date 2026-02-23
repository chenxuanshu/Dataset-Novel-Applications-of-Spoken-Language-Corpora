import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 正常显示负号
plt.rcParams['axes.unicode_minus'] = False

# 读取文件
excel_file = pd.ExcelFile('../test_data/features.xlsx')

# 获取指定工作表中的数据
df = excel_file.parse('Sheet1')

# 提取特征和目标变量
X = df.drop(['Unnamed: 0', 'label'], axis=1)
y = df['label']

# 将目标变量转换为数值
y = y.map({'young': 0, 'ad': 1})

# 直接处理目标变量中的缺失值
y = y.dropna()

# 确保特征变量与目标变量的索引一致
X = X.loc[y.index]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义模型
models = {
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42, probability=True)
}

# 存储结果
results = {}

# 遍历模型
for model_name, model in models.items():
    # 交叉验证
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)

    # 拟合模型
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # 计算效应大小（以Cohen's d为例）
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    d = (np.mean(y_pred_proba[y_test == 1]) - np.mean(y_pred_proba[y_test == 0])) / np.sqrt(
        (np.std(y_pred_proba[y_test == 1]) ** 2 + np.std(y_pred_proba[y_test == 0]) ** 2) / 2)

    # 计算置信区间
    n = len(y_pred)
    z = 1.96  # 95%置信区间
    error = z * np.sqrt((accuracy * (1 - accuracy)) / n)
    confidence_interval = (accuracy - error, accuracy + error)

    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # 存储结果
    results[model_name] = {
        '交叉验证分数': cv_scores,
        '准确率': accuracy,
        '精确率': precision,
        '召回率': recall,
        'F1分数': f1,
        '效应大小': d,
        '置信区间': confidence_interval,
        'FPR': fpr,
        'TPR': tpr,
        'AUC': roc_auc
    }

# 输出结果
for model_name, result in results.items():
    print(f'model: {model_name}')
    print(f"交叉验证分数: {result['交叉验证分数']}")
    print(f"平均交叉验证分数: {np.mean(result['交叉验证分数']):.2f}")
    print(f"准确率: {result['准确率']:.2f}")
    print(f"精确率: {result['精确率']:.2f}")
    print(f"召回率: {result['召回率']:.2f}")
    print(f"F1分数: {result['F1分数']:.2f}")
    print(f"效应大小: {result['效应大小']:.2f}")
    print(f"置信区间: ({result['置信区间'][0]:.2f}, {result['置信区间'][1]:.2f})")
    print(f"AUC: {result['AUC']:.2f}")
    print('-' * 50)



# 查看Random Forest模型的特征重要性
if 'Random Forest' in models:
    rf_model = models['Random Forest']
    feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
    sorted_feature_importance = feature_importance.sort_values(ascending=False)
    print("Random Forest模型特征重要性排序:")
    print(sorted_feature_importance)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
for model_name, result in results.items():
    plt.plot(result['FPR'], result['TPR'], label=f'{model_name} (AUC = {result["AUC"]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend(loc='lower right')

plt.show()