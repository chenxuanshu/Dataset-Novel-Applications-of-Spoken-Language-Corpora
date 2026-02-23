import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

# 1. 加载数据
features_vocabulary = '../features_data/features_vocabulary.xlsx'
features_syntactic = '../features_data/features_syntactic.xlsx'
features_semantic = '../features_data/features_semantic.xlsx'
features_all = '../features_data/features.xlsx'

# 1. 数据读取
data = pd.read_excel(features_syntactic) 
X = data.iloc[1:, :-1]  # 特征
y = data.iloc[1:, -1]   # 标签

# 2. 检查标签类型并转换
if not np.issubdtype(y.dtype, np.integer):
    y = y.astype('category').cat.codes 

# 3. 创建预处理和KNN管道
pipeline = Pipeline([
    ('scaler', StandardScaler()),      # 标准化
    ('pca', PCA(n_components=0.95)),   # 保留95%方差的PCA降维
    ('knn', KNeighborsClassifier(
        n_neighbors=20,                # 使用5个最近邻
        weights='distance',            # 根据距离加权投票
        algorithm='auto',             # 自动选择最佳算法
        p=2,                          # 使用欧氏距离
        n_jobs=-1                     # 使用所有CPU核心
    ))
])

# 4. 配置五折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 5. 定义评估指标（使用宏平均）
scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='macro'),
    'recall': make_scorer(recall_score, average='macro'),
    'f1': make_scorer(f1_score, average='macro')
}

# 6. 执行交叉验证并获取多个指标结果
cv_results = cross_validate(
    pipeline, 
    X, 
    y,
    cv=kf,
    scoring=scoring,
    return_train_score=False  # 只返回测试集分数
)

# 7. 提取并打印结果
print("\nKNN 交叉验证结果:")
print("=" * 50)

# 打印每个折的详细结果
for i in range(5):
    print(f"折 {i+1}:")
    print(f"  准确率: {cv_results['test_accuracy'][i]:.4f}")
    print(f"  精确率: {cv_results['test_precision'][i]:.4f}")
    print(f"  召回率: {cv_results['test_recall'][i]:.4f}")
    print(f"  F1分数: {cv_results['test_f1'][i]:.4f}")
    print("-" * 30)

# 计算并打印平均结果
print("\n平均性能:")
print(f"平均准确率: {np.mean(cv_results['test_accuracy']):.4f} ± {np.std(cv_results['test_accuracy']):.4f}")
print(f"平均精确率: {np.mean(cv_results['test_precision']):.4f} ± {np.std(cv_results['test_precision']):.4f}")
print(f"平均召回率: {np.mean(cv_results['test_recall']):.4f} ± {np.std(cv_results['test_recall']):.4f}")
print(f"平均F1分数: {np.mean(cv_results['test_f1']):.4f} ± {np.std(cv_results['test_f1']):.4f}")

# 8. 查看PCA降维后的特征数
pipeline.fit(X, y)  # 在整个数据集上拟合一次
print(f"\n原始特征数: {X.shape[1]}, 降维后特征数: {pipeline.named_steps['pca'].n_components_}")

# 9. 寻找最佳K值
print("\n寻找最佳K值 (1-20范围内):")
k_values = list(range(1, 21))
k_accuracies = []

for k in k_values:
    pipeline.set_params(knn__n_neighbors=k)
    cv_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='accuracy')
    mean_acc = np.mean(cv_scores)
    k_accuracies.append(mean_acc)
    print(f"K={k:2d} | 平均准确率: {mean_acc:.4f}")

# 找到最佳K值
best_k = k_values[np.argmax(k_accuracies)]
print(f"\n最佳K值: {best_k} (准确率: {max(k_accuracies):.4f})")

# 用最佳K值重新训练模型
pipeline.set_params(knn__n_neighbors=best_k)
pipeline.fit(X, y)
print(f"使用最佳K值({best_k})重新训练模型完成")