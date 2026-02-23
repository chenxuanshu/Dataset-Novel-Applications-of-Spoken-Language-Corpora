import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy import stats
import numpy as np
from sklearn.impute import SimpleImputer

# 加载年轻组数据
control_df = pd.read_csv('../embeddings-chinese-bert-wwm-ext/control_embedding.csv')

# 为年轻组数据添加标签 0
control_df['label'] = 0

# 加载 AD 组数据
ad_df = pd.read_csv('../embeddings-chinese-bert-wwm-ext/ad_embedding.csv')

# 为 AD 组数据添加标签 1
ad_df['label'] = 1

# 合并数据
combined_df = pd.concat([control_df, ad_df], ignore_index=True)

# 划分特征和目标变量
X = combined_df.drop('label', axis=1)
y = combined_df['label']

# 使用均值填充缺失值
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# 定义 5 折分层交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 定义要评估的指标
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score),
    'roc_auc': make_scorer(roc_auc_score)
}

# 定义模型列表
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(random_state=42)
}

# 初始化一个空字典来存储结果
results = {}

# 遍历每个模型并进行交叉验证
for model_name, model in models.items():
    cv_results = cross_validate(model, X_imputed, y, cv=cv, scoring=scoring)
    results[model_name] = {
        'Accuracy': np.mean(cv_results['test_accuracy']),
        'Precision': np.mean(cv_results['test_precision']),
        'Recall': np.mean(cv_results['test_recall']),
        'F1 Score': np.mean(cv_results['test_f1']),
        'AUC': np.mean(cv_results['test_roc_auc'])
    }

    # 计算效应大小
    accuracies = cv_results['test_accuracy']
    mean_diff = np.mean(accuracies) - np.mean(1 - accuracies)
    pooled_std = np.sqrt((np.std(accuracies, ddof=1) ** 2 + np.std(1 - accuracies, ddof=1) ** 2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0
    results[model_name]['Effect Size (Cohen\'s d)'] = cohens_d

    # 计算置信区间
    n = len(accuracies)
    std_err = stats.sem(accuracies)
    t_value = stats.t.ppf((1 + 0.95) / 2, n - 1)
    lower_bound = np.mean(accuracies) - t_value * std_err
    upper_bound = np.mean(accuracies) + t_value * std_err
    results[model_name]['95% Confidence Interval'] = (lower_bound, upper_bound)


results_df = pd.DataFrame.from_dict(results, orient='index')
print(results_df)