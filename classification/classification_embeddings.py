import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ===================== 1. 数据加载 =====================
df_control = pd.read_csv('../embeddings_data/embedding_test_control_bert.csv')
df_mci = pd.read_csv('../embeddings_data/embedding_test_older_bert.csv')
df_ad = pd.read_csv('../embeddings_data/embedding_test_ad_bert.csv')

df_control['label'] = 0
df_mci['label'] = 1
df_ad['label'] = 2

# ===================== 清理所有NaN =====================
df_control = df_control.fillna(0)
df_mci = df_mci.fillna(0)
df_ad = df_ad.fillna(0)

# ===================== 2. 划分训练/测试集 =====================
control_train, control_test = train_test_split(
    df_control, test_size=12, random_state=42
)

train_features = pd.concat([control_train.iloc[:, :-1], df_mci.iloc[:, :-1]], axis=0)
test_features = pd.concat([control_test.iloc[:, :-1], df_ad.iloc[:, :-1]], axis=0)

train_features = train_features.fillna(0)
test_features = test_features.fillna(0)

X_train = train_features.values
X_test = test_features.values

y_train = pd.concat([control_train['label'], df_mci['label']], axis=0).values
y_test = pd.concat([control_test['label'], df_ad['label']], axis=0).values
y_test = np.where(y_test == 2, 1, y_test)

print("训练集特征 NaN 数量：", np.isnan(X_train).sum())
print("测试集特征 NaN 数量：", np.isnan(X_test).sum())

print(f"训练集样本数：{X_train.shape[0]} (36年轻人+42MCI)")
print(f"测试集样本数：{X_train.shape[0]} (12年轻人+51AD)")

# ===================== 3. 标准化 + PCA降维 =====================
scaler = StandardScaler()
pca = PCA(n_components=0.95)

X_train_scaled = scaler.fit_transform(X_train)
X_train_pca = pca.fit_transform(X_train_scaled)

X_test_scaled = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_scaled)

print(f"\n降维前特征维度：{X_train.shape[1]}")
print(f"降维后特征维度：{X_train_pca.shape[1]}")

# ===================== 4. 模型定义 =====================
models = {
    '随机森林': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        }
    },
    '支持向量机(SVM)': {
        'model': SVC(probability=True, random_state=42),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
    },
    'K近邻(KNN)': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        }
    }
}

# ===================== 5. 评估函数 =====================
def calculate_metrics(y_true, y_pred, y_pred_proba):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba[:, 1])

    control_scores = y_pred_proba[y_true == 0, 1]
    patient_scores = y_pred_proba[y_true == 1, 1]
    cohen_d = (np.mean(patient_scores) - np.mean(control_scores)) / np.sqrt(
        (np.var(control_scores, ddof=1) + np.var(patient_scores, ddof=1)) / 2
    )

    n = len(y_true)
    correct = np.sum(y_true == y_pred)
    z = stats.norm.ppf(0.975)
    p = acc
    ci_lower = (p + z**2/(2*n) - z * np.sqrt((p*(1-p) + z**2/(4*n))/n)) / (1 + z**2/n)
    ci_upper = (p + z**2/(2*n) + z * np.sqrt((p*(1-p) + z**2/(4*n))/n)) / (1 + z**2/n)

    return {
        '准确率': acc,
        '精确率': precision,
        '召回率': recall,
        'F1分数': f1,
        'AUC': auc,
        '效应量(Cohen\'s d)': cohen_d,
        '95%置信区间': f'[{ci_lower:.4f}, {ci_upper:.4f}]'
    }

# ===================== 6. 训练与评估 =====================
results = []
for name, config in models.items():
    print(f"\n{'='*50}")
    print(f"正在训练：{name}")
    print(f"{'='*50}")

    grid = GridSearchCV(config['model'], config['params'], cv=5, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train_pca, y_train)

    best_model = grid.best_estimator_
    print(f"最优参数：{grid.best_params_}")

    y_pred = best_model.predict(X_test_pca)
    y_pred_proba = best_model.predict_proba(X_test_pca)

    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    metrics['模型'] = name
    results.append(metrics)

    for k, v in metrics.items():
        print(f"{k}：{v:.4f}" if isinstance(v, float) else f"{k}：{v}")

# ===================== 7. 输出结果 =====================
print(f"\n{'='*80}")
print(f"所有模型评估结果（已修复NaN+标准化+降维）")
print(f"{'='*80}")
result_df = pd.DataFrame(results).round(4)
print(result_df)