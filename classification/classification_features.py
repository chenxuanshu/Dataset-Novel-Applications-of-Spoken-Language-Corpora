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
from sklearn.pipeline import Pipeline              
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ===================== 1. 数据加载与标签设置 =====================
df_control = pd.read_excel('../features_data/features_test_control.xlsx')  # 年轻人 48人
df_mci = pd.read_excel('../features_data/features_test_older.xlsx')        # MCI老年人 42人
df_ad = pd.read_excel('../features_data/features_test_ad.xlsx')  

df_control['label'] = 0
df_mci['label'] = 1
df_ad['label'] = 2

# ===================== 2. 划分训练/测试集 =====================
control_train, control_test = train_test_split(
    df_control, test_size=12, random_state=42
)

X_train = pd.concat([control_train.iloc[:, :-1], df_mci.iloc[:, :-1]], axis=0).values
y_train = pd.concat([control_train['label'], df_mci['label']], axis=0).values

X_test = pd.concat([control_test.iloc[:, :-1], df_ad.iloc[:, :-1]], axis=0).values
y_test = pd.concat([control_test['label'], df_ad['label']], axis=0).values
y_test = np.where(y_test == 2, 1, y_test)

print(f"训练集样本数：{X_train.shape[0]} (36年轻人+42MCI)")
print(f"测试集样本数：{X_test.shape[0]} (12年轻人+51AD)")

# ===================== 3. 标准化 + PCA降维 + 模型管道 =====================
def create_pipeline(model):
    return Pipeline([
        ('scaler', StandardScaler()),       
        ('pca', PCA(n_components=0.95)),    
        ('classifier', model)               
    ])

# ===================== 4. 模型 + 网格搜索 =====================
models = {
    '随机森林': {
        'pipeline': create_pipeline(RandomForestClassifier(random_state=42)),
        'params': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [5, 10, None],
            'classifier__min_samples_split': [2, 5]
        }
    },
    '支持向量机(SVM)': {
        'pipeline': create_pipeline(SVC(probability=True, random_state=42)),
        'params': {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf'],
            'classifier__gamma': ['scale', 'auto']
        }
    },
    'K近邻(KNN)': {
        'pipeline': create_pipeline(KNeighborsClassifier()),
        'params': {
            'classifier__n_neighbors': [3, 5, 7, 9],
            'classifier__weights': ['uniform', 'distance']
        }
    }
}

# ===================== 5. 评估指标函数 =====================
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

# ===================== 6. 训练 + 网格搜索 + 评估 =====================
results = []
for name, config in models.items():
    print(f"\n{'='*50}")
    print(f"正在训练：{name}（已标准化+PCA降维）")
    print(f"{'='*50}")

    # 网格搜索
    grid = GridSearchCV(
        estimator=config['pipeline'],  
        param_grid=config['params'],
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print(f"最优参数：{grid.best_params_}")

    # 预测
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)

    # 评估
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    metrics['模型'] = name
    results.append(metrics)

    for k, v in metrics.items():
        if k != '模型':
            print(f"{k}：{v:.4f}" if isinstance(v, float) else f"{k}：{v}")

# ===================== 7. 输出结果 =====================
print(f"\n{'='*80}")
print(f"所有模型评估结果对比（标准化+PCA降维版）")
print(f"{'='*80}")
result_df = pd.DataFrame(results).round(4)
print(result_df)