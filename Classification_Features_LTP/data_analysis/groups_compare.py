import pandas as pd
import scipy.stats as stats

# 数据加载
data = "./1.xlsx"
df = pd.read_excel(data)

# 按组分开数据
ad_group = df[df['label'] == 'older']
control_group = df[df['label'] == 'young']

# 初始化结果存储
results = []

# 遍历特征列
# features = ['是', '的', '这个', '然后', '在', '了', '东西', '不']
# features = ['words_num', 'types_num', 'ttr', 'filled_num', 'filled_num', 'func_ratio', 'real_ratio']
# features = ['a', 'b', 'c', 'd', 'e', 'i', 'j', 'k',	'm', 'n', 'nd',	'nh', 'nl', 'ns', 'nt', 'nz', 'o', 'p', 'q', 'r', 'u', 'v', 'wp', 'ws', 'z']
# features = ['SBV','VOB','IOB','FOB','DBL','ATT','ADV','CMP','COO','POB','LAD','RAD','HED','sentences_num','mean_sentence_length','mean_dep_distance']
# features = ['words_num', 'types_num', 'ttr', 'filled_num', 'filled_num', 'func_ratio', 'real_ratio', 'sentences_num','mean_sentence_length','mean_dep_distance']
features = ['func_ratio', 'real_ratio']


for feature in features:
    # 提取两组数据
    ad_values = ad_group[feature]
    control_values = control_group[feature]

    # 正态性检验
    ad_normal = stats.shapiro(ad_values).pvalue > 0.05
    control_normal = stats.shapiro(control_values).pvalue > 0.05

    # 方差齐性检验
    equal_var = stats.levene(ad_values, control_values).pvalue > 0.05

    # 根据正态性和方差齐性选择检验方法
    if ad_normal and control_normal:
        t_stat, p_value = stats.ttest_ind(ad_values, control_values, equal_var=equal_var)
        test_used = "t-test"
    else:
        t_stat, p_value = stats.mannwhitneyu(ad_values, control_values)
        test_used = "Mann-Whitney U"

    # 记录结果
    results.append({
        'Feature': feature,
        'Test Used': test_used,
        'Statistic': t_stat,
        'p-value': p_value
    })

# 转为DataFrame输出结果
result_df = pd.DataFrame(results)
print(result_df)


