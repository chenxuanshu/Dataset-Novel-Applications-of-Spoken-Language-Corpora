import pandas as pd
import numpy as np
import json
from collections import Counter
from collections import defaultdict





# 读取保存话语文本的json文件
def get_cws_from_json(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = json.load(file)['result_cws']
        return content

# 读取保存话语文本的json文件
def get_dep_from_json(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = json.load(file)['result_dep']
        return content
    
# 读取保存话语文本的json文件
def get_sentences_from_json(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = json.load(file)['sentences']
        return content



#计算各种句法关系比重，从高到低排序
def compute_syntax_relation(filename):
    content = get_dep_from_json(filename)
    syntax_relation = []
    for i in range(content.__len__()):
        syntax_relation = syntax_relation + content[i]['label']
    count = Counter(syntax_relation)
    count_list = count.most_common()
    for i in range(count_list.__len__()):
        print(count_list[i][0], format(count_list[i][1]/syntax_relation.__len__(), '.2%'), round(count_list[i][1]/len(content), 2))
    return None


#计算平均句长, 返回列表
def compute_mean_sentence_length_list(cws_filename, sentence_filename):
    cws_content = get_cws_from_json(cws_filename)
    sentence_content = get_sentences_from_json(sentence_filename)
    mean_sentence_length_list = []
    for i in range(sentence_content.__len__()):
        mean_sentence_length_list.append(round(cws_content[i].__len__()/sentence_content[i].__len__(),2))
    return mean_sentence_length_list

# 统计平均句子数，返回列表
def compute_mean_sentence_num_list(sentence_filename):
    sentence_content = get_sentences_from_json(sentence_filename)
    sentence_length_list = []
    for person in sentence_content:
        sentence_length_list.append(len(person))
    return sentence_length_list


#计算一个句子的平均依存距离
def compute_dep_distance(head_list):
    distance = 0
    for i in range(len(head_list)):
        distance += abs(i + 1 - head_list[i])
    return distance / len(head_list)

#计算平均依存距离，返回列表
def compute_avg_dep_distance(filename):
    dep_content = get_dep_from_json(filename)
    avg_dep_length_list = []
    for person in dep_content:
        avg_dep_length_list.append(round(compute_dep_distance(person['head']), 2))
    return avg_dep_length_list



# 统计最大依存距离
def compute_max_dep_distance(filename):
    dep_content = get_dep_from_json(filename)
    max_dep_distance_list = []
    for person in dep_content:
        max_dep_distance = 0
        for i in range(len(person['head'])):
            if person['head'][i] - i > max_dep_distance:
                max_dep_distance = person['head'][i] - i
        max_dep_distance_list.append(max_dep_distance)
    return max_dep_distance_list

# 计算一个人的最大依存树深度
def compute_max_dep_depth(head):
    # 构建依存树（子节点列表）
    tree = defaultdict(list)
    root = -1
    for idx, h in enumerate(head):
        if h == 0:
            root = idx # ROOT 节点编号（从 0 开始）
        else:
            tree[h - 1].append(idx) # LTP/其他工具中head是从1开始的，所以减1对齐Python索引
    # 递归计算最大深度
    def dfs(node):
        if not tree[node]:
            return 1
        return 1 + max(dfs(child) for child in tree[node])
    if root == -1:
        raise ValueError("No root found in dependency tree!")
    return dfs(root)


# 统计一个组的最大依存树深度
def compute_max_dep_depth_group(filename):
    dep_content = get_dep_from_json(filename)
    max_dep_depth_list = []
    for person in dep_content:
        max_dep_depth = compute_max_dep_depth(person['head'])
        max_dep_depth_list.append(max_dep_depth)
    # return np.mean(compute_max_dep_depth_list)
    return max_dep_depth_list



# 计算各种句法关系分布比重
def get_dep_ratio(filename):
    dep_list = {
        'SBV': [],
        'VOB': [],
        'IOB': [],
        'FOB': [],
        'DBL': [],
        'ATT': [],
        'ADV': [],
        'CMP': [],
        'COO': [],
        'POB': [],
        'LAD': [],
        'RAD': [],
        'IS': [],
        'HED': []
    }
    dep_content = get_dep_from_json(filename)
    for person_dep in dep_content:
        count_list = Counter(person_dep['label']).most_common()
        for dep_name, ration_list in dep_list.items():
            for dep, count in count_list:
                if dep == dep_name:
                    ration_list.append(round(count/len(person_dep['label']), 2))
                    # ration_list.append(count)
                    break
            else:
                ration_list.append(0) # 未出现该词性的词，计为0
    return dep_list





if __name__ == '__main__':
    cws_ad_filename = '../LTP_analyze_data/cws_ad.json'
    cws_control_filename = '../LTP_analyze_data/cws_control.json'
    cws_older_filename = '../LTP_analyze_data/cws_older.json'
    
    sentence_ad_filename = '../LTP_analyze_data/sentences_ad.json'
    sentence_control_filename = '../LTP_analyze_data/sentences_control.json'
    sentence_older_filename = '../LTP_analyze_data/sentences_older.json'

    dep_ad_filename = '../LTP_analyze_data/dep_ad.json'
    dep_control_filename = '../LTP_analyze_data/dep_control.json'
    dep_older_filename = '../LTP_analyze_data/dep_older.json'


    # print(np.mean(sentence_length_list), np.std(sentence_length_list))


    # control_dep_list = get_dep_ratio(dep_control_filename)
    # older_dep_list = get_dep_ratio(dep_older_filename)
    # ad_dep_list = get_dep_ratio(dep_ad_filename)

    # #删掉比率都为0的句法关系
    # blank_list = []
    # for dep_name, ad_ratio_list in ad_dep_list.items():
    #     if np.sum(ad_dep_list[dep_name]) == 0 and np.sum(control_dep_list[dep_name]) == 0 and np.sum(older_dep_list[dep_name]) == 0:
    #         blank_list.append(dep_name)
    # for dep_name in blank_list:
    #     del ad_dep_list[dep_name]
    #     del control_dep_list[dep_name]
    #     del older_dep_list[dep_name]

    # ad_dep_df = pd.DataFrame(ad_dep_list)
    # control_dep_df = pd.DataFrame(control_dep_list)
    # older_dep_df = pd.DataFrame(older_dep_list)
    # ad_dep_df.to_excel('ad_dep_ratio.xlsx', index=False)
    # control_dep_df.to_excel('control_dep_ratio.xlsx', index=False)
    # older_dep_df.to_excel('older_dep_ratio.xlsx', index=False)



# 计算最大依存距离
    # max_dep_distance = compute_max_dep_distance(dep_ad_filename)

# 计算最大依存树深度
    # max_dep_depth = compute_max_dep_depth_group(dep_ad_filename)

# 计算平均句子数
    # sentences_num = compute_mean_sentence_num_list(sentence_ad_filename)

#计算平均句长
    # mean_sentence_length = compute_mean_sentence_length_list(cws_control_filename, sentence_control_filename)

#计算平均依存距离
    # mean_dep_distance = compute_avg_dep_distance(dep_control_filename)
    
    
    # control_df = pd.DataFrame({'sentences_num': sentences_num,
    #                            'mean_sentence_length': mean_sentence_length,
    #                            'mean_dep_distance': mean_dep_distance,})
    
    # control_df.to_excel('older_syntactic_features.xlsx', index=False)

# compute_syntax_relation(dep_ad_filename)