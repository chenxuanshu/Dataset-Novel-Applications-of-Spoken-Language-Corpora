import pandas as pd
import numpy as np
import json
from collections import Counter
import numpy as np



# 读取保存话语文本的json文件
def get_cws_from_json(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = json.load(file)['result_cws']
        return content
    
# 读取保存话语文本的json文件
def get_pos_from_json(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = json.load(file)['result_pos']
        return content



# 计算词汇量tokens，返回列表
def get_words_num(filename):
    content = get_cws_from_json(filename)
    words_num_list = []
    for i in range(len(content)):
        words_num_list.append(len(content[i]))
    return words_num_list



# 计算types数量，返回列表
def get_types_num(filename):
    content = get_cws_from_json(filename)
    types_num_list = []
    for i in range(len(content)):
        types_num_list.append(len(set(content[i])))
    return types_num_list


# 计算ttr，返回列表
def compute_ttr(filename):
    cws = get_cws_from_json(filename)
    ttr_list = []
    for i in range(cws.__len__()):
        tokens = len(cws[i])
        types = len(set(cws[i]))
        ttr_list.append(round(types/tokens, 2))
    return ttr_list



# 计算词频,从高到低排序
def word_sort(filename):
    content = get_cws_from_json(filename)
    word_list = []
    for i in range(content.__len__()):
        word_list = word_list + content[i]
    count = Counter(word_list)
    count_list = count.most_common()
    for i in range(20):
        print(count_list[i][0], count_list[i][1], format(count_list[i][1]/len(word_list), '.2%'),round(count_list[i][1]/len(content),2)) #计算均次
    return None


# 计算词性频率,从低到高排序
def word_class_sort(filename):
    content = get_pos_from_json(filename)
    word_list = []
    for i in range(content.__len__()):
        word_list = word_list + content[i]
    filtered_list = [item for item in word_list if item != 'wp']
    count = Counter(filtered_list)
    count_list = count.most_common()
    for each in count_list:
        print(each[0], each[1], format(each[1]/len(filtered_list), '.2%'), round(each[1]/len(content)))
    return None



#计算词汇填充物数量和比重，返回两个列表
def compute_person_filled_num(filename):
    content = get_cws_from_json(filename)
    filled_num_list = []
    filled_ration_list = []
    filled_words = ['啊', '嗯', '呃', '哦', '额', '这个', '那个', '就是', '嘛','吧']
    for person_words in content:
        filled_num = 0
        for word in person_words:
            if word in filled_words:
                filled_num += 1
        filled_num_list.append(filled_num)
        filled_ration_list.append(round(filled_num/len(person_words), 2))
    return filled_num_list, filled_ration_list




# 计算功能词和实意词频率，返回列表
def get_func_real_ratio(filename):
    pos_content = get_pos_from_json(filename)
    func_ratio = []
    real_ratio = []
    for i in range(pos_content.__len__()):
        pos_list = pos_content[i]
        func_num = 0
        real_num = 0
        for pos in pos_list:
            if pos in ['r','p', 'u', 'd', 'e', 'o', 'k']:
                func_num += 1
            else:
                real_num += 1
        func_ratio.append(round(func_num/len(pos_list), 2))
        real_ratio.append(round(real_num/len(pos_list), 2))
    return func_ratio, real_ratio


# 计算词性分布比重
def get_pos_ratio(filename):
    pos_list = {
        'a': [],
        'b': [],
        'c': [],
        'd': [],
        'e': [],
        'g': [],
        'h': [],
        'i': [],
        'j': [],
        'k': [],
        'm': [],
        'n': [],
        'nd': [],
        'nh': [],
        'ni': [],
        'nl': [],
        'ns': [],
        'nt': [],
        'nz': [],
        'o': [],
        'p': [],
        'q': [],
        'r': [],
        'u': [],
        'v': [],
        'wp': [],
        'ws': [],
        'x': [],
        'z': []
    }
    pos_content = get_pos_from_json(filename)
    for person_pos in pos_content:
        count_list = Counter(person_pos).most_common()
        for pos_name, ration_list in pos_list.items():
            for pos, count in count_list:
                if pos == pos_name:
                    ration_list.append(round(count/len(person_pos), 2))
                    # ration_list.append(count)
                    break
            else:
                ration_list.append(0) # 未出现该词性的词，计为0
    return pos_list



# 计算词性分布比重
def get_high_freq_words_ratio(filename):
    cws_list = {
        '是': [],
        '的': [],
        '这个': [],
        '然后': [],
        '在': [],
        '了': [],
        '东西': [],
        '不': []
    }
    cws_content = get_cws_from_json(filename)
    for person_cws in cws_content:
        count_list = Counter(person_cws).most_common()
        for cws_name, ration_list in cws_list.items():
            for cws, count in count_list:
                if cws == cws_name:
                    ration_list.append(round(count/len(person_cws), 2))
                    # ration_list.append(count)
                    break
            else:
                ration_list.append(0) # 未出现该词性的词，计为0
    return cws_list



# 计算总体某一词性中的词类排名
def compute_words_in_class(cws_filename, pos_filename):
    cws_content = get_cws_from_json(cws_filename)
    pos_content = get_pos_from_json(pos_filename)
    n_word_list = []
    for i in range(len(pos_content)):
        for j in range(len(pos_content[i])):
            if pos_content[i][j] == 'c':  # 可替换为v、r等其他词性
                n_word_list.append(cws_content[i][j])
    count = Counter(n_word_list)
    return count.most_common()








if __name__ == '__main__':

    cws_control_filename = '../LTP_analyze_data/cws_control.json'
    cws_ad_filename = '../LTP_analyze_data/cws_ad.json'
    cws_older_filename = '../LTP_analyze_data/cws_older.json'
    pos_control_filename = '../LTP_analyze_data/pos_control.json'
    pos_ad_filename = '../LTP_analyze_data/pos_ad.json'
    pos_older_filename = '../LTP_analyze_data/pos_older.json'

#计算词类比率并保存到Excel文件
    # control_pos_list = get_pos_ratio(pos_control_filename)
    # older_pos_list = get_pos_ratio(pos_older_filename)
    # ad_pos_list = get_pos_ratio(pos_ad_filename)
    # #删掉都为0的词类
    # blank_pos = []
    # for keys in ad_pos_list.keys():
    #     if np.sum(ad_pos_list[keys]) == 0 and np.sum(control_pos_list[keys]) == 0 and np.sum(older_pos_list[keys]) == 0:
    #         blank_pos.append(keys)
    # for pos_name in blank_pos:
    #     ad_pos_list.pop(pos_name)
    #     control_pos_list.pop(pos_name)
    #     older_pos_list.pop(pos_name)

    # ad_df = pd.DataFrame(ad_pos_list)
    # ad_df.to_excel('ad_pos_ratio.xlsx')
    # control_df = pd.DataFrame(control_pos_list)
    # control_df.to_excel('control_pos_ratio.xlsx')
    # older_df = pd.DataFrame(older_pos_list)
    # older_df.to_excel('older_pos_ratio.xlsx')

# 计算高频词比率并保存到Excel文件
    # words_control_list = get_high_freq_words_ratio(cws_control_filename)
    # words_older_list = get_high_freq_words_ratio(cws_older_filename)
    # words_ad_list = get_high_freq_words_ratio(cws_ad_filename)
    # control_df = pd.DataFrame(words_control_list)
    # older_df = pd.DataFrame(words_older_list)
    # ad_df = pd.DataFrame(words_ad_list)
    # control_df.to_excel('control_words_ratio.xlsx')
    # older_df.to_excel('older_words_ratio.xlsx')
    # ad_df.to_excel('ad_words_ratio.xlsx')


# words_num = get_words_num(cws_ad_filename)
# types_num = get_types_num(cws_older_filename)
# ttr = compute_ttr(cws_older_filename)
# filled_num, filled_ration = compute_person_filled_num(cws_control_filename)
# func_ratio, real_ratio = get_func_real_ratio(pos_older_filename)

# vocabulary_features = {
#     "words_num": words_num,
#     "types_num": types_num,
#     "ttr": ttr,
#     "filled_num": filled_num,
#     "filled_ration": filled_ration,
#     "func_ratio": func_ratio,
#     "real_ratio": real_ratio,
# }
# df = pd.DataFrame(vocabulary_features)
# df.to_excel('features2.xlsx')

# word_class_sort(pos_ad_filename)
# word_sort(cws_ad_filename)