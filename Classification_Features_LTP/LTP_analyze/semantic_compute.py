import pandas as pd
import numpy as np
import json
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import OpenHowNet
import torch
from umap import UMAP
from transformers import pipeline
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer

# 读取保存口语转录文本的json文件
def get_text_from_json(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = json.load(file)
        text = []
        for k,v in content.items():
            text.append(v)
        return text

# 读取保存语义依存分析结果的json文件
def get_sdp_from_json(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = json.load(file)['result_sdp']
        return content
    
# 读取保存语义角色标注的json文件
def get_srl_from_json(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = json.load(file)['result_srl']
        return content
    
# 读取保存分词结果的json文件
def get_cws_from_json(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        words = json.load(file)['result_cws']
        return words
    
# 读取保存分句结果的json文件
def get_sentences_from_json(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        sentences = json.load(file)['sentences']
        return sentences
    

# 计算单个样本的词汇语义类别，传入一个词列表
def compute_semantic_category(words_list, hownet_dict):
    semantic_category_counter = Counter()
    for word in words_list:
        sememes = hownet_dict.get_sememes_by_word(word)
        if sememes:
            semantic_category_counter[sememes[0]['sememes'][0]] += 1
        else:
            semantic_category_counter['unknown'] += 1
    return semantic_category_counter

# 统计一个组中特定此类的分布情况， DeChinese|构助、FuncWord|功能词、future|将来
def compute_group_word_distribution(filename, hownet_dict):
    content = get_cws_from_json(filename)
    words_list = []
    function_word_counter = Counter()
    for words in content:
        words_list += words
    for word in words_list:
        sememes = hownet_dict.get_sememes_by_word(word)
        if sememes and str(sememes[0]['sememes'][0]) == 'FuncWord|功能词':
            function_word_counter[word] += 1
    return function_word_counter



# 统计一个组总体的语义类别，传入一个二维词语列表
def compute_group_semantic_category(words_list, hownet_dict):
    counter_list = []
    for words in words_list:
        category_counter = compute_semantic_category(words, hownet_dict)
        counter_list.append(category_counter)
    group_counter = Counter()
    for counter in counter_list:
        group_counter += counter
    return group_counter


# 统计一个组总体平均的信息密度，即谓语动词的数量
def compute_idea_density(filename):
    srl_result = get_srl_from_json(filename)
    idea_density_list = []
    for person_srl in srl_result:
        idea_density_list.append(len(person_srl))
    # return np.mean(idea_density_list)
    return idea_density_list



# 计算个人的平均相邻句相似度
def compute_sentence_similarity(sentences, model):
    embeddings = model.encode(sentences)
    scores = []
    for i in range(len(sentences)-1):
        cos_sim = cosine_similarity(embeddings[i].reshape(1, -1), embeddings[i+1].reshape(1, -1))[0][0]
        scores.append(cos_sim)
    return sum(scores) / len(scores)  # 平均相邻句相似度


# 计算一个组的平均相邻句相似度
def compute_semantic_coherence(filename):
    sentences = get_sentences_from_json(filename)
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    sim_list = []
    for person_sentences in sentences:
        sim_list.append(compute_sentence_similarity(person_sentences, model))
    # return np.mean(sim_list)
    return sim_list



# 计算语篇主题一致性
def compute_topic_coherence(filename):
    sentences = get_sentences_from_json(filename)
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    hdbscan_model = HDBSCAN(min_cluster_size=2, min_samples=1)
    umap_model = UMAP(n_components=2) #加这一行，避免稀疏矩阵降维出错
    topic_model = BERTopic(language="multilingual",hdbscan_model=hdbscan_model,umap_model=umap_model)
    main_topic_ratio_list = []
    for person_sentences in sentences:
        if len(person_sentences) < 4:
            # 如果句子数量小于等于3条，就跳过，避免报错
            main_topic_ratio_list.append(1.0)
        else:
            embeddings = model.encode(person_sentences)
            topics, probs = topic_model.fit_transform(person_sentences, embeddings)
            topic_counts = Counter(topics)
            main_topic, main_topic_count = topic_counts.most_common(1)[0]
            main_topic_ratio = main_topic_count / len(person_sentences)
            main_topic_ratio_list.append(main_topic_ratio)
    # return np.mean(main_topic_ratio_list)
    return main_topic_ratio_list




#计算各种语义关系比重，从高到低排序
def compute_semantic_relation(filename):
    content = get_sdp_from_json(filename)
    semantic_relation = []
    for i in range(content.__len__()):
        semantic_relation = semantic_relation + content[i]['label']
    count_list = Counter(semantic_relation).most_common()
    for i in range(len(count_list)):
        print(count_list[i][0], format(count_list[i][1]/len(semantic_relation), '.2%'), round(count_list[i][1]/len(content), 2))

    return None




# 计算各种语义依存关系分布比重
def get_sdp_ratio(filename):
    sdp_list = {
        'dTIME': [], 
        'rEXP': [], 
        'mNEG':[], 
        'TIME': [], 
        'mRELA': [], 
        'eSUCC': [], 
        'dREAS': [], 
        'DATV': [], 
        'dLINK': [], 
        'rPAT': [], 
        'Root': [], 
        'rAGT': [], 
        'eCOO': [], 
        'mDEPD': [], 
        'REAS': [], 
        'MEAS': [], 
        'AGT': [], 
        'LINK': [], 
        'dCONT': [], 
        'STAT': [], 
        'SCO': [], 
        'rCONT': [], 
        'PAT': [], 
        'mPUNC': [], 
        'MANN': [], 
        'rTIME': [], 
        'LOC': [], 
        'dMANN': [], 
        'dEXP': [], 
        'TOOL': [], 
        'ePREC': [], 
        'EXP': [], 
        'FEAT': [], 
        'dFEAT': [], 
        'CONT': []
        }

    sdp_content = get_sdp_from_json(filename)
    for person_sdp in sdp_content:
        count_list = Counter(person_sdp['label']).most_common()
        for sdp_name, ration_list in sdp_list.items():
            for sdp, count in count_list:
                if sdp == sdp_name:
                    ration_list.append(round(count/len(person_sdp['label']), 2))
                    # ration_list.append(count)
                    break
            else:
                ration_list.append(0) # 未出现该词性的词，计为0
    return sdp_list

# 计算语义层面的一些其他特征
def compute_semantic_features(srl_filename, sentences_filename):
    idea_density_list = compute_idea_density(srl_filename)
    semantic_coherence = compute_semantic_coherence(sentences_filename)
    topic_coherence = compute_topic_coherence(sentences_filename)
    semantic_features = {
        'idea_density': idea_density_list,
       'semantic_coherence': semantic_coherence,
        'topic_coherence': topic_coherence
    }
    df = pd.DataFrame(semantic_features)
    df.to_excel('semantic_features.xlsx')
    return None


if __name__ == '__main__':
    text_control_file = '../text_data/control_text.json'
    text_older_file = '../text_data/older_text.json'
    text_ad_file = '../text_data/ad_text.json'

    sdp_ad_file = '../LTP_analyze_data/sdp_ad.json'
    sdp_control_file = '../LTP_analyze_data/sdp_control.json'
    sdp_older_file = '../LTP_analyze_data/sdp_older.json'

    ad_dep_file = '../LTP_analyze_data/dep_ad.json'
    control_dep_file = '../LTP_analyze_data/dep_control.json'
    older_dep_file = '../LTP_analyze_data/dep_older.json'

    control_cws_file = '../LTP_analyze_data/cws_control.json'
    older_cws_file = '../LTP_analyze_data/cws_older.json'
    ad_cws_file = '../LTP_analyze_data/cws_ad.json'

    srl_control_file = '../LTP_analyze_data/srl_control.json'
    srl_older_file = '../LTP_analyze_data/srl_older.json'
    srl_ad_file = '../LTP_analyze_data/srl_ad.json'

    sentences_control_file = '../LTP_analyze_data/sentences_control.json'
    sentences_older_file = '../LTP_analyze_data/sentences_older.json'
    sentences_ad_file = '../LTP_analyze_data/sentences_ad.json'

    control_words = get_cws_from_json(control_cws_file)
    older_words = get_cws_from_json(older_cws_file)
    ad_words = get_cws_from_json(ad_cws_file)


    OpenHowNet.download()     # 下载OpenHowNet词典
    hownet_dict = OpenHowNet.HowNetDict()


    # 统计一个组总体的语义类别,传入一个二维词语列表
    control_category_counter = compute_group_semantic_category(control_words, hownet_dict).most_common()
    print(control_category_counter)
    # 计算信息密度
    idea_density_ad = compute_idea_density(srl_ad_file)
    idea_density_control = compute_idea_density(srl_control_file)
    idea_density_older = compute_idea_density(srl_older_file)
    print('ad idea density:', idea_density_ad)
    print('control idea density:', idea_density_control)
    print('older idea density:', idea_density_older)

    # 计算相邻句相似度
    print(compute_semantic_coherence(sentences_ad_file))

    # 计算语篇主题一致性
    print(compute_topic_coherence(sentences_control_file))

    # 统计一个类别功能词的分布情况
    function_word_counter = compute_group_word_distribution(ad_cws_file, hownet_dict)
    print(function_word_counter.most_common()[:10])

    # 计算语义层面的一些其他特征
    compute_semantic_features(srl_ad_file, sentences_ad_file)

    # 计算各种语义关系比重并保存到Excel文件
    control_sdp_list = get_sdp_ratio(sdp_control_file)
    older_sdp_list = get_sdp_ratio(sdp_older_file)
    ad_sdp_list = get_sdp_ratio(sdp_ad_file)


    # 删掉都为0的语义关系
    blank_sdp = []
    for keys in ad_sdp_list.keys():
        if np.sum(ad_sdp_list[keys]) == 0 and np.sum(control_sdp_list[keys]) == 0 and np.sum(older_sdp_list[keys]) == 0:
            blank_sdp.append(keys)
    for sdp_name in blank_sdp:
        ad_sdp_list.pop(sdp_name)
        control_sdp_list.pop(sdp_name)
        older_sdp_list.pop(sdp_name)

    ad_df = pd.DataFrame(ad_sdp_list)
    ad_df.to_excel('ad_sdp_ratio.xlsx')
    control_df = pd.DataFrame(control_sdp_list)
    control_df.to_excel('control_sdp_ratio.xlsx')
    older_df = pd.DataFrame(older_sdp_list)
    older_df.to_excel('older_sdp_ratio.xlsx')
    compute_semantic_relation(sdp_ad_file)


    