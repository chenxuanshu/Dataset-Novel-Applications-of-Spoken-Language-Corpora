import numpy as np
import pandas as pd
import json

from ltp import LTP
from ltp import StnSplit

ltp = LTP()
SentenceSplitter = StnSplit()




# 读取保存话语文本的json文件
def get_text_from_json(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        ad_text = json.load(file)
        ad_all = []
        for k, v in ad_text.items():
            ad_all.append(v)
        return ad_all



# 调用LTP进行分词、词性标注、依存句法分析、语义角色标注 并保存结果至json文件
def get_text_ltp_result(filename, ltp):
    text_list = get_text_from_json(filename)
    result = ltp.pipeline(text_list, tasks=['cws','pos','srl','dep', 'sdp'])
    result_cws = result.cws
    result_pos = result.pos
    result_srl = result.srl
    result_dep = result.dep
    result_sdp = result.sdp
    cws_dict = {
        "result_cws": result_cws,
    }
    pos_dict = {
        "result_pos": result_pos,
    }
    srl_dict = {
        "result_srl": result_srl,
    }
    dep_dict = {
        "result_dep": result_dep,
    }
    sdp_dict = {
        "result_sdp": result_sdp,
    }
    with open('cws_ad.json', 'w', encoding='utf-8') as f:
        json.dump(cws_dict, f, ensure_ascii=False)
    with open('pos_ad.json', 'w', encoding='utf-8') as f:
        json.dump(pos_dict, f, ensure_ascii=False)
    with open('srl_ad.json', 'w', encoding='utf-8') as f:
        json.dump(srl_dict, f, ensure_ascii=False)
    with open('dep_ad.json', 'w', encoding='utf-8') as f:
        json.dump(dep_dict, f, ensure_ascii=False)
    with open('sdp_ad.json', 'w', encoding='utf-8') as f:
        json.dump(sdp_dict, f, ensure_ascii=False)
    return None


# 调用LTP进行分句
def get_sentences_split(filename):
    text_list = get_text_from_json(filename)
    sentences = []
    for text in text_list:
        result = SentenceSplitter.split(text)
        sentences.append(result)
    return sentences




if __name__ == '__main__':
    
    control_filename = '../text_data/control_text.json'
    ad_filename = '../text_data/ad_text.json'
    older_filename = '../text_data/older_text.json'

    control_sentences = get_sentences_split(control_filename)
    ad_sentences = get_sentences_split(ad_filename)
    older_sentences = get_sentences_split(older_filename)

    control_sentences_dict = {
        "sentences": control_sentences,
    }
    ad_sentences_dict = {
        "sentences": ad_sentences,
    }
    older_sentences_dict = {
        "sentences": older_sentences,
    }
    with open('sentences_control.json', 'w', encoding='utf-8') as f:
        json.dump(control_sentences_dict, f, ensure_ascii=False)
    with open('sentences_ad.json', 'w', encoding='utf-8') as f:
        json.dump(ad_sentences_dict, f, ensure_ascii=False)
    with open('sentences_older.json', 'w', encoding='utf-8') as f:
        json.dump(older_sentences_dict, f, ensure_ascii=False)


    # get_text_ltp_result(ad_filename, ltp)
