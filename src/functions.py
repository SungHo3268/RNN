import numpy as np
import pickle
import torch
from tqdm.auto import tqdm
import os
import sys
import re
sys.path.append(os.getcwd())


def load_data(file):
    with open(file, 'r', encoding='utf8') as fr:
        data = fr.readlines()
    return data


def english_dict():
    # Get English dictionary
    en_to_id = {}
    id_to_en = {}
    with open('datasets/raw/vocab.50K.en', 'r', encoding='utf8') as fr:
        data = fr.readlines()
        for word in data:
            en_to_id[word.strip()] = len(en_to_id)
            id_to_en[len(id_to_en)] = word.strip()
    # Save English dictionary
    with open('datasets/preprocessed/vocab.en.pkl', 'wb') as fw:
        pickle.dump((en_to_id, id_to_en), fw)
    return en_to_id, id_to_en


def german_dict():
    # Get German dictionary
    de_to_id = {}
    id_to_de = {}
    with open('datasets/raw/vocab.50K.de', 'r', encoding='utf8') as fr:
        data = fr.readlines()
        for word in data:
            de_to_id[word.strip()] = len(de_to_id)
            id_to_de[len(id_to_de)] = word.strip()
    # Save German dictionary
    with open('datasets/preprocessed/vocab.de.pkl', 'wb') as fw:
        pickle.dump((de_to_id, id_to_de), fw)
    return de_to_id, id_to_de


def get_prediction_result():
    # Get prediction en-de
    dic = []
    with open('datasets/raw/dict.en-de', 'r', encoding='utf8') as fr:
        data = fr.readlines()
        for line in data:
            dic.append(line.split())
    # Save prediction en-de
    dic = np.array(dic)
    with open('datasets/preprocessed/dict.en-de.pkl', 'wb') as fw:
        pickle.dump(dic, fw)
    return dic


def clean_line(line):
    """
    Strip leading and trailing spaces
    """
    line = re.sub('(^\s+|\s$)', '', line)
    return line


def len_filter(data1, data2, max_sen_len=50):
    idx1 = set()
    idx2 = set()
    for i, sen in tqdm(enumerate(data1), total=len(data1),
                       desc='filtering data1...', bar_format='{l_bar}{bar:30}{r_bar}'):
        sen = sen.split()
        sen_len = len(sen)
        if sen_len > max_sen_len:
            idx1.add(i)
    for i, sen in tqdm(enumerate(data2), total=len(data2),
                       desc='filtering data2...', bar_format='{l_bar}{bar:30}{r_bar}'):
        sen = sen.split()
        sen_len = len(sen)
        if sen_len > max_sen_len:
            idx2.add(i)

    idx1.update(idx2)
    fdata1 = []
    fdata2 = []
    for i in tqdm(range(len(data1)), desc='filtering...',
                  bar_format='{l_bar}{bar:30}{r_bar}'):
        if i not in idx1:
            line1 = clean_line(data1[i])
            line2 = clean_line(data2[i])
            fdata1.append(line1)
            fdata2.append(line2)
    print(f"After filtering, the number of sentence pair is {len(fdata1)}.")
    return fdata1, fdata2


def make_source(data, word_to_id, max_sen_len, reverse=True):
    source = []
    source_len = []
    if reverse:
        for line in tqdm(data, desc='making source input', bar_format='{l_bar}{bar:30}{r_bar}'):
            line = line.split() + ['</s>']
            ll = len(line)
            temp = [0] * (max_sen_len - ll)
            for word in line[::-1]:
                if word not in word_to_id:
                    word = '<unk>'
                temp.append(word_to_id[word])
            source.append(np.array(temp))
            source_len.append(ll)
    else:
        for line in tqdm(data, desc='making source input', bar_format='{l_bar}{bar:30}{r_bar}'):
            line = line.split() + ['</s>']
            ll = len(line)
            temp = []
            for word in line:
                if word not in word_to_id:
                    word = '<unk>'
                temp.append(word_to_id[word])
            temp += [0]*(max_sen_len - ll)
            source.append(np.array(temp))
            source_len.append(ll)
    return np.array(source), np.array(source_len)


def make_target(data, word_to_id, max_sen_len):
    target_input = []
    for line in tqdm(data, desc='making target input', bar_format='{l_bar}{bar:30}{r_bar}'):
        line = ['<s>'] + line.split()
        ll = len(line)
        temp = []
        for word in line:
            if word not in word_to_id:
                word = '<unk>'
            temp.append(word_to_id[word])
        temp += [0]*(max_sen_len - ll)
        target_input.append(np.array(temp))
    target_output = []
    for line in tqdm(data, desc='making target output', bar_format='{l_bar}{bar:30}{r_bar}'):
        line = line.split() + ['</s>']
        ll = len(line)
        temp = []
        for word in line:
            if word not in word_to_id:
                word = '<unk>'
            temp.append(word_to_id[word])
        temp += [0]*(max_sen_len - ll)
        target_output.append(np.array(temp))
    return np.array(target_input), np.array(target_output)


def make_batch(data, mini_batch):
    batch_size = len(data) // mini_batch
    batch_data = []
    for i in range(batch_size):
        batch_data.append(data[i*mini_batch: (i+1)*mini_batch])
    return np.array(batch_data)


def make_position_vec(pt, hs, src_len, window_size, gpu, cuda):           # hhs is zero tensor placeholder
    mini_batch, seq_len = pt.size()
    hhs = torch.zeros_like(hs)                  # hhs = (mini_batch, max_sen_len(window), lstm_dim)
    left_min = torch.zeros(mini_batch)
    if gpu:                                # hhs = (mini_batch, max_sen_len(window), lstm_dim)
        hhs = hhs.to(torch.device(f'cuda:{cuda}'))
        left_min = left_min.to(torch.device(f'cuda:{cuda}'))
    for i in range(seq_len):
        left = torch.max(left_min, pt[:, i]-window_size).to(torch.int64)     # left = (mini_batch, )
        right = torch.min(src_len-1, pt[:, i]+window_size).to(torch.int64)                  # right = (mini_batch, )
        for j in range(mini_batch):
            hhs[j, left[i]:right[i]+1] = hs[j, left[i]:right[i]+1]
    return hhs


def softmax_masking(data, neg_inf=-1e+06):
    mask = (data == 0)
    mask = mask.to(torch.int64) * neg_inf
    return mask*data
