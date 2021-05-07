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
    for i in tqdm(range(len(data1)), desc='filtering1...',
                  bar_format='{l_bar}{bar:30}{r_bar}'):
        if i not in idx1:
            line1 = clean_line(data1[i])
            line2 = clean_line(data2[i])
            fdata1.append(line1)
            fdata2.append(line2)
    print(f"After filtering, the number of sentence pair is {len(fdata1)}.")
    return fdata1, fdata2


def dif_filter(data1, data2, dif=10):
    ffdata1 = []
    ffdata2 = []
    for i in tqdm(range(len(data1)), desc='filtering2...', bar_format='{l_bar}{bar:30}{r_bar}'):
        line1 = data1[i].split()
        line2 = data2[i].split()
        dif_len = abs(len(line1) - len(line2))
        if dif_len < dif:
            ffdata1.append(data1[i])
            ffdata2.append(data2[i])
    return ffdata1, ffdata2


def make_source(data, word_to_id, max_sen_len, reverse=True, unk=True):
    source = []
    source_len = []
    if reverse:
        for line in tqdm(data, desc='making source input', bar_format='{l_bar}{bar:30}{r_bar}'):
            line = line.split() + ['</s>']
            ll = len(line)
            temp = [2] * (max_sen_len - ll)
            for word in line[::-1]:
                if word not in word_to_id:
                    if unk:
                        word = '<unk>'
                    else:
                        temp = [2] + temp
                        continue
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
                    if unk:
                        word = '<unk>'
                    else:
                        ll -= 1
                        continue
                temp.append(word_to_id[word])
            temp += [2]*(max_sen_len - ll)
            source.append(np.array(temp))
            source_len.append(ll)
    return np.array(source), np.array(source_len)


def make_target(data, word_to_id, max_sen_len, unk):
    target_input = []
    for line in tqdm(data, desc='making target input', bar_format='{l_bar}{bar:30}{r_bar}'):
        line = ['<s>'] + line.split()
        ll = len(line)
        temp = []
        for word in line:
            if word not in word_to_id:
                if unk:
                    word = '<unk>'
                else:
                    ll -= 1
                    continue
            temp.append(word_to_id[word])
        temp += [2]*(max_sen_len - ll)
        target_input.append(np.array(temp))
    target_output = []
    for line in tqdm(data, desc='making target output', bar_format='{l_bar}{bar:30}{r_bar}'):
        line = line.split() + ['</s>']
        ll = len(line)
        temp = []
        for word in line:
            if word not in word_to_id:
                if unk:
                    word = '<unk>'
                else:
                    ll -= 1
                    continue
            temp.append(word_to_id[word])
        temp += [2]*(max_sen_len - ll)
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


def position_masking(hs, ht, window_size):
    mini_batch, max_sen_len, lstm_dim = hs.size()
    seq_len = ht.shape[1]
    hhs = torch.zeros(mini_batch, seq_len, max_sen_len, lstm_dim)
    for i in range(seq_len):
        left = max(0, i - window_size)
        right = min(max_sen_len, i+window_size+1)
        hhs[:, i, left:right] = hs[:, left:right]
    return hhs


def test_eval(model, log_dir, mini_batch, lstm_layer, lstm_dim, max_sen_len,
              gpu, cuda, reverse, unk, trunc, epoch, id_to_de):
    mini_batch = int(mini_batch/4)

    # check dir, make dir
    test_dir = os.path.join(log_dir, 'test')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    # load test data
    print("Load the preprocessed test data..")
    if unk:
        if reverse:
            with open(f'datasets/preprocessed/test/test{trunc}_source_reverse_unk.pkl', 'rb') as fr:
                test_source_input, test_source_len = pickle.load(fr)
        else:
            with open(f'datasets/preprocessed/test/test{trunc}_source_unk.pkl', 'rb') as fr:
                test_source_input, test_source_len = pickle.load(fr)
        with open(f'datasets/preprocessed/test/test{trunc}_label_unk.pkl', 'rb') as fr:
            test_target_output = pickle.load(fr)
    else:
        if reverse:
            with open(f'datasets/preprocessed/test/test{trunc}_source_reverse.pkl', 'rb') as fr:
                test_source_input, test_source_len = pickle.load(fr)
        else:
            with open(f'datasets/preprocessed/test/test{trunc}_source.pkl', 'rb') as fr:
                test_source_input, test_source_len = pickle.load(fr)
        with open(f'datasets/preprocessed/test/test{trunc}_label.pkl', 'rb') as fr:
            test_target_output = pickle.load(fr)
    print("Complete. \n")

    print("Split the data into mini_batch..")
    test_src_input = make_batch(test_source_input, mini_batch)
    test_src_len = make_batch(test_source_len, mini_batch)
    test_tgt_output = make_batch(test_target_output, mini_batch)
    print("Complete.\n")

    test_src_input = torch.from_numpy(test_src_input)
    test_src_len = torch.from_numpy(test_src_len)
    test_tgt_output = torch.from_numpy(test_tgt_output)

    test_src_input = test_src_input.to(torch.int64)
    test_src_len = test_src_len.to(torch.int64)
    test_tgt_output = test_tgt_output.to(torch.int64)

    # test start
    cur = 0
    output = torch.zeros_like(test_src_input)   # output = (40, 64, 51)
    for batch_src_input, batch_src_len in tqdm(zip(test_src_input, test_src_len), total=len(test_src_input),
                                               bar_format='{l_bar}{bar:30}{r_bar}'):
        # init hidden state
        h_0 = torch.zeros(lstm_layer, mini_batch, lstm_dim)  # (4, 128, 1000)
        c_0 = torch.zeros(lstm_layer, mini_batch, lstm_dim)
        hidden = (h_0, c_0)
        # hidden = [state.detach() for state in hidden]

        tgt = torch.ones(mini_batch, 1)  # tgt = (mini_batch, 1)  ==> SOS tokens
        tgt = tgt.to(torch.int64)
        if gpu:
            device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
            batch_src_input = batch_src_input.to(device)
            batch_src_len = batch_src_len.to(device)
            tgt = tgt.to(device)
            hidden = [state.to(device) for state in hidden]

        # first decoder (past) output
        hht = torch.zeros(mini_batch, 1, lstm_dim)  # first time-step prev decoder context
        if gpu:
            device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
            hht = hht.to(device)

        for i in range(max_sen_len):
            out = model(batch_src_input, tgt, hidden, hht, batch_src_len)
            if gpu:
                device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
                out = out.to(device)  # out = (mini_batch, seq_len, tgt_vocab)
            pred = torch.max(out, dim=-1)[1]  # pred = (mini_batch, seq_len)
            tgt = torch.cat((tgt, pred[:, i].unsqueeze(1)), dim=1)
        output[cur] = tgt[:, 1:]  # output[cur] = (mini_batch, seq_len)
        cur += 1

    # make prediction.txt
    output = output.view(-1, max_sen_len)
    test_pred_output = []
    for line in output:
        sentence = ' '.join([id_to_de[int(idx)] for idx in line])
        sentence = sentence.replace('</s>', '').strip() + ' \n'
        test_pred_output.append(sentence)
    # make label.txt
    test_tgt_output = test_tgt_output.view(-1, max_sen_len)
    test_label = []
    for line in test_tgt_output:
        sentence = ' '.join([id_to_de[int(idx)] for idx in line])
        sentence = sentence.replace('</s>', '').strip() + ' \n'
        test_label.append(sentence)

    # save the prediction and label text.
    test_dir = os.path.join(log_dir, 'test')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    with open(os.path.join(test_dir, f'output_{epoch}.txt'), 'w', encoding='utf8') as fw:
        fw.writelines(test_pred_output)
    with open(os.path.join(test_dir, f'label_{epoch}.txt'), 'w', encoding='utf8') as fw:
        fw.writelines(test_label)
    print("Succeed to save the prediction and label text file!")
