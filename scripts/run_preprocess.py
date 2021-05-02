import os
import sys
sys.path.append(os.getcwd())
from src.functions import *


# en_to_id, id_to_en = english_dict()
# de_to_id, id_to_de = german_dict()
# dict_en_de = get_prediction_result()

with open('datasets/preprocessed/vocab.en.pkl', 'rb') as fr:
    en_to_id, id_to_en = pickle.load(fr)
with open('datasets/preprocessed/vocab.de.pkl', 'rb') as fr:
    de_to_id, id_to_de = pickle.load(fr)


# # Load English training dataset
# file = 'datasets/raw/train.en'
# train_en = load_data(file)
# # Load German training dataset
# file = 'datasets/raw/train.de'
# train_de = load_data(file)
# # filter out the data longer than the length 50
# f_train_en, f_train_de = len_filter(train_en, train_de, max_sen_len=50)
# with open('datasets/preprocessed/f_train.en.pkl', 'wb') as fw:
#     pickle.dump(f_train_en, fw)
# with open('datasets/preprocessed/f_train.de.pkl', 'wb') as fw:
#     pickle.dump(f_train_de, fw)

with open('datasets/preprocessed/f_train.en.pkl', 'rb') as fr:
    f_train_en = pickle.load(fr)
with open('datasets/preprocessed/f_train.de.pkl', 'rb') as fr:
    f_train_de = pickle.load(fr)


source_input, source_len = make_source(f_train_en, en_to_id, max_sen_len=51, reverse=True, unk=True)
target_input, target_output = make_target(f_train_de, de_to_id, max_sen_len=51, unk=True)
with open('datasets/preprocessed/source_reverse_unk.pkl', 'wb') as fw:
    pickle.dump((source_input, source_len), fw)
with open('datasets/preprocessed/target_unk.pkl', 'wb') as fw:
    pickle.dump((target_input, target_output), fw)

# with open('datasets/preprocessed/source_reverse_unk.pkl', 'rb') as fr:
#     source_input, source_len = pickle.load(fr)
# with open('datasets/preprocessed/target_unk.pkl', 'rb') as fr:
#     target_input, target_output = pickle.load(fr)
