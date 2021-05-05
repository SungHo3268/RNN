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
# ff_train_en, ff_train_de = dif_filter(f_train_en, f_train_de, dif=10)
# with open('datasets/preprocessed/ff_train.en.pkl', 'wb') as fw:
#     pickle.dump(ff_train_en, fw)
# with open('datasets/preprocessed/ff_train.de.pkl', 'wb') as fw:
#     pickle.dump(ff_train_de, fw)
with open('datasets/preprocessed/ff_train.en.pkl', 'rb') as fr:
    ff_train_en = pickle.load(fr)
with open('datasets/preprocessed/ff_train.de.pkl', 'rb') as fr:
    ff_train_de = pickle.load(fr)


# source_input, source_len = make_source(ff_train_en, en_to_id, max_sen_len=51, reverse=True, unk=True)
# with open('datasets/preprocessed/source_reverse_unk.pkl', 'wb') as fw:
#     pickle.dump((source_input, source_len), fw)
# target_input, target_output = make_target(ff_train_de, de_to_id, max_sen_len=51, unk=True)
# with open('datasets/preprocessed/target_unk.pkl', 'wb') as fw:
#     pickle.dump((target_input, target_output), fw)
with open('datasets/preprocessed/source_reverse_unk.pkl', 'rb') as fr:
    source_input, source_len = pickle.load(fr)
with open('datasets/preprocessed/target_unk.pkl', 'rb') as fr:
    target_input, target_output = pickle.load(fr)


###################### Preprocess test data ######################
# Load English test dataset
file = 'datasets/raw/newstest2014.en'
test_en = load_data(file)

# Load German test dataset
file = 'datasets/raw/newstest2014.de'
test_de = load_data(file)


# # filter out the data longer than the length 50
# f_test_en, f_test_de = len_filter(test_en, test_de, max_sen_len=50)
# ff_test_en, ff_test_de = dif_filter(f_test_en, f_test_de, dif=10)
# with open('datasets/preprocessed/ff_test.en.pkl', 'wb') as fw:
#     pickle.dump(ff_test_en, fw)
# with open('datasets/preprocessed/f_test.de.pkl', 'wb') as fw:
#     pickle.dump(ff_test_de, fw)
with open('datasets/preprocessed/ff_test.en.pkl', 'rb') as fr:
    ff_test_en = pickle.load(fr)
with open('datasets/preprocessed/ff_test.de.pkl', 'rb') as fr:
    ff_test_de = pickle.load(fr)


test_source_input, test_source_len = make_source(ff_test_en, en_to_id, max_sen_len=51, reverse=True, unk=True)
with open('datasets/preprocessed/test_source_reverse_unk.pkl', 'wb') as fw:
    pickle.dump((test_source_input, test_source_len), fw)
_, test_target_output = make_target(ff_test_de, de_to_id, max_sen_len=51, unk=True)
with open('datasets/preprocessed/test_label_unk.pkl', 'wb') as fw:
    pickle.dump(test_target_output, fw)
with open('datasets/preprocessed/test_source.pkl', 'rb') as fr:
    test_source_input, test_source_len = pickle.load(fr)
with open('datasets/preprocessed/test_label_unk.pkl', 'rb') as fr:
    test_target_output = pickle.load(fr)
