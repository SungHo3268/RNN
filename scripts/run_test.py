import argparse
from distutils.util import strtobool as _bool
import sys
import os


sys.path.append(os.getcwd())
from src.models import *
from src.functions import *


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--port', type=int, default=5678)
parser.add_argument('--att_type', type=str, default='global',
                    help='global  |  local_p  |  local_m')
parser.add_argument('--align', type=str, default='location',
                    help='location  |  dot  |  general  |  concat')
parser.add_argument('--input_feed', type=_bool, default=False)
parser.add_argument('--reverse', type=_bool, default=True)
parser.add_argument('--unk', type=str, default=True)
parser.add_argument('--mini_batch', type=int, default=128)
parser.add_argument('--max_epoch', type=int, default=12)
parser.add_argument('--fine_tune_epoch', type=int, default=8)
parser.add_argument('--eval_interval', type=int, default=50)
parser.add_argument('--random_seed', type=int, default=515)
parser.add_argument('--gpu', type=_bool, default=True)
parser.add_argument('--cuda', type=int, default=0)
args = parser.parse_args()


if args.reverse:
    if args.unk:
        log_dir = f'log/{args.att_type}_{args.align}_reverse_unk'
    else:
        log_dir = f'log/{args.att_type}_{args.align}_reverse'
else:
    if args.unk:
        log_dir = f'log/{args.att_type}_{args.align}_unk'
    else:
        log_dir = f'log/{args.att_type}_{args.align}'


############################ Hyperparameter ############################
max_sen_len = 51        # add EOS or SOS token (50 + 1)
embed_dim = 1000
lstm_layer = 4
lstm_dim = 1000
dropout = 0.2
window_size = 10
lr = 1.0
max_norm = 5.0

with open('datasets/preprocessed/vocab.en.pkl', 'rb') as fr:
    en_to_id, id_to_en = pickle.load(fr)
with open('datasets/preprocessed/vocab.de.pkl', 'rb') as fr:
    de_to_id, id_to_de = pickle.load(fr)
src_vocab_size = len(en_to_id)
tgt_vocab_size = len(de_to_id)


############################ Load test data ############################
print("Load the preprocessed test data..")
with open('datasets/preprocessed/test_source_reverse_unk.pkl', 'rb') as fr:
    test_source_input, test_source_len = pickle.load(fr)
with open('datasets/preprocessed/test_label_unk.pkl', 'rb') as fr:
    test_target_output = pickle.load(fr)

print("Split the data into mini_batch..")
test_src_input = make_batch(test_source_input, args.mini_batch)
test_src_len = make_batch(test_source_len, args.mini_batch)
test_tgt_output = make_batch(test_target_output, args.mini_batch)
print("Complete.\n")

test_src_input = torch.from_numpy(test_src_input)
test_src_len = torch.from_numpy(test_src_len)
test_tgt_output = torch.from_numpy(test_tgt_output)

test_src_input = test_src_input.to(torch.int64)
test_src_len = test_src_len.to(torch.int64)
test_tgt_output = test_tgt_output.to(torch.int64)


############################ InitNet ############################
model = RnnNMT(src_vocab_size, tgt_vocab_size, embed_dim, lstm_dim, lstm_layer, dropout,
               args.align, args.att_type, max_sen_len, args.input_feed, window_size, args.gpu, args.cuda)
model.load_state_dict(torch.load(os.path.join(log_dir, 'ckpt/model.ckpt'), map_location='cuda:0'))
model.eval()
device = None
if args.gpu:
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    model.to(device)

np.random.seed(args.random_seed)
torch.random.manual_seed(args.random_seed)


############################ Test ############################
cur = 0
output = torch.zeros_like(test_src_input)           # output = (44, 64, 51)
for batch_src_input, batch_src_len in tqdm(zip(test_src_input, test_src_len), total=len(test_src_input),
                                           bar_format='{l_bar}{bar:30}{r_bar}'):
    # init hidden state
    h_0 = torch.zeros(lstm_layer, args.mini_batch, lstm_dim)  # (4, 128, 1000)
    c_0 = torch.zeros(lstm_layer, args.mini_batch, lstm_dim)
    hidden = (h_0, c_0)
    # hidden = [state.detach() for state in hidden]

    tgt = torch.ones(args.mini_batch, 1)           # tgt = (mini_batch, 1)  ==> SOS tokens
    tgt = tgt.to(torch.int64)
    if args.gpu:
        batch_src_input = batch_src_input.to(device)
        batch_src_len = batch_src_len.to(device)
        tgt = tgt.to(device)
        hidden = [state.to(device) for state in hidden]

    # first decoder (past) output
    hht = torch.zeros(args.mini_batch, 1, lstm_dim)  # first time-step prev decoder context
    if args.gpu:
        hht = hht.to(device)
    for i in range(max_sen_len):
        out = model(batch_src_input, tgt, hidden, hht, batch_src_len)
        if args.gpu:
            out = out.to(device)                        # out = (mini_batch, seq_len, tgt_vocab)
        pred = torch.max(out, dim=-1)[1]      # pred = (mini_batch, seq_len)
        tgt = torch.cat((tgt, pred[:, i].unsqueeze(1)), dim=1)
    output[cur] = tgt[:, 1:]            # output[cur] = (mini_batch, seq_len)
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
with open(os.path.join(test_dir, 'output.txt'), 'w', encoding='utf8') as fw:
    fw.writelines(test_pred_output)
with open(os.path.join(test_dir, 'label.txt'), 'w', encoding='utf8') as fw:
    fw.writelines(test_label)
