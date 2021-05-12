import argparse
from distutils.util import strtobool as _bool
import json
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import sys
import os
sys.path.append(os.getcwd())
from src.models import *
from src.functions import *


############################ Argparse ############################
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--port', type=int, default=5678)
parser.add_argument('--att_type', type=str, default='local_m',
                    help='global  |  local_p  |  local_m')
parser.add_argument('--align', type=str, default='general',
                    help='location  |  dot  |  general  |  concat')
parser.add_argument('--input_feed', type=_bool, default=False)
parser.add_argument('--reverse', type=_bool, default=True)
parser.add_argument('--unk', type=_bool, default=False)
parser.add_argument('--mini_batch', type=int, default=64)
parser.add_argument('--max_epoch', type=int, default=12)
parser.add_argument('--fine_tune_epoch', type=int, default=8)
parser.add_argument('--eval_interval', type=int, default=50)
parser.add_argument('--trunc', type=int, default=10)
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

if not os.path.exists(log_dir):
    os.mkdir(log_dir)
with open(os.path.join(log_dir, 'args.json'), 'w') as f:
    json.dump(args.__dict__, f)


############################ Tensorboard ############################
tb_dir = os.path.join(log_dir, 'tb')
ckpt_dir = os.path.join(log_dir, 'ckpt')
if not os.path.exists(tb_dir):
    os.mkdir(tb_dir)
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)
tb_writer = SummaryWriter(tb_dir)


############################ Hyperparameter ############################
max_sen_len = 51        # add EOS or SOS token (50 + 1)
embed_dim = 1000
lstm_layer = 4
lstm_dim = 1000
dropout = 0.2
window_size = 10
lr = 1.0
max_norm = 5.0


############################ Load preprocessed data ############################
print('\n')
print("Load the preprocessed data..")
with open('datasets/preprocessed/vocab.en.pkl', 'rb') as fr:
    en_to_id, id_to_en = pickle.load(fr)
with open('datasets/preprocessed/vocab.de.pkl', 'rb') as fr:
    de_to_id, id_to_de = pickle.load(fr)
src_vocab_size = len(en_to_id)
tgt_vocab_size = len(de_to_id)

src_input = None
tgt_input = None
tgt_output = None
if args.unk:
    if args.reverse:
        with open('datasets/preprocessed/train/source_reverse_unk.pkl', 'rb') as fr:
            src_input, src_len = pickle.load(fr)
    else:
        with open('datasets/preprocessed/train/source_unk.pkl', 'rb') as fr:
            src_input, src_len = pickle.load(fr)
    with open('datasets/preprocessed/train/target_unk.pkl', 'rb') as fr:
        tgt_input, tgt_output = pickle.load(fr)
else:
    if args.reverse:
        with open('datasets/preprocessed/train/source_reverse.pkl', 'rb') as fr:
            src_input, src_len = pickle.load(fr)
    else:
        with open('datasets/preprocessed/train/source.pkl', 'rb') as fr:
            src_input, src_len = pickle.load(fr)
    with open('datasets/preprocessed/train/target.pkl', 'rb') as fr:
        tgt_input, tgt_output = pickle.load(fr)
print("Complete.")

print("Split the data into mini_batch..")
src_input = make_batch(src_input, args.mini_batch)
src_len = make_batch(src_len, args.mini_batch)
tgt_input = make_batch(tgt_input, args.mini_batch)
tgt_output = make_batch(tgt_output, args.mini_batch)
print("Complete.")


############################ InitNet ############################
model = RnnNMT(src_vocab_size, tgt_vocab_size, embed_dim, lstm_dim, lstm_layer, dropout,
               args.align, args.att_type, max_sen_len, args.input_feed, window_size, args.gpu, args.cuda)
model.init_param()

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=lr)

'''
# If you want to continue the training...
model.load_state_dict(torch.load(os.path.join(log_dir, 'ckpt/model.ckpt'), map_location='cuda:0'))
optimizer.load_state_dict(torch.load(os.path.join(log_dir, 'ckpt/optimizer.ckpt'), map_location='cuda:0'))
'''

device = None
if args.gpu:
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    model.to(device)

np.random.seed(args.random_seed)
torch.random.manual_seed(args.random_seed)

src_input = torch.from_numpy(src_input)
src_len = torch.from_numpy(src_len)
tgt_input = torch.from_numpy(tgt_input)
tgt_output = torch.from_numpy(tgt_output)


############################ Training start ############################
sen_num = 0             # count the number of real train sentence pair.
total_loss = 0
count = 0
for epoch in range(args.max_epoch):
    # shuffle the training data
    print("Shuffling the data..")
    per = torch.randperm(len(src_input))
    src_input = src_input[per]
    src_len = src_len[per]
    tgt_input = tgt_input[per]
    tgt_output = tgt_output[per]
    print("Complete.")

    src_input = src_input.to(torch.int64)
    src_len = src_len.to(torch.int64)
    tgt_input = tgt_input.to(torch.int64)
    tgt_output = tgt_output.to(torch.int64)

    # schedule the learning rate
    if (epoch+1) > args.fine_tune_epoch:
        lr /= 2
    for batch_src_input, batch_src_len, batch_tgt_input, batch_tgt_output \
            in tqdm(zip(src_input, src_len, tgt_input, tgt_output), total=len(src_input),
                    desc=f'epoch {epoch+1}/{args.max_epoch}', bar_format='{l_bar}{bar:30}{r_bar}'):
        '''if (epoch+1) < 12:
            sen_num += len(src_input)
            break'''

        model.train()
        sen_num += 1

        # init hidden state
        h_0 = torch.zeros(lstm_layer, args.mini_batch, lstm_dim)  # (4, 128, 1000)
        c_0 = torch.zeros(lstm_layer, args.mini_batch, lstm_dim)
        hidden = (h_0, c_0)
        hidden = [state.detach() for state in hidden]
        if args.gpu:
            batch_src_input = batch_src_input.to(device)
            batch_src_len = batch_src_len.to(device)
            batch_tgt_input = batch_tgt_input.to(device)
            batch_tgt_output = batch_tgt_output.to(device)
            hidden = [state.to(device) for state in hidden]

        # first decoder (past) output
        hht = torch.zeros(args.mini_batch, max_sen_len, lstm_dim)         # first time-step prev decoder context
        if args.gpu:
            hht = hht.to(device)

        # train
        optimizer.zero_grad()
        out = model(batch_src_input, batch_tgt_input, hidden, hht, batch_src_len)
        if args.gpu:
            out = out.to(device)
        loss = criterion(out.view(-1, tgt_vocab_size), batch_tgt_output.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm, norm_type=2)
        optimizer.step()

        total_loss += loss.data
        count += 1

        if (args.eval_interval is not None) and ((sen_num % args.eval_interval) == 1):
            avg_loss = total_loss/count
            tb_writer.add_scalar(f'loss/(sentences * mini_batch{args.mini_batch})', avg_loss, sen_num)
            tb_writer.add_scalar('lr/epoch', lr, epoch+1)

            total_loss = 0
            count = 0
        tb_writer.flush()
    '''if (epoch + 1) < 12:
        continue'''
    print("Save the model..")
    torch.save(model.state_dict(), os.path.join(log_dir, 'ckpt/model.ckpt'))
    torch.save(optimizer.state_dict(), os.path.join(log_dir, 'ckpt/optimizer.ckpt'))
    print("Complete..!")
    print('\n')

    # evaluate test set and save the text file.
    test_eval(model, log_dir, args.mini_batch, lstm_layer, lstm_dim, max_sen_len,
              args.gpu, args.cuda, args.reverse, args.unk, args.trunc, epoch, id_to_de)
