import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.getcwd())
from src.functions import *


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, lstm_dim, lstm_layer, dropout):
        """
        :param src_vocab_size: vocabulary size
        :param embed_dim: embedding layer weight size
        :param lstm_dim: lstm hidden dimension
        :param lstm_layer: # lstm layer
        :param dropout: dropout ratio
        """
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, lstm_dim, lstm_layer, batch_first=True, dropout=dropout)

    def forward(self, x, hidden):
        """
        :param x: (mini_batch, max_sen_len) = tg
        :param hidden: (lstm_layer, mini_batch, lstm_dim)
        :return: output=(mini_batch, max_sen_len, lstm_dim),
                 hidden=(lstm_layer, mini_batch, lstm_dim)
        """
        x = self.embedding(x)                       # x = (mini_batch, max_sen_len, embed_dim)
        x = self.dropout(x)
        output, hidden = self.lstm(x, hidden)       # output = (mini_batch, max_sen_len, lstm_dim)
        return output, hidden

    def init_param(self):
        # embedding layer
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        # lstm layer
        for layer in self.lstm.all_weights:
            for weight in layer:
                if weight.ndim == 2:
                    weight.data.uniform_(-0.1, 0.1)     # weight matrix
                else:
                    weight.data.fill_(0)                # bias
        return None


class Attention(nn.Module):
    def __init__(self, align, lstm_dim, max_sen_len, input_feed):
        super(Attention,  self).__init__()
        self.align = align
        self.input_feed = input_feed

        if align == 'general' or align == 'concat':
            self.att_score = nn.Linear(lstm_dim, lstm_dim, bias=False)
        elif align == 'concat':
            self.att_score = nn.Linear(lstm_dim*2, lstm_dim, bias=False)
            self.v = nn.Parameter(torch.FloatTensor(lstm_dim, 1))
        elif align == 'location':
            self.att_score = nn.Linear(lstm_dim, max_sen_len, bias=False)
        else:
            print("It's the wrong alignment method.")

    def forward(self, ht, hs):
        """
        :param ht: (mini_batch, seq_len, lstm_dim)
        :param hs: (mini_batch, max_sen_len(window), lstm_dim)
        :return: align_vec = (mini_batch, seq_len, max_seq_len(window))
        """
        mini_batch, max_sen_len, lstm_dim = hs.shape
        score = 0
        if self.align == 'dot':
            score = torch.bmm(ht, hs.transpose(2, 1))       # score = (mini_batch, seq_len, max_sen_len(window))
        elif self.align == 'general':
            score = self.att_score(hs)                      # score = (mini_batch, max_sen_len(window), lstm_dim)
            score = torch.bmm(ht, score.transpose(2, 1))    # score = (mini_batch, seq_len, max_sen_len(window))
        elif self.align == 'concat':
            ht = ht.tile(1, max_sen_len, 1)                 # ht = (mini_batch, max_sen_len(window), lstm_dim)
            cat = ht.cat(hs, dim=2)                         # cat = (mini_batch, max_sen_len(window), lstm_dim * 2)
            score = self.att_score(cat)                     # score = (mini_batch, max_sen_len(window), lstm_dim)
            score = torch.tanh(score)
            score = torch.matmul(score, self.v)             # score = (mini_batch, max_sen_len(window), seq_len)
        elif self.align == 'location':
            score = self.att_score(ht)                      # score = (mini_batch, seq_len, max_sen_len(window))
        if self.input_feed:
            score = score.view(mini_batch, 1, max_sen_len)  # score = (mini_batch, seq_len, max_sen_len)
        align_vec = F.softmax(score, dim=2)                 # align_vec = (mini_batch, seq_len, max_sen_len)
        return align_vec

    def init_param(self):
        self.att_score.weight.data.uniform_(-0.1, 0.1)
        if self.align == 'concat':
            self.v.data.uniform_(-0.1, 0.1)
        return None


class Decoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, lstm_dim, lstm_layer, dropout,
                 align, att_type, max_sen_len, input_feed, window_size, gpu, cuda):
        """
        :param src_vocab_size:
        :param embed_dim:
        :param lstm_dim:
        :param lstm_layer:
        :param dropout:
        :param align:
        :param att_type:
        :param max_sen_len:
        :param input_feed:
        :param window_size:
        :param gpu:
        :param cuda:
        """
        super(Decoder, self).__init__()
        self.align = align
        self.att_type = att_type
        self.input_feed = input_feed
        self.window_size = window_size
        self.gpu = gpu
        self.cuda = cuda

        self.embedding = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        if self.input_feed:
            self.lstm = nn.LSTM(embed_dim+lstm_dim, lstm_dim, lstm_layer, batch_first=True, dropout=dropout)
        else:
            self.lstm = nn.LSTM(embed_dim, lstm_dim, lstm_layer, batch_first=True, dropout=dropout)
        self.attention = Attention(align, lstm_dim, max_sen_len, input_feed)
        self.linear = nn.Linear(lstm_dim*2, lstm_dim, bias=False)

        if self.att_type == 'local':
            self.position = nn.Linear(lstm_dim, lstm_dim, bias=False)
            self.v_p = nn.Parameter(torch.FloatTensor(lstm_dim, 1))

    def forward(self, x, prev_hht, hidden, encoder_outputs, src_len):
        """
        'att_len' is the range of source hidden states within the window
        :param x: (mini_batch, seq_len)
        :param prev_hht: (mini_batch, seq_len, lstm_dim): the output vector of the past time step decoder
        :param hidden: (lstm_layer, mini_batch, lstm_dim)
        :param encoder_outputs: (mini_batch, max_sen_len, lstm_dim)
        :param src_len: (mini_batch, ): the source sentence length of encoder outputs
        :return: hht = (mini_batch, seq_len, tgt_vocab_size)
                 hidden = (h, c) = ((lstm_layer, mini_batch, lstm_dim), (lstm_layer, mini_batch, lstm_dim))
        """
        mini_batch, max_sen_len, lstm_dim = encoder_outputs.size()
        x = self.embedding(x)                                       # x = (mini_batch, seq_len, embed_dim)
        x = self.dropout(x)
        if self.input_feed:
            x = torch.cat((x, prev_hht), dim=2)                     # x = (mini_batch, seq_len, embed_dim+lstm_dim)
            ht, hidden = self.lstm(x, hidden)                       # ht = (mini_batch, seq_len, lstm_dim)
        else:
            ht, hidden = self.lstm(x, hidden)                       # ht = (mini_batch, seq_len, lstm_dim)

        if self.att_type == 'global':
            at = self.attention(ht, encoder_outputs)                # at = (mini_batch, seq_len, att_len)
            ct = torch.bmm(at, encoder_outputs)                     # ct = (mini_batch, seq_len, lstm_dim)
        else:  # att_type == 'local'
            p_sig = torch.sigmoid_(torch.matmul(torch.tanh(self.position(ht)), self.v_p)).squeeze()
            pt = src_len.unsqueeze(1) * p_sig.squeeze(2)             # pt = (mini_batch, seq_len) = p_sig.squeeze(2)
            hhs = make_position_vec(pt, encoder_outputs, src_len, self.window_size)
            if self.gpu:                                            # hhs = (mini_batch, max_sen_len(window), lstm_dim)
                hhs = hhs.to(torch.device(f'cuda:{self.cuda}'))
            hhs = softmax_masking(hhs)                              # hhs = (mini_batch, max_sen_len(window), lstm_dim)
            align = self.attention(ht, hhs)                         # align = (mini_batch, seq_len, max_sen_len(window)
            a_t = align * torch.exp(-pow((src_len.unsqeeze(1).tile(1, max_sen_len)-pt), 2) /
                                    (2*pow(self.window_size/2, 2))).unsqueeze(2).tile(1, 1, max_sen_len)
                                                                    # a_t = (mini_batch, seq_len, max_sen_len(window))
            ct = torch.bmm(a_t, hhs)                                # ct = (mini_batch, seq_len, lstm_dim)
        hht = torch.tanh(self.linear(torch.cat((ct, ht), dim=2)))   # hht = (mini_batch, seq_len, lstm_dim)
        return hht, hidden

    def init_param(self):
        # embedding layer
        self.embedding.weight.data.uniform_(-0.1, 0.1)

        # lstm layer
        for layer in self.lstm.all_weights:
            for weight in layer:
                if weight.ndim == 2:
                    weight.data.uniform_(-0.1, 0.1)  # weight matrix
                else:
                    weight.data.fill_(0)  # bias
        # attention
        self.attention.init_param()
        # linear
        self.linear.weight.data.uniform_(-0.1, 0.1)
        if self.att_type == 'local':
            self.position.weight.data.uniform_(-0.1, 0.1)
        return None
