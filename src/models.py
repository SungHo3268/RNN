import sys
import os
sys.path.append(os.getcwd())
from src.layers import *


class RnnNMT(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim, lstm_dim, lstm_layer, dropout,
                 align, att_type, max_sen_len, input_feed, window_size, gpu, cuda):
        """"""
        super(RnnNMT, self).__init__()
        self.tgt_vocab_size = tgt_vocab_size
        self.input_feed = input_feed
        self.lstm_dim = lstm_dim

        self.encoder = Encoder(src_vocab_size, embed_dim, lstm_dim, lstm_layer, dropout)
        self.decoder = Decoder(src_vocab_size, embed_dim, lstm_dim, lstm_layer, dropout,
                               align, att_type, max_sen_len, input_feed, window_size, gpu, cuda)
        self.linear = nn.Linear(lstm_dim, tgt_vocab_size, bias=False)

    def forward(self, src, tgt, hidden, hht, src_len):
        """
        :param src: (mini_batch, max_sen_len)
        :param tgt: (mini_batch, seq_len)
        :param hidden: (lstm_layer, mini_batch, lstm_dim)
        :param hht: (mini_batch, seq_len, lstm_dim)
        :param src_len: (mini_batch, )
        :return: (mini_batch, seq_len, tgt_vocab_size)
        """
        mini_batch, max_sen_len = tgt.size()
        encoder_outputs, hidden = self.encoder(src, hidden)     # encoder_outputs=(mini_batch, max_sen_len, lstm_dim)
        output = torch.zeros(mini_batch, max_sen_len, self.tgt_vocab_size)
        if self.input_feed:
            for i in range(max_sen_len):
                hht, hidden = self.decoder(tgt[:, i].unsqueeze(1), hht, hidden, encoder_outputs, src_len)
                out = self.linear(hht)          # out = (mini_batch, 1, tgt_vocab_size), hht = (mini_batch, 1, lstm_dim)
                output[:, i] = out.squeeze()
        else:
            hht, hidden = self.decoder(tgt, hht, hidden, encoder_outputs, src_len)
            output = self.linear(hht)           # output = (mini_batch, seq_len, tgt_vocab_size)
        return output

    def init_param(self):
        self.encoder.init_param()
        self.decoder.init_param()
        self.linear.weight.data.uniform_(-0.1, 0.1)
