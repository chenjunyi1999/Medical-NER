import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel

from config import EMBEDDING_DIM, HIDDEN_DIM


class Bert_BiLSTM_CRF(nn.Module):

    def __init__(self, tag2idx, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM):
        super(Bert_BiLSTM_CRF, self).__init__()
        self.tag_to_ix = tag2idx
        self.tagset_size = len(tag2idx)
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim // 2,
                            num_layers=2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(hidden_dim, self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=True)

    def getfeature(self, sentence):
        with torch.no_grad():
            # BERT默认返回两个 last_hidden_state, pooler_output
            # last_hidden_state：输出序列每个位置的语义向量，形状为：(batch_size, sequence_length, hidden_size)
            # pooler_output：[CLS]符号对应的语义向量，经过了全连接层和tanh激活；该向量可用于下游分类任务
            embeds, _ = self.bert(sentence, return_dict=False)
        # LSTM默认返回两个 output, (h,c)
        # output:[batch_size,seq_len,hidden_dim * 2]   if birectional
        # h,c :[num_layers * 2,batch_size,hidden_dim]  if birectional
        # h 为LSTM最后一个时间步的隐层结果，c 为LSTM最后一个时间步的Cell状态
        out, _ = self.lstm(embeds)
        out = self.dropout(out)
        feats = self.linear(out)
        return feats

    def forward(self, sentence, tags, mask, is_test=False):
        feature = self.getfeature(sentence)
        # training
        if not is_test:
            # return log-likelihood
            # make this value negative as our loss
            loss = -self.crf.forward(feature, tags, mask, reduction='mean')
            return loss
        # testing
        else:
            decode = self.crf.decode(feature, mask)
            return decode
