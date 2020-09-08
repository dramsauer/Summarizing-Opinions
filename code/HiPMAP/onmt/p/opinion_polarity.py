import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

MODEL_OUTPUT_FILENAME = '../../../../output/opinion_polarity/BERTGRUSentiment-model-MPQA_v2_100.pt'

bert = BertModel.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id
max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']


class OpinionPolarityPredictor(object):
    def __init__(self):
        self.model = BERTGRUSentiment(
            bert=BertModel.from_pretrained('bert-base-uncased')
        )
        self.model = self.model.to(device)
        self.model.load_state_dict(torch.load(MODEL_OUTPUT_FILENAME))

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def predict_sentiment(self, sentence):
        self.model.eval()
        tokens = self.tokenizer.tokenize(sentence)
        tokens = tokens[:max_input_length - 2]
        indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(0)
        prediction = torch.tanh(self.model(tensor))
        return prediction.item()


class BERTGRUSentiment(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim=HIDDEN_DIM,
                 output_dim=OUTPUT_DIM,
                 n_layers=N_LAYERS,
                 bidirectional=BIDIRECTIONAL,
                 dropout=DROPOUT):

        super().__init__()

        self.bert = bert

        embedding_dim = bert.config.to_dict()['hidden_size']

        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)

        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, sent len]
        with torch.no_grad():
            embedded = self.bert(text)[0]
        # embedded = [batch size, sent len, emb dim]

        _, hidden = self.rnn(embedded)
        # hidden = [n layers * n directions, batch size, emb dim]

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])
        # hidden = [batch size, hid dim]

        output = self.out(hidden)
        # output = [batch size, out dim]
        return output