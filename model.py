
import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, emb_dim, init_weight, device):
        #
        # init_weight: (vocab_size, emb_dim)-tensor
        #
        super(Classifier, self).__init__()

        self.emb_dim = emb_dim
        self.vocab_size = init_weight.size()[0]
        self.padding_idx = self.vocab_size - 1
        self.device = device

        self.emb = nn.Embedding(self.vocab_size, self.emb_dim)
        self.emb.weight = nn.Parameter(init_weight)

        self.cossim = nn.CosineSimilarity(dim=1)
        self.relu = nn.ReLU()
        self.W1 = nn.Linear(emb_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, question):
        # question: (batch_size, length, emb_dim)-tensor
        ones = torch.ones([question.size()[0], question.size()[1], self.emb_dim]).float().to(self.device)

        h = self.emb(question)
        h = self.cossim(h, ones)
        h = self.relu(h)
        h = self.W1(h)
        out = self.sigmoid(h)
        return out

    def set_optimizer(self, opt):
        self.opt = opt

class GRUBase(nn.Module):
    def __init__(self, emb_dim, init_weight, device):
        #
        # init_weight: (vocab_size, emb_dim)-tensor
        #
        super(GRUBase, self).__init__()

        self.emb_dim = emb_dim
        self.vocab_size = init_weight.size()[0]
        self.padding_idx = self.vocab_size - 1
        self.device = device

        self.emb = nn.Embedding(self.vocab_size, self.emb_dim)
        self.emb.weight = nn.Parameter(init_weight)

        self.hidden_dim = 90

        self.h_0 = torch.rand(4, 1, self.hidden_dim, requires_grad=True).to(self.device)
        self.dropout = nn.Dropout(p=0.2)
        self.gru = nn.GRU(emb_dim, self.hidden_dim, 2, batch_first=True, bidirectional=True)
        self.M = nn.Linear(self.hidden_dim * 2, 1)

        self.W1 = nn.Linear(self.hidden_dim * 2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, question):
        # question: (batch_size, seq_len, emb_dim)-tensor
        batch_size = question.size()[0]
        seq_len = question.size()[1]

        input = self.emb(question)
        input = self.dropout(input)

        h_0 = torch.cat([self.h_0] * batch_size, dim=1) # (4, batch_size, hidden_dim)
        out, _ = self.gru(input, h_0)   # (batch_size, seq_len, 2 * hidden_dim)
        out = out.contiguous().view([batch_size, seq_len, 2 * self.hidden_dim])
        weight = self.M(out)    # (batch_size * seq_len, 1)
        weight = weight.view([batch_size, seq_len])
        weight = self.softmax(weight)
        out = torch.einsum('bij,bi->bj', out, weight)
        out = self.relu(out)
        out = self.W1(out)
        out = self.sigmoid(out)
        return out

    def set_optimizer(self, opt):
        self.opt = opt
