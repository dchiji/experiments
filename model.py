
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

        self.hidden_dim = int(emb_dim / 2)

        self.gru = nn.GRU(emb_dim, self.hidden_dim, 1, batch_first=True, bidirectional=True)
        self.W1 = nn.Linear(self.hidden_dim * 2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, question):
        # question: (batch_size, length, emb_dim)-tensor
        batch_size = question.size()[0]

        h_0 = torch.zeros([2, batch_size, self.hidden_dim])
        input = self.emb(question)
        _, h_final = self.gru(input, h_0)   # (2, batch_size, hidden_dim)
        h_final = h_final.transpose(0, 1)
        out = h_final.contiguous().view([batch_size, -1])

        out = self.relu(out)
        out = self.W1(out)
        out = self.sigmoid(out)
        return out

    def set_optimizer(self, opt):
        self.opt = opt
