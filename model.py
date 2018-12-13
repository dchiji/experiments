
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
        self.W1 = nn.Linear(emb_dim, int(emb_dim / 2))
        self.W2 = nn.Linear(int(emb_dim / 2), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, question):
        # question: (batch_size, length, emb_dim)-tensor
        ones = torch.ones([question.size()[0], question.size()[1], self.emb_dim]).float().to(self.device)

        h = self.emb(question)
        h = self.cossim(h, ones)
        h = self.W1(h)
        h = self.relu(h)
        h = self.W2(h)
        out = self.sigmoid(h)
        return out

    def set_optimizer(self, opt):
        self.opt = opt
