import torch.nn as nn
import torch

class EntityEmbedding(nn.Module):
    def __init__(self, bins_center, embedding_dim, eps=1e-8):
        super(EntityEmbedding, self).__init__()
        self.embedding = nn.Embedding(len(bins_center), embedding_dim)
        self.softmax = nn.Softmax(dim=1)
        self.bins_center = torch.FloatTensor(bins_center).detach_()
        self.eps = eps
        
    def forward(self, x):
        x = x.unsqueeze(1)
        w = self.softmax((1 / ((x - self.bins_center).abs() + self.eps)))
        v = torch.mm(w, self.embedding.weight)
        return v
        
        
# -------------------------        
bins_center = [1,2,3,4,5] # class List
emb_dim = 3 
x = torch.FloatTensor([1,2,3,4,5]) # dim=(n,)

entityembedding = EntityEmbedding(bins_center, emb_dim)
emb_vector = entityembedding(x)

# result the same as entityembedding.embedding.weigth
