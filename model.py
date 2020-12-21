
class CNN(nn.Module):
    def __init__(self, vocab_size, vector_size, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()     
        self.embedding = nn.Embedding(vocab_size, vector_size, padding_idx = pad_idx)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels = 1, out_channels = n_filters, kernel_size = (fs, vector_size)) for fs in filter_sizes])
        self.linear = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        
        
    def forward(self, text):
        embedded = self.embedding(text).unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]   
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        return self.linear(cat)