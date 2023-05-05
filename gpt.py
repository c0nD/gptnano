import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyper-parameters
BATCH_SIZE = 32  # the amt of independent sequences to be processed in parallel
BLOCK_SIZE = 8  # maximum context len for predictions
MAX_ITER = 4000
EVAL_INTERVAL = 250
LEARNING_RATE = 1e-2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EVAL_ITERS = 200

# -------------------


torch.manual_seed(420_69_1337)

with open('data.txt', 'r') as f:
    text = f.read()
    
# Create vocab
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create char to int and int to char maps
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoding a string to a list of ints
decode = lambda x: ''.join([itos[i] for i in x])  # decoding a list of ints to a string


# Training & test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.8)
train_data, val_data = data[:n], data[n:]

# -------------------


# Loading Data
def get_batch(split):
    # Gen small batch data of x and y
    data = train_data if split == 'train' else val_data
    xs = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in xs])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in xs])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for i in range(EVAL_ITERS):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# -------------------
    
# Bigram model
class Bigram(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        # each tkn reads the logits for the next tkn from a lookup tbl
        self.token_emb = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets=None):
        # idx & targets are both (B,T) tensors of ints
        logits = self.token_emb(idx) # (B,T,V)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    
    def generate(self, idx, max_new_tkns):
        for i in range(max_new_tkns):
            logits, loss, = self(idx)  # gets the predictions
            logits = logits[:, -1, :]  # (B,C)
            probs = F.softmax(logits, dim=-1)  # (B,C)
            next_idx = torch.multinomial(probs, num_samples=1)  # (B,1)
            idx = torch.cat([idx, next_idx], dim=-1)  # (B,T+1)
        return idx
    
model = Bigram(vocab_size)
m = model.to(DEVICE)


# pytorch optimization
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for i_ in range(MAX_ITER):
    
    # evaluate loss on train and val sets
    if i_ % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f'Iteration {i_} | Train Loss: {losses["train"]} | Val Loss: {losses["val"]}')
        
    # get batch
    x, y = get_batch('train')
    
    # eval loss
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
# -------------------

# Generate from the model
ctx = torch.zeros((1,1), dtype=torch.long, device=DEVICE)
print(decode(m.generate(ctx, max_new_tkns=500)[0].tolist()))