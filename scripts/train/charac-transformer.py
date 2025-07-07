import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import os
from torch.utils.data import Dataset, DataLoader
import wandb

# hyperparameters
VOCAB_SIZE = 50 
EMBED_SIZE = 128
NUM_HEADS = 4
NUM_LAYERS = 2
FFN_DIM = 512
BATCH_SIZE = 8
SEQ_LEN = 50
LEARNING_RATE = 0.1
EPOCHS = 10
SAVE_INTERVAL = 200 # Save parameter changes every 200 steps
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CharDataset(Dataset):
    def __init__(self, text, seq_len):
        self.seq_len = seq_len
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.data = [self.char_to_idx[ch] for ch in text]
        
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        input_seq = self.data[idx:idx + self.seq_len]
        target_seq = self.data[idx + 1:idx + self.seq_len + 1]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, ffn_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size, 
            nhead=num_heads, 
            dim_feedforward=ffn_dim, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)
    
    def forward(self, x):
        x = self.embed(x) * math.sqrt(EMBED_SIZE)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.fc(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_param_changes(model, prev_params):
    changes = []
    for name, param in model.named_parameters():
        if prev_params[name] is not None:
            change = (param.data.detach() - prev_params[name].detach()).abs().flatten()
            changes.append(change.cpu().numpy())
        prev_params[name] = param.data.clone().detach()
    return np.concatenate(changes), prev_params

def save_param_changes_npy(change_buffer, file_idx, output_dir="param_changes"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(os.path.join(output_dir, f"param_changes_file_{file_idx}.npy"), np.array(change_buffer))
    print(np.array(change_buffer).shape)

def train(model, dataloader, optimizer, criterion, epochs, save_interval, output_dir="param_changes"):
    model.train()
    prev_params = {name: p.data.clone() if p.requires_grad else None 
                   for name, p in model.named_parameters()}
    change_buffer = []
    file_idx = 0
    global_step = 0
    
    for epoch in range(epochs):
        total_loss = 0
        for step, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, VOCAB_SIZE), targets.view(-1))
            loss.backward()
            optimizer.step()

            wandb.log({"loss": loss.item()})
            
            # compute and save parameter changes
            changes, prev_params = compute_param_changes(model, prev_params)
            change_buffer.append(changes)
            
            if len(change_buffer) >= save_interval:
                save_param_changes_npy(change_buffer, file_idx, output_dir)
                print(f"Saved parameter changes to param_changes_file_{file_idx}.npy")
                change_buffer = []
                file_idx += 1
            
            global_step += 1
            total_loss += loss.item()
            print(f"Epoch {epoch+1}, Step {step+1}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
    
    if change_buffer:
        save_param_changes_npy(change_buffer, file_idx, output_dir)
        print(f"Saved remaining parameter changes to param_changes_file_{file_idx}.npy")

if __name__ == "__main__":

    text = "thequickbrownfoxjumpsoverthelazydog" * 100
    dataset = CharDataset(text, SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = SimpleTransformer(VOCAB_SIZE, EMBED_SIZE, NUM_HEADS, NUM_LAYERS, FFN_DIM).to(DEVICE)
    print(f"Total parameters: {count_parameters(model)}")
    

    wandb.init(project="Criticality-LLM", group="LLM",
                    config={"dataset": 'text',
                            "details": text, "num_epochs": EPOCHS, 
                            "optimizer": 'SGD',
                            "learning_rate": LEARNING_RATE, "device": DEVICE,
                            "batch_size": BATCH_SIZE})
    
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    train(model, dataloader, optimizer, criterion, EPOCHS, SAVE_INTERVAL,output_dir=f'/Volumes/Elements S1/LLM/lr{LEARNING_RATE}/param_changes_{BATCH_SIZE}/')
    wandb.finish()