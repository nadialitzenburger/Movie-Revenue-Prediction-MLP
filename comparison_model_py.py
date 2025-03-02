#!/usr/bin/env python
# coding: utf-8

# In[119]:


import csv
import os
from torch.utils.data import Dataset
import torch
import numpy as np

UNKNOWN = "UNK"
PADDING = "PAD"

class MovieDataset(Dataset):
    def __init__(self, filename, split_indices, lower=True, max_len=None, vocab=None):
        super().__init__()

        texts, labels, numeric_data = self.read_csv(filename, split_indices)
        
        if vocab is None:
            self.vocab = self.get_vocab(texts, lower)
        else:
            self.vocab = vocab

        self.max_len = max_len

        self.data_tensors, self.lengths_tensors = self.convert_text_to_tensors(texts, lower)
        self.numeric_tensors = self.convert_numeric_to_tensors(numeric_data)    
        self.labels_tensors = self.convert_labels_to_tensors(labels)

    def __len__(self):
        return len(self.data_tensors)

    def __getitem__(self, idx):
        data = {'x': self.data_tensors[idx],
                'lengths': self.lengths_tensors[idx],
                'numeric': self.numeric_tensors[idx],
                'y': self.labels_tensors[idx],
               }
        return data

    def get_vocab(self, texts, lower):
        vocab = {PADDING: 0, UNKNOWN: 1}
        
        for t in texts:
            if lower:
                t = t.lower()
            words = t.split()
        
            for word in words:
                if word not in vocab:
                    vocab[word] = len(vocab)
        
        return vocab

    def pad(self, idx_list):
        if self.max_len is None:
            self.max_len = 0
            for instance in idx_list:
                if len(instance) > self.max_len:
                    self.max_len = len(instance)
        padded_list = []
        original_lengths = []
        
        for seq in idx_list:
            original_lengths.append(len(seq))
            if len(seq) > self.max_len:
                seq = seq[:self.max_len]
            else:
                seq = seq + [self.vocab[PADDING]] * (self.max_len - len(seq))
            padded_list.append(seq)

        return padded_list, original_lengths

    def read_csv(self, filename, split_indices, lower=True):
        texts = []
        numeric_features = []
        labels = []
    
        # First pass to collect valid values for median calculation
        valid_months = []
        valid_runtimes = []
    
        with open(filename, encoding="utf-8") as csvfile:
            csvreader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
            for row in csvreader:
                if row['id'] not in split_indices:
                    continue
                
                # Collect valid values
                try:
                    if row['release_date']:
                        month = int(row['release_date'].split("-")[1])
                        valid_months.append(month)
                    if row['runtime']:
                        runtime = int(row['runtime'])
                        valid_runtimes.append(runtime)
                except ValueError:
                    continue
    
        # Calculate medians
        median_month = int(np.median(valid_months))
        median_runtime = int(np.median(valid_runtimes))
    
        # Second pass with a new file handle
        with open(filename, encoding="utf-8") as csvfile:
            csvreader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
            for row in csvreader:
                if row['id'] not in split_indices:
                    continue
            
                text = row['title']
                if lower and text is not None:
                    text = text.lower()
                texts.append(text)

                try:
                    release_date = row['release_date']
                    release_month = int(release_date.split("-")[1]) if release_date else median_month
                    runtime = int(row['runtime']) if row['runtime'] else median_runtime
                except ValueError:
                    release_month = median_month
                    runtime = median_runtime

                numeric_features.append([release_month, runtime])
                labels.append(float(row['revenue']))

        numeric_features = np.array(numeric_features)
    
        # Z-Score normalization
        means = numeric_features.mean(axis=0)
        stds = numeric_features.std(axis=0)
        stds[stds == 0] = 1  # Prevent division by zero
        numeric_data = (numeric_features - means) / stds

        return texts, labels, numeric_data

    def convert_text_to_tensors(self, text, lower):
        python_list = []
        for text_instance in text:
            if lower:
                text_instance = text_instance.lower()
            idx_instance = []
            for word in text_instance.split():
                if word not in self.vocab:
                    word = UNKNOWN
                idx = self.vocab[word]
                idx_instance.append(idx)
            python_list.append(idx_instance)
        
        python_list_padded, instance_lengths = self.pad(python_list)
        vectors_numpy = np.array(python_list_padded)
        tensors = torch.from_numpy(vectors_numpy)
        lengths_tensors = torch.from_numpy(np.array(instance_lengths))
        return tensors, lengths_tensors
        
    def convert_labels_to_tensors(self, labels):
        label_tensors = torch.from_numpy(np.array(labels))
        return label_tensors

    def convert_numeric_to_tensors(self, numeric_data):
        numeric_tensors = torch.from_numpy(np.array(numeric_data))
        return numeric_tensors


# In[120]:


import random
import csv as csv

random.seed(111)

def get_indices_split(filename, split=[0.8, 0.1, 0.1]):
    ids = []
    with open(filename, encoding = "utf-8") as csvfile:
        csvreader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for row in csvreader:
            ids.append(row['id'])
            # TODO: remove before doing the actual experiments!
            if len(ids) > 5000:
                break
    random.shuffle(ids)
    start_idx = 0
    splitted_ids = []
    for part in split:
        part_length = int(part * len(ids))
        splitted_ids.append(ids[start_idx:start_idx+part_length])
        start_idx += part_length
    return splitted_ids

filename = "/Users/nadialitzenburger/Downloads/TMDB_movie_dataset_v11_cleaned.csv"

train_indices, dev_indices, test_indices = get_indices_split(filename)

train_set = MovieDataset(filename=filename,
                          split_indices=train_indices
                          )
dev_set = MovieDataset(filename=filename,
                        split_indices=dev_indices,
                        max_len=train_set.max_len,
                        vocab=train_set.vocab,
                       )
test_set = MovieDataset(filename=filename,
                         split_indices=test_indices,
                         max_len=train_set.max_len,
                         vocab=train_set.vocab,
                        )


# In[121]:


from torch.utils.data import DataLoader

batch_size = 32
torch.manual_seed(111)

def collate_fn(batch):
    """
    Stellt sicher, dass alle Daten korrekt zu Tensoren zusammengefasst werden.
    """
    x = torch.stack([item['x'] for item in batch])
    lengths = torch.stack([item['lengths'] for item in batch])
    numeric = torch.stack([item['numeric'] for item in batch])  # Numerische Features als Tensor
    y = torch.stack([item['y'] for item in batch])

    return {'x': x, 'lengths': lengths, 'numeric': numeric, 'y': y}

# DataLoader mit collate_fn für korrektes Batch-Handling
train_dataloader = DataLoader(train_set, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
dev_dataloader = DataLoader(dev_set, batch_size=batch_size, collate_fn=collate_fn)
test_dataloader = DataLoader(test_set, batch_size=batch_size, collate_fn=collate_fn)


# In[122]:


for batch in test_dataloader:
    X = batch['x']  # Texte (Filmtitel als Indizes)
    numeric_features = batch['numeric']  # Numerische Features 
    y = batch['y']  # Umsatz (Label)

    print(f"Shape of X: {X.shape}")  # Größe der Titel-Daten
    print(f"Shape of numeric_features: {numeric_features.shape}")  # Größe der numerischen Features
    print(f"Shape of y: {y.shape} {y.dtype}")  # Größe der Labels (Zielwerte)

    print("Beispiel für X:", X)
    print("Beispiel für numeric_features:", numeric_features)
    
    break  # Nur einen Batch ausgeben


# In[123]:


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    
print(f"Using {device} device")


# In[145]:


from torch import nn

config = {
    'num_classes': 1,  
    'embedding_dim': 128,  
    'hidden_dim1': 322,
    'hidden_dim2': 290,
    'hidden_dim3': 132,  
    'hidden_dim_numeric': 128,
    'vocab_size': len(train_set.vocab),
}

class MovieMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Word Embeddings
        self.embeddings = nn.Embedding(
            num_embeddings=config['vocab_size'],
            embedding_dim=config['embedding_dim'],
            padding_idx=0  # Use 0 since PADDING is mapped to 0 in vocab
        )
        
        # Numeric features layer - CHANGED from 3 to 2 features (month, runtime)
        self.numeric_layer = nn.Linear(in_features=2, out_features=config['hidden_dim_numeric'])
        
        # Combined input dimension
        combined_input_dim = config['embedding_dim'] + config['hidden_dim_numeric']
        
        # MLP layers - now with four linear layers instead of three
        self.linear1 = nn.Linear(in_features=combined_input_dim, out_features=config['hidden_dim1'])
        self.linear2 = nn.Linear(in_features=config['hidden_dim1'], out_features=config['hidden_dim2'])
        self.linear3 = nn.Linear(in_features=config['hidden_dim2'], out_features=config['hidden_dim3'])
        self.linear4 = nn.Linear(in_features=config['hidden_dim3'], out_features=config['num_classes'])
        self.relu = nn.ReLU()

    def forward(self, x, lengths, numeric_features):
        # Word Embeddings
        emb = self.embeddings(x)
        # Average embeddings (accounting for padding)
        sentence = emb.sum(dim=1) / lengths.view(-1, 1)
        
        # Process numeric features (2 features: month, runtime)
        numeric_out = self.relu(self.numeric_layer(numeric_features))
        
        combined = torch.cat((sentence, numeric_out), dim=1)
        
        # MLP layers - now using all four layers
        z1 = self.relu(self.linear1(combined))
        z2 = self.relu(self.linear2(z1))
        z3 = self.relu(self.linear3(z2))  # Added third hidden layer
        logits = self.linear4(z3)  # Changed to linear4 for final output
        
        return logits


# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MovieMLP(config).to(device)
print(model)


# In[172]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

# Define hyperparameters at the top
lr = 0.13

model = MovieMLP(config).to(device)
print(model)

# Loss function for regression
loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,  # Using a smaller learning rate because Lion typically works better with lower values
    weight_decay=0.0001,
    betas=(0.9, 0.9998)
)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        X = batch['x'].to(device)
        lengths = batch['lengths'].to(device)
        numeric_features = batch['numeric'].to(device).float()
        y = batch['y'].to(device).float()
        
        pred = model(X, lengths, numeric_features)[:, 0]
        loss = loss_fn(pred, y)
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        optimizer.zero_grad()
        
        if batch_idx % 100 == 0:
            current = (batch_idx + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    return total_loss / len(dataloader)

def test(dataloader, model, loss_fn):
    model.eval()
    predictions = []
    actuals = []
    test_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            X = batch['x'].to(device)
            lengths = batch['lengths'].to(device)
            numeric_features = batch['numeric'].to(device).float()
            y = batch['y'].to(device).float()
            
            pred = model(X, lengths, numeric_features)[:, 0]
            test_loss += loss_fn(pred, y).item()
            
            predictions.extend(pred.cpu().numpy())
            actuals.extend(y.cpu().numpy())
    
    test_loss /= len(dataloader)
    r2 = r2_score(actuals, np.array(predictions))
    print(f"Avg loss: {test_loss:>8f}")
    print(f"R2 Score: {r2:>8f}") 
    
    return test_loss, r2

# Training loop
epochs = 80
best_r2 = -float('inf')  # Initialize with negative infinity for R² score

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    train_loss = train(train_dataloader, model, loss_fn, optimizer)
    val_loss, val_r2 = test(dev_dataloader, model, loss_fn)
    
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation R² Score: {val_r2:.4f}")
    
    if val_r2 > best_r2:  # Save model based on R² score instead of loss
        best_r2 = val_r2
        print(f"New best R² score: {best_r2:.4f}")
        save_path = f"model_lr_{lr:.4f}.pth"
        torch.save(model.state_dict(), save_path)
    else:
        print(f"No improvement in model learning rate {lr}")


# In[173]:


# PART 3e: Load and test the best model

# if you want to load the best model from training before testing:
model = MovieMLP(config).to(device)
model.load_state_dict(torch.load(save_path))
print("testing final model on test...")
test(test_dataloader, model, loss_fn)
print("Done!")


# In[ ]:





# In[ ]:




