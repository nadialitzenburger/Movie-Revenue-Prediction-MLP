#!/usr/bin/env python
# coding: utf-8

# In[3]:


import csv
import os
from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

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
                texts = t.lower()
            words = t.split()
        
            for word in words:
                if word not in vocab:
                    vocab [word] = len(vocab)
        
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
        
        with open(filename, encoding = "utf-8") as csvfile:
            csvreader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
            for row in csvreader:
                if not row['id'] in split_indices:
                    continue
                
                # TODO: adapt to the inputs and output of your task
                text = row['title']
                if lower and text is not None:
                    text = text.lower()
                texts.append(text)

                try:
                    vote_average = float(row['vote_average'])
                    runtime = float(row['runtime'])
                    vote_count = float(row['vote_count'])
                except ValueError:
                # Falls ein Fehler auftritt (z.B. ungültige Werte), mit 0 füllen
                    vote_average = runtime = vote_count = 0.0

                numeric_features.append([runtime, vote_average, vote_count])
                labels.append(float(row['revenue']))

        numeric_features = np.array(numeric_features)

        # Z-Score Normierung der Werte:
        numeric_data = (numeric_features - numeric_features.mean(axis=0)) / numeric_features.std(axis=0)

       
        return texts, labels, numeric_data

    def convert_text_to_tensors(self, text, lower):
        python_list = []
        for text_instance in text:
            if lower:
                text_instance = text_instance.lower()
            idx_instance = []
            for word in text_instance.split():
                if not word in self.vocab:
                    word = UNKNOWN
                idx = self.vocab[word]
                idx_instance.append(idx)
            python_list.append(idx_instance)
        # the instances in the list are of different length
        # for numpy arrays and pytorch tensors we need the same length
        # solution: padding and cropping
        print(len(python_list))
        python_list_padded, instance_lengths = self.pad(python_list)
        print(len(python_list_padded))
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

    



# In[4]:


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
            #if len(ids) > 5000:
           #     break
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


# In[5]:


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


# In[6]:


for batch in test_dataloader:
    X = batch['x']  # Texte (Filmtitel als Indizes)
    numeric_features = batch['numeric']  # Numerische Features (vote_average, runtime, vote_count)
    y = batch['y']  # Umsatz (Label)

    print(f"Shape of X: {X.shape}")  # Größe der Titel-Daten
    print(f"Shape of numeric_features: {numeric_features.shape}")  # Größe der numerischen Features
    print(f"Shape of y: {y.shape} {y.dtype}")  # Größe der Labels (Zielwerte)

    print("Beispiel für X:", X)
    print("Beispiel für numeric_features:", numeric_features)
    
    break  # Nur einen Batch ausgeben


# In[7]:


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    
print(f"Using {device} device")


# In[8]:


from torch import nn

config = {'num_classes': 1,
          'embedding_dim': 100, #100-300
          'hidden_dim1': 118,
          'hidden_dim2': 110,
          'hidden_dim_numeric': 82,
          'vocab_size': len(train_set.vocab),
         }

# PART 2: Define model class

# PART 2: Define the neural network

class MovieMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Word Embeddings
        self.embeddings = nn.Embedding(
            num_embeddings=config['vocab_size'],
            embedding_dim=config['embedding_dim'],
            padding_idx=config['vocab_size'] - 1
        )

        # Numerische Features werden durch eine eigene Schicht verarbeitet
        self.numeric_layer = nn.Linear(in_features=3, out_features=config['hidden_dim_numeric'])
        
        # Korrigierte Input-Größe
        combined_input_dim = config['embedding_dim'] + config['hidden_dim_numeric']

        # MLP-Schichten
        self.linear1 = nn.Linear(in_features=combined_input_dim, out_features=config['hidden_dim1'])
        self.linear2 = nn.Linear(in_features=config['hidden_dim1'], out_features=config['hidden_dim2'])
        self.linear3 = nn.Linear(in_features=config['hidden_dim2'], out_features=config['num_classes'])

        self.relu = nn.ReLU()

    def forward(self, x, lengths, numeric_features):
        # Word Embeddings
        emb = self.embeddings(x)
        sentence = emb.sum(dim=1) / lengths.view(-1, 1)

        # Numerische Features verarbeiten
        numeric_out = self.relu(self.numeric_layer(numeric_features))

        # Text + numerische Features kombinieren
        combined = torch.cat((sentence, numeric_out), dim=1)  # Jetzt richtige Größe

        # Durch das MLP-Netzwerk
        z1 = self.relu(self.linear1(combined))
        z2 = self.relu(self.linear2(z1))
        logits = self.linear3(z2)  # KEIN ReLU nach der letzten Schicht!

        return logits

model = MovieMLP(config).to(device)
print(model)


# In[21]:


#from sklearn.metrics import f1_score, classification_report, mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

model = MovieMLP(config).to(device)
print(model)
print(model.parameters())

# PART 3: Training loop

loss_fn = nn.MSELoss()


# PART 3b: Training function
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()  # model will be trained here (important for some layers)
    for batch_idx, batch in enumerate(dataloader):
        X = batch['x'].to(device)
        lengths = batch['lengths'].to(device)
        numeric_features = batch['numeric'].to(device).float()  # Numerische Features hinzufügen!
        y = batch['y'].to(device).float()

        # Compute prediction and loss function
        pred = model(X, lengths, numeric_features)[:, 0]  # Numerische Features übergeben!
     
        # get predictions (forward pass)
        loss = loss_fn(pred, y.float())  # calculate loss function

        # Backpropagation
        loss.backward()  # calculate gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5) #args.clip)
        optimizer.step()  # update parameters
        optimizer.zero_grad()  # set gradients back to zero

        if batch_idx % 100 == 0:  # optional: print some statistics
            loss, current = loss.item(), (batch_idx + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
# PART 3c: Testing function
def test(dataloader, model, loss_fn):
    predictions = []
    labels = []
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()  # model will be evaluated here (important for some layers)
    test_loss, correct = 0, 0
    with torch.no_grad():  # tell pytorch to do no updates here
        for batch_idx, batch in enumerate(dataloader):
            X = batch['x'].to(device)
            lengths = batch['lengths'].to(device)
            numeric_features = batch['numeric'].to(device).float()  # Numerische Features hinzufügen!
            y = batch['y'].to(device).float()

            outputs = model(X, lengths, numeric_features)[:, 0]  # get predictions
            test_loss += loss_fn(outputs, y).item()  # accumulate loss
            # important: use item() in the line above or tonumpy() or something like this
            # to accumulate only the value of the tensor without gradients / computation graph information
            # otherwise: memory issues

            predictions.extend(outputs.cpu().numpy().flatten())
            labels.extend(y.cpu().numpy().flatten())
            
    mse_score = r2_score(labels, predictions)
            
    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")
    print(f"R2 Score: {mse_score:>8f}")

    #plt.hist(predictions)
    #plt.show() 
    #plt.hist(labels)
    #plt.show() 
    
    return mse_score

# PART 3d: Start the training
epochs = 25

#lr=0.0100111
lr = 0.020011

model = MovieMLP(config).to(device)

# Initialize the optimizer with the current learning rate and weight decay = 0.0001001
optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay = 0.0001001)

# Track best MSE score for early stopping or model saving
best_mse_score = -1000

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    train(train_dataloader, model, loss_fn, optimizer)  # Train the model with the current optimizer

    mse_score = test(dev_dataloader, model, loss_fn)  # Test the model with the current optimizer
        
    if mse_score > best_mse_score:  # Early stopping based on the best score
        save_path = f"model_lr_{lr:.4f}.pth"  # Save model for each learning rate
        torch.save(model.state_dict(), save_path)
        best_mse_score = mse_score
        print(f"New best model with R2 score: {mse_score}")
    else:
        print(f"No improvement in model learning rate {lr}")


# In[22]:


# PART 3e: Load and test the best model

# if you want to load the best model from training before testing:
model = MovieMLP(config).to(device)
model.load_state_dict(torch.load(save_path))
print("testing final model on test...")
test(test_dataloader, model, loss_fn)
print("Done!")


# In[ ]:




