import pandas as pd
import os
import gc
import numpy as np

from GLC.data_loading.common import load_patch
from GLC.metrics import top_30_error_rate, top_k_error_rate_from_sets, predict_top_30_set

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

"""
CustomImageDataset ist eine Klasse zur Verwaltung der Datensätze mit denen das Neuronale Netz trainiert werden soll.
Parameter:
    subset = 'train' oder 'val'
"""
class CustomImageDataset(Dataset):
    def __init__(self, subset='train'):
        data = pd.read_csv("../data/observations/observations_fr_train.csv", sep=";", index_col="observation_id")
        ind = data.index[data["subset"] == subset]
        data = data.loc[ind]
        self.observation_ids = data.index
        self.labels = data.species_id.values

    """gibt die Anzahl der Datensätze zurück"""
    def __len__(self):
        return len(self.labels)

    """Für einen übergebenen Index, werden verschiedenee Bilddaten geladen und in einen Tensor umgewandelt.
    Der Tensor wird zusammen mit dem Zielwert (speciesId) als Tupel zurückgegeben."""
    def __getitem__(self, index):
        to_tensor = []
        to_tensor.append(load_patch(self.observation_ids[index], "../data", data="near_ir")[0])
        to_tensor.append(load_patch(self.observation_ids[index], "../data", data="rgb")[0][:,:,2])
        to_tensor.append(load_patch(self.observation_ids[index], "../data", data="rgb")[0][:,:,0])
        to_tensor = np.array(to_tensor)
        return torch.from_numpy(to_tensor).float(), self.labels[index]
    
# Erstellung eines Datasets für die Trainingsdaten sowie die Testdaten
train_dataset = CustomImageDataset('train')
test_dataset = CustomImageDataset('val')

# Partitionierung der Datasets in batches der Größe 32
train_loader = DataLoader(train_dataset, batch_size=32,num_workers=0,shuffle = True,drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0, shuffle = False, drop_last=True)

# Festlegen der Ausgabeklassen
# n_classes = 17036 (Anzahl der Arten in Frankreich + USA)
# n_classes = 4911 (Anzahl der Arten in Frankreich)
# Wahl fiel auf die 17036, sodass das Modell problemlos auf zusätzlich USA ausgeweitet werden könnte
n_classes = 17036

# Definieren des Modells
model = models.resnet50(pretrained=True)
model.fc = nn.Sequential(
               nn.Linear(2048, 4096),
               nn.ReLU(inplace=True),
              nn.Linear(4096, n_classes))

# Berechnung soll auf der Grafikkarte durchgeführt werden (Nvidia-Grafikkarte notwendig)
model = model.cuda()

# Laden des Modells, damit nicht immer neu trainiert werden muss
if os.path.isfile('model/GeoLife_3.pt'):
    print("Load model...")
    model.load_state_dict(torch.load('model/GeoLife_3.pt'))

"""
Methode, zum trainieren des neuronalen Netzes
Parameter:
    epoch: Anzahl der Trainingsdurchläufe
    model: neuronales Netz, welches trainiert werden soll
    train_loader: Daten, mit denen trainiert werden soll
    optimizer: gewählter Optimierer
    loss_fn: gewählte Loss-Funktion
"""
def train(epoch, model, train_loader, optimizer, loss_fn):
    model.train()
    for j in range(0, epoch):
        for i, data in enumerate(train_loader):
            inputs, targets = data
            
            inputs = inputs.float().cuda()
            targets = targets.cuda()
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
        
            loss = loss_fn(outputs, targets)
            loss.backward()
        
            optimizer.step()
    model.eval()
    return model

gc.collect()
# Definieren Optimizer und Loss-Funktion
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
loss_fn = nn.CrossEntropyLoss()
# Starten des Trainingsprozesses
# Anpassung der Trainingsrunden notwendig, wenn man das geladene Modell nochmal trainieren möchte
train(0, model, train_loader, optimizer, loss_fn)

# Speichern des Modells
torch.save(model.state_dict(), "model/GeoLife_3.pt")

# Evaluierung der Modells anhand der Testdaten
model.eval()
result = []

for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs = inputs.float()
        inputs = inputs.cuda()
        outputs = model(inputs)
        
        result.append(top_30_error_rate(labels.cpu(), outputs.cpu().detach()))
        
print("Fehlerrate: ", np.mean(result))

# Ergebnisse
# - Trainingsdurchlauf 1: 0,79946
# - Trainingsdurchlauf 2: 0,77321
# - Trainingsdurchlauf 3: 0,75294