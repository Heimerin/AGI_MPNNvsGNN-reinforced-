import os
import sys
from pprint import pprint as pp
from time import time as tt

import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pandas as pd
import seaborn as sns
import random
import torch
import pickle

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv # Nowy import dla GCNConv
from torch_scatter import scatter_add # Wymagane przez make_mlp jeśli włączony jest layer_norm z scatter_add

# Get rid of RuntimeWarnings, gross
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 2. Konfiguracja Urządzenia ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Używane urządzenie: {device}")

# --- 3. Funkcja Pomocnicza: make_mlp ---
# Ta funkcja jest używana do budowania prostych sieci neuronowych (MLP)
def make_mlp(input_size, sizes,
             hidden_activation='ReLU',
             output_activation='ReLU',
             layer_norm=False):
    """Construct an MLP with specified fully-connected layers."""
    hidden_activation_fn = getattr(nn, hidden_activation)
    output_activation_fn = getattr(nn, output_activation) if output_activation is not None else None

    layers = []
    n_layers = len(sizes)
    current_input_size = input_size

    # Hidden layers
    for i in range(n_layers - 1):
        layers.append(nn.Linear(current_input_size, sizes[i]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i]))
        layers.append(hidden_activation_fn())
        current_input_size = sizes[i]

    # Final layer
    layers.append(nn.Linear(current_input_size, sizes[-1]))
    if output_activation_fn is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1]))
        layers.append(output_activation_fn())

    return nn.Sequential(*layers)

# --- 4. Wczytywanie Danych i Budowanie Grafów ---


DATA_PATH = 'C:/Users/jakub/agi/data.pickle' 

print(f"\nWczytuję dane z: {DATA_PATH}")
with open(DATA_PATH, 'rb') as f:
    data_raw = pickle.load(f)
print(f"Wczytano {len(data_raw)} zdarzeń.")

print("\n=== ANALIZA SUROWYCH DANYCH ===")

def check_track_ids(data_raw):
    for event_idx, event in enumerate(data_raw[:5]):  # Sprawdź pierwsze 5 eventów
        track_ids = [hit[3] for hit in event]
        unique_tracks = set(track_ids)
        print(f"Event {event_idx}: {len(event)} hits, {len(unique_tracks)} unikalnych tracków")
        
        # Sprawdź rozkład tracków
        from collections import Counter
        track_counts = Counter(track_ids)
        print(f"  Rozkład tracków: {track_counts.most_common(5)}")
        
        # Dodatkowo: sprawdź ile hitów ma każdy track
        print(f"  Przykładowe tracki: ", end="")
        for track_id, count in track_counts.most_common(3):
            print(f"Track {track_id}: {count} hits, ", end="")
        print()

check_track_ids(data_raw)

# DODAJ TAKŻE ANALIZĘ WARSTW:
def check_layer_distribution(data_raw):
    print("\n=== ANALIZA ROZKŁADU WARSTW ===")
    zs_layers = [17, 21, 89, 93, 117, 121, 189, 193, 217, 221, 289, 293]
    
    for event_idx, event in enumerate(data_raw[:3]):  # Pierwsze 3 eventy
        print(f"\nEvent {event_idx}:")
        layer_hits = {layer: 0 for layer in zs_layers}
        
        for hit in event:
            r = hit[0]  # współrzędna R
            # Znajdź najbliższą warstwę
            closest_layer = min(zs_layers, key=lambda z: abs(z - r))
            if abs(closest_layer - r) < 2.0:  # Taki sam warunek jak w buildGraphs
                layer_hits[closest_layer] += 1
        
        for layer in zs_layers:
            if layer_hits[layer] > 0:
                print(f"  Warstwa {layer}: {layer_hits[layer]} hitów")

check_layer_distribution(data_raw)

def buildGraphs(data_events):
    graphs = []
    # Z-coordinates of detector layers (example values, adjust if needed)
    # These values are crucial for defining 'neighboring' layers
    zs_layers = [17, 21, 89, 93, 117, 121, 189, 193, 217, 221, 289, 293]
    
    for event_idx, event in enumerate(data_events):
        event = sorted(event, key=lambda x: x[0]) # Sort by first coordinate (R)
        X = [[hit[0], hit[1], hit[2]] for hit in event]  # Features: [R, PHI, Z]

        edge_index = []
        
        # Iterate over all possible pairs of hits to form potential edges
        for i in range(len(event)):
            r1 = event[i][0]
            # Find which layer hit 'i' belongs to
            ir1 = next((idx for idx, z in enumerate(zs_layers) if abs(r1 - z) < 2.0), None)
            if ir1 is None:
                continue # Hit is not on a defined layer, skip

            for j in range(len(event)):
                r2 = event[j][0]
                ir2 = next((idx for idx, z in enumerate(zs_layers) if abs(r2 - z) < 2.0), None)
                if ir2 is None: continue
                
                # POZYTYWNE: ta sama cząstka, sąsiednie warstwy
                if abs(ir1 - ir2) == 1 and event[i][3] == event[j][3]:
                    edge_index.append([i, j])
                # NEGATYWNE: różne cząstki, dowolne warstwy (lub ogranicz do 2-3 warstw różnicy)
                elif abs(ir1 - ir2) <= 2 and event[i][3] != event[j][3]:
                    # LOSUJ część negatywów żeby nie przytłoczyć
                    if random.random() < 0.3:  # Tylko 30% możliwych negatywów
                        edge_index.append([i, j])

        # If no edges were formed for this event, skip it to avoid empty graphs
        if not edge_index:
            # print(f"Ostrzeżenie: Zdarzenie {event_idx} pominięte ze względu na brak krawędzi.")
            continue 

        # Assign edge labels: 1 if tracks are the same, 0 otherwise
        # `event[k][3]` is the Track ID
        y_labels = [1 if event[edge[0]][3] == event[edge[1]][3] else 0 for edge in edge_index]

        graphs.append(Data(
            x=torch.tensor(X, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            y=torch.tensor(y_labels, dtype=torch.float)
        ))
    return graphs

# Nazwa pliku do zapisu/odczytu zbudowanych grafów
GRAPHS_CACHE_PATH = 'muone_graphs_gcn.pickle'

# Budowanie i zapisywanie grafów (lub wczytywanie z cache'u)
graphs = []
if os.path.exists(GRAPHS_CACHE_PATH):
    print(f"Wczytuję grafy z pliku cache: {GRAPHS_CACHE_PATH}")
    with open(GRAPHS_CACHE_PATH, 'rb') as f:
        graphs = pickle.load(f)
else:
    print(f"Plik cache '{GRAPHS_CACHE_PATH}' nie znaleziony. Buduję grafy...")
    graphs = buildGraphs(data_raw)
    with open(GRAPHS_CACHE_PATH, 'wb') as f:
        pickle.dump(graphs, f)
    print(f"Grafy zbudowane i zapisane do {GRAPHS_CACHE_PATH}")

print("\n" + "="*60)
print("ANALIZA ZBUDOWANYCH GRAFÓW")
print("="*60)

def analyze_built_graphs(graphs):
    total_pos = 0
    total_neg = 0
    total_edges = 0
    
    print("Pierwsze 5 zbudowanych grafów:")
    for i, graph in enumerate(graphs[:5]):
        num_edges = graph.edge_index.shape[1]
        num_pos = graph.y.sum().item()
        num_neg = num_edges - num_pos
        
        total_edges += num_edges
        total_pos += num_pos
        total_neg += num_neg
        
        print(f"Graph {i}: {num_edges} krawędzi, "
              f"{num_pos} pozytywnych ({num_pos/num_edges*100:.1f}%), "
              f"{num_neg} negatywnych ({num_neg/num_edges*100:.1f}%)")
    
    print(f"\nSUMA wszystkich {len(graphs)} grafów:")
    print(f"  Łącznie krawędzi: {total_edges}")
    print(f"  Pozytywnych: {total_pos} ({total_pos/total_edges*100:.1f}%)")
    print(f"  Negatywnych: {total_neg} ({total_neg/total_edges*100:.1f}%)")
    
    return total_pos / total_edges if total_edges > 0 else 0  # proporcja pozytywnych

pos_ratio = analyze_built_graphs(graphs)
print(f"Globalna proporcja krawędzi pozytywnych: {pos_ratio:.3f}")

# Oblicz POS_WEIGHT na podstawie rzeczywistych danych
total_pos = sum(g.y.sum().item() for g in graphs)
total_neg = sum((1 - g.y).sum().item() for g in graphs)
POS_WEIGHT = total_neg / total_pos if total_pos > 0 else 1.0
print(f"Obliczona POS_WEIGHT: {POS_WEIGHT:.3f}")



# Ensure edge_index is of correct type and move each graph to the specified device
for g in graphs:
    g.edge_index = g.edge_index.long()
    g.to(device) # Move data to GPU/CPU

# --- 5. Podział Danych i DataLoader ---
train_size = int(0.8 * len(graphs))
train_dataset = graphs[:train_size]
test_dataset = graphs[train_size:]

# DataLoaders for batching graphs
# Batch size może być mniejszy dla GCN ze względu na złożoność grafów
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) # No need to shuffle test set

print(f"\nLiczba grafów treningowych: {len(train_loader.dataset)}")
print(f"Liczba grafów walidacyjnych: {len(test_loader.dataset)}")
print(f"Rozmiar batcha: {BATCH_SIZE}")

# --- 6. Definicja Modelu GCN_EdgeClassifier ---
class GCN_EdgeClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_gcn_layers, edge_classifier_hidden_dims, layer_norm=True):
        super(GCN_EdgeClassifier, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_gcn_layers = num_gcn_layers
        self.apply_layer_norm = layer_norm # Flaga do stosowania LayerNorm po aktywacji

        # Warstwy GCN do uczenia embeddingów węzłów
        self.gcn_layers = nn.ModuleList()
        # Pierwsza warstwa GCN
        self.gcn_layers.append(GCNConv(input_dim, hidden_dim))
        # Pozostałe warstwy GCN
        for _ in range(num_gcn_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))

        # Opcjonalne LayerNorm dla wyjść warstw GCN (stosowane po ReLU)
        if self.apply_layer_norm:
            self.gcn_layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_gcn_layers)])


        # Klasyfikator krawędzi (MLP)
        # Przyjmuje połączone cechy dwóch węzłów (2 * hidden_dim)
        self.edge_classifier = make_mlp(2 * hidden_dim,
                                        edge_classifier_hidden_dims + [1], # Ostatnia warstwa ma 1 wyjście (logit)
                                        output_activation=None, # Bez aktywacji na wyjściu dla BCEWithLogitsLoss
                                        layer_norm=layer_norm) # LayerNorm wewnątrz MLP klasyfikatora (opcjonalnie)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        send_idx = torch.cat([edge_index[0], edge_index[1]], dim=0)
        recv_idx = torch.cat([edge_index[1], edge_index[0]], dim=0)
        bi_edge_index = torch.stack([send_idx, recv_idx], dim=0)
        x0 = x

        # Przekazywanie informacji przez warstwy GCN
        for i, conv_layer in enumerate(self.gcn_layers):
            x = conv_layer(x, edge_index) # Agregacja i transformacja cech sąsiadów
            # Aktywacja i opcjonalna normalizacja dla warstw ukrytych GCN
            x = F.relu(x) # Aktywacja po każdej warstwie GCN
            if self.apply_layer_norm:
                # Stosujemy LayerNorm po aktywacji
                x = self.gcn_layer_norms[i](x)
            if i % 2 == 1 and x.shape == x0.shape:
                x = x + x0
                x0 = x
        
        # Na tym etapie 'x' zawiera końcowe embeddingi dla wszystkich węzłów
        # Użyj ich do klasyfikacji krawędzi

        start_idx, end_idx = data.edge_index
        
        # Połącz embeddingi węzła początkowego i końcowego dla każdej krawędzi
        clf_inputs = torch.cat([x[start_idx], x[end_idx]], dim=1)
        
        # Przepuść przez klasyfikator krawędzi
        return self.edge_classifier(clf_inputs).squeeze(-1) # Zwróć logity

# --- 7. Funkcje Treningu i Ewaluacji ---
def train_model(model, train_loader, optimizer, pos_weight):
    model.train()
    correct_predictions = 0
    total_edges = 0
    total_loss_sum = 0
    for batch in train_loader:
        optimizer.zero_grad()
        data = batch.to(device)
        pred_logits = model(data) # Model zwraca logity
        
        # Obliczenie straty
        # pos_weight jest używane do radzenia sobie z niezbalansowanymi klasami
        loss = F.binary_cross_entropy_with_logits(pred_logits, data.y, pos_weight=torch.tensor(pos_weight, device=device))
        
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Klipowanie gradientów
        optimizer.step()
        
        total_loss_sum += loss.item() * data.num_edges # Sumujemy stratę ważoną liczbą krawędzi w batchu

        # Obliczanie dokładności (thresholding na 0.5 po sigmoid)
        predicted_labels = (torch.sigmoid(pred_logits) > 0.5).float()
        correct_predictions += (predicted_labels == data.y).sum().item()
        total_edges += data.num_edges
    
    avg_loss = total_loss_sum / total_edges if total_edges > 0 else 0
    accuracy = correct_predictions / total_edges if total_edges > 0 else 0
    return accuracy, avg_loss

def evaluate_model(model, test_loader, pos_weight):
    model.eval() # Ustawienie modelu w tryb ewaluacji
    correct_predictions = 0
    total_edges = 0
    total_loss_sum = 0
    with torch.no_grad(): # Ważne: wyłącz gradienty podczas ewaluacji
        for batch in test_loader:
            data = batch.to(device)
            pred_logits = model(data)
            
            # Obliczenie straty
            loss = F.binary_cross_entropy_with_logits(pred_logits, data.y, pos_weight=torch.tensor(pos_weight, device=device))
            total_loss_sum += loss.item() * data.num_edges

            # Obliczanie dokładności
            predicted_labels = (torch.sigmoid(pred_logits) > 0.5).float()
            correct_predictions += (predicted_labels == data.y).sum().item()
            total_edges += data.num_edges

    avg_loss = total_loss_sum / total_edges if total_edges > 0 else 0
    accuracy = correct_predictions / total_edges if total_edges > 0 else 0
    return accuracy, avg_loss

# --- 8. Konfiguracja Modelu, Optymalizatora i Pętli Treningowej ---

# Parametry GCN
gcn_configs = {
    "input_dim": 3,  # Cechy węzłów: R, PHI, Z
    "hidden_dim": 32, # Rozmiar ukrytych embeddingów węzłów po GCN
    "num_gcn_layers": 8, # Liczba warstw GCN
    "edge_classifier_hidden_dims": [64, 32, 16], # Warstwy ukryte w MLP klasyfikatora krawędzi
    "layer_norm": True # Czy stosować LayerNorm po warstwach GCN i w MLP
}

# Inicjalizacja modelu GCN
model_gcn = GCN_EdgeClassifier(**gcn_configs).to(device)
print(f"\nArchitektura modelu GCN:\n{model_gcn}")
print(f"Model GCN przeniesiony na: {device}")

# Optymalizator
optimizer_gcn = torch.optim.Adam(model_gcn.parameters(), lr=0.0005, weight_decay=1e-3, amsgrad=True)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_gcn, mode='min', factor=0.5, patience=10)

def calculate_pos_weight(loader):
    pos = 0
    neg = 0
    for batch in loader:
        pos += batch.y.sum().item()
        neg += (1 - batch.y).sum().item()
    return neg / pos if pos > 0 else 1.0

POS_WEIGHT = calculate_pos_weight(train_loader)
print(f"Obliczona waga dla klasy pozytywnej (pos_weight): {POS_WEIGHT}")


# Hiperparametry treningu
epochs = 100 # Liczba epok
# Waga dla pozytywnej klasy (true edges). Dostosuj w zależności od balansu klas.
# Jeśli true_edges << false_edges, pos_weight > 1.
# Przykład: 1000 true_edges, 9000 false_edges -> 9000/1000 = 9
print(f"Używana waga dla klasy pozytywnej (pos_weight): {POS_WEIGHT}")

# Listy do przechowywania wyników dla wykresów
train_losses_gcn = []
train_accuracies_gcn = []
val_losses_gcn = []
val_accuracies_gcn = []

print(f"\nRozpoczynanie treningu GCN na {epochs} epokach...")

# --- Główna pętla treningowa ---
start_time = tt()
for epoch in range(epochs):
    epoch_start_time = tt()
    # Trening
    train_acc, train_loss = train_model(model_gcn, train_loader, optimizer_gcn, POS_WEIGHT)
    train_accuracies_gcn.append(train_acc)
    train_losses_gcn.append(train_loss)

    # Ewaluacja
    val_acc, val_loss = evaluate_model(model_gcn, test_loader, POS_WEIGHT)
    val_accuracies_gcn.append(val_acc)
    val_losses_gcn.append(val_loss)

    scheduler.step(val_loss)  # Aktualizacja harmonogramu LR na podstawie straty walidacyjnej

    epoch_end_time = tt()
    print(f'Epoch {epoch+1}/{epochs} (Czas: {(epoch_end_time - epoch_start_time):.2f}s): '
          f'Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f} | '
          f'Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}')

end_time = tt()
print(f"\nTrening zakończony! Całkowity czas: {(end_time - start_time):.2f} sekund.")

# --- 9. Generowanie Wykresów Wyników Treningu ---
print("\nGenerowanie wykresów strat i dokładności...")

plt.figure(figsize=(14, 6))


plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_accuracies_gcn, label='Train Accuracy', color='skyblue')
plt.plot(range(1, epochs + 1), val_accuracies_gcn, label='Validation Accuracy', color='salmon')
plt.title('GCN: Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(0, 1) # Dokładność w zakresie 0-1


plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_losses_gcn, label='Train Loss', color='skyblue')
plt.plot(range(1, epochs + 1), val_losses_gcn, label='Validation Loss', color='salmon')
plt.title('GCN: Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
# plt.yscale('log') # Możesz użyć skali logarytmicznej, jeśli strata bardzo spada

plt.tight_layout()
plt.show()

print("\nStdOut without errors")