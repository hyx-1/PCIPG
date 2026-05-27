# PPI Protein Complex Prediction — SAGE-GAT Dual Graph Neural Network

## Overview

This project uses a **dual graph neural network architecture** (SAGE-GAT) to predict protein complexes from a protein-protein interaction (PPI) network. The overall workflow consists of four stages:

1. **Data preprocessing** — Construct residue-level feature graphs, the PPI adjacency matrix, and protein name mappings.
2. **Model training** — Train the model with unsupervised contrastive learning so that protein embeddings reflect PPI relationships.
3. **Model testing** — Output the protein embedding matrix and generate top-k candidate complex lists.
4. **Evaluation and filtering** — Compute precision, recall, F1, Acc, and other metrics based on gold-standard complexes.

Core idea: The model first uses **amino-acid residue contact graphs** (connectivity between residues) and SAGE graph convolution to obtain protein-level embeddings. It then applies a GAT attention mechanism on the PPI network to learn interactions between proteins. Finally, the model is trained using an unsupervised contrastive loss function.

---

## Directory Structure

```
fuxian/
├── train.py                # Entry point for model training
├── test.py                 # Entry point for model testing / inference
├── Select_eva.py           # Entry point for evaluation and filtering
├── elevation.py            # Evaluation metric functions (F1, Acc, Sn, PPV, precision, recall)
├── utils.py                # Utility functions (file loading, clique detection, PPI edge-ratio calculation)
├── ciluqes.py              # Utility script for computing all cliques on the gold-standard PPI network
├── weight_train.py         # Weighted training version using Unsupvise_weight_loss
├── unsupvise_loss.py       # Definition of the unsupervised contrastive loss function
├── sage_gat.py             # Model definition (SAGE + GAT dual graph neural network)
├── readme.md               # Legacy README
│
├── data/                   # Preprocessed data for each dataset
│   ├── collins/
│   ├── biogrid/
│   ├── krogan_core/
│   ├── krogan14k/
│   ├── dip/
│   └── HCT116/
│
├── models/                 # Trained model weights (.ckpt)
├── logs/                   # TensorBoard training logs
└── result/                 # Output directory for prediction results
```

---

## 1. Model Architecture (`sage_gat.py`)

The model consists of **two serially connected subnetworks**:

```
ppi_model (sage_gat.py:132)
│
├── self.BGNN = SAGE()          # Residue-level graph convolution: residue features → protein embeddings
│
├── self.TGNN = GAT()           # PPI-level graph attention: protein embeddings → final embeddings
│
└── forward(batch, p_x_all, p_edge_all, edge_index)
```

### 1.1 SAGE: Residue-Level Graph Neural Network

**File**: `sage_gat.py` → `class SAGE`

**Function**: Each protein contains multiple amino-acid residues, and residue-residue contacts form a contact map. SAGE models each protein as a subgraph, where residues are nodes and residue contacts are edges. GraphSAGE convolutions aggregate neighborhood information layer by layer.

**Input**:

| Parameter | Shape | Description |
|---|---|---|
| `x` | `[T, 18]` | A large matrix formed by concatenating residue features from all proteins, where T is the total number of residues |
| `edge_index` | `[2, E_amino]` | Residue contact edges, concatenated from all intra-protein residue graphs; E_amino is the total number of residue-level edges |
| `batch` | `[T]` | Indicates which protein each residue belongs to (`0 ~ N-1`), used by `SAGPooling` and `global_mean_pool` |

**Network structure**:

```
Input: [T, 18]
  │
  ├─ SAGEConv(18→128) + FC(128→128) + ReLU + BatchNorm1d
  ├─ SAGPooling(ratio=0.5)     ← Downsample by 50%
  │
  ├─ SAGEConv(128→128) + FC(128→128) + ReLU + BatchNorm1d
  ├─ SAGPooling(ratio=0.5)     ← Downsample by 50%
  │
  ├─ SAGEConv(128→128) + FC(128→128) + ReLU + BatchNorm1d
  ├─ SAGPooling(ratio=0.5)     ← Downsample by 50%
  │
  └─ global_mean_pool          ← Average all remaining residues within each protein
       │
       Output: [N, 128]        ← N = number of proteins
```

**SAGPooling**: Uses a self-attention mechanism to select the most important nodes. Each layer retains 50% of the residue nodes, gradually compressing the subgraph while preserving the most important structural information.

**Output dimension**: Each row corresponds to a 128-dimensional embedding vector for one protein. After the SAGE module, fine-grained residue-level information is aggregated into protein-level representations.

### 1.2 GAT: PPI-Level Graph Attention Network

**File**: `sage_gat.py` → `class GAT`

**Function**: The PPI network is represented as a graph where proteins are nodes and PPIs are edges. GAT uses an attention mechanism to learn the mutual influence between interacting proteins and generates the final protein embeddings.

**Input**:

| Parameter | Shape | Description |
|---|---|---|
| `x` | `[N, 128]` | Protein embeddings output by SAGE, where N is the number of proteins |
| `edge_index` | `[2, E_ppi]` | PPI edge list of an undirected graph, stored symmetrically; E_ppi = number of PPI edges × 2 |

**Network structure**:

```
Input: [N, 128]
  │
  ├─ Linear(128→512) + BatchNorm1d
  ├─ GATConv(512→512, heads=4, concat=False) + ReLU + BatchNorm1d
  │
  ├─ GATConv(512→512, heads=4, concat=False)
  ├─ Linear(512→512) + ReLU + BatchNorm1d
  │
  ├─ GATConv(512→512, heads=4, concat=False)
  ├─ Linear(512→512) + ReLU + BatchNorm1d
  │
  ├─ Linear(512→512) + ReLU + Dropout(0.5)
  │
  └─ Linear(512→2000)
       │
       Output: [N, 2000]       ← Final protein embedding matrix
```

**Output dimension**: Each row corresponds to a **2000-dimensional embedding** for one protein. This embedding space is trained to reflect protein-protein interaction relationships, so proteins connected by PPIs should be closer in the embedding space.

### 1.3 Complete Forward Propagation

**File**: `sage_gat.py` → `ppi_model.forward()`

```python
def forward(self, batch, p_x_all, p_edge_all, edge_index):
    # 1. Move data to GPU
    edge_index = edge_index.to(device)          # [2, E_ppi]
    batch = batch.to(torch.int64).to(device)    # [T]
    x = p_x_all.to(torch.float32).to(device)    # [T, 18]
    edge = torch.LongTensor(p_edge_all.to(torch.int64)).to(device)  # [2, E_amino]

    # 2. SAGE: residue graph → protein embeddings
    embs = self.BGNN(x, edge, batch - 1)        # [N, 128]

    # 3. GAT: PPI graph → final embeddings
    final = self.TGNN(embs, edge_index)          # [N, 2000]

    return final
```

---

## 2. Loss Function (`unsupvise_loss.py`)

### 2.1 `Unsupvise_loss`: Unsupervised Contrastive Loss

**Core idea**: Given the protein embedding matrix F `[N, 2000]` output by the model, if proteins i and j interact in the PPI network, namely `A[i][j] = 1`, their embeddings should be similar and should have a large probability after softmax. If they do not interact, their embeddings should be dissimilar and should have a small probability.

```python
# 1. Construct the PPI adjacency matrix A
A[edges[0], edges[1]] = 1
A[edges[1], edges[0]] = 1                         # Symmetrization

# 2. Compute the cosine similarity matrix and apply softmax
FUV = matmul(F, F.T) / sqrt(2000)                  # [N, N], scaled dot product
FUV.masked_fill_(diag_mask, -inf)                  # Set diagonal entries to -inf
FUV = softmax(FUV, dim=-1)                         # Normalize each row into a probability distribution

# 3. Positive-sample loss for PPI edges
# For each PPI edge (i,j), FUV[i][j] is the probability that i selects j
# The form -log(1 - exp(-eps - p)) is similar to a negative log-likelihood
edges_loss = -log1p(-exp(-eps - FUV[edges[0], edges[1]]))
edges_loss = sum(edges_loss)

# 4. Negative-sample loss for non-PPI edges
non_edges_loss = FUV * (1 - A - I)                  # Similarity scores at non-edge positions
non_edges_loss = sum(non_edges_loss)

# 5. Total loss as a weighted average
loss = (edges_loss / |E| + non_edges_loss / |non_E|) / (1 + neg_scale)
```

**Parameters**:
- `p_no_comm=1e-4`: The false-observation rate, which controls the strength of the positive-sample loss.
- `neg_scale=1.0`: Weight scaling factor for negative samples.

### 2.2 `Unsupvise_weight_loss`: Weighted Unsupervised Loss

This loss adds two terms to the basic loss:
- **`logpipj`**: The negative log-likelihood of embedding similarity for PPI edges, encouraging interacting proteins to have higher cosine similarity.
- **`logsijpij`**: Uses an external weight matrix S, such as confidence scores from HI-union, to assign higher weights to high-confidence PPI edges.

```python
logpipj = -sum(log(FUV[edges[0], edges[1]]))         # Positive-sample log-likelihood
B = S[edges[0], edges[1]]                             # External weights
weight = (1 - B) / B                                  # Higher-confidence edges receive larger weights
logsijpij = sum(weight * FUV[edges[0], edges[1]])     # Weighted similarity
```

---

## 3. Training Workflow (`train.py`)

### 3.1 Inputs

| Parameter | Source | Description |
|---|---|---|
| `--Protein_name` | Preprocessing step | `proteincollins_name.json` — protein name-to-ID mapping |
| `--Amino_conact_matrix` | Preprocessing step | `.npy` — residue contact edge lists for each protein |
| `--x_list_feature` | Preprocessing step | `.pt` — residue feature matrix for each protein |
| `--ppi_npy` | Preprocessing step | `.npy` — PPI adjacency matrix |
| `--model_save` | User-defined | Path for saving model weights |
| `--log` | User-defined | TensorBoard log path |

### 3.2 Data Loading and Processing

```python
# 1. Read the protein name mapping and obtain the number of proteins
with open(args.Protein_name, 'r') as f:
    data = json.load(f)
protein_nodes = len(data)           # N = number of proteins

# 2. Load raw data
p_x_all = torch.load(args.x_list_feature)        # [N, ?] — each element is [num_residues_i, 18]
p_edge_all = np.load(args.Amino_conact_matrix, allow_pickle=True)  # [N, ?] — each element is [2, num_edges_i]
ppi = np.load(args.ppi_npy)                      # [N, N] — PPI adjacency matrix
```

### 3.3 Key Data Transformations

Three key functions convert the original **per-protein list format** into a **large unified graph** format:

#### `multi2big_x` (`train.py L51`)

**Function**: Stack residue features from all proteins into one large matrix.

```
Input: p_x_all = [tensor(N0,18), tensor(N1,18), ..., tensor(Nn-1,18)]
              ↑  protein 0   ↑  protein 1             ↑  protein n-1

Output: x_cat = [concat(all proteins)] = [T, 18]    (T = sum(Ni))
        x_num_index = [N0, N1, ..., Nn-1]           (number of residues in each protein)
```

| Residues of the i-th protein | Row range in the large matrix |
|---|---|
| protein 0 | `[0, N0)` |
| protein 1 | `[N0, N0+N1)` |
| ... | ... |
| protein n-1 | `[sum(N0..Nn-2), sum(N0..Nn-1))` |

#### `multi2big_edge` (`train.py L80`)

**Function**: Stack residue contact edges from all proteins into one large edge set.

```
Input: edge_ori[i] = [2, Ei]    — intra-protein residue contact edges of the i-th protein
                                (node indices are local: 0 ~ Ni-1)

Output: edge_cat = [2, sum(Ei)] — global edge list
                                (node indices are global and require an offset)
```

**Specific steps**:
```python
for i in range(N):
    offset = sum(num_index[:i])    # Total number of residues in proteins before protein i
    edge_cat = cat(edge_cat, edge_ori[i].T + offset)  # Local index → global index
```

| Local edge in protein i | Global offset | Mapped edge |
|---|---|---|
| `(2, 5)` | offset = N0 + ... + Ni-1 | `(2+offset, 5+offset)` |

#### `multi2big_batch` (`train.py L60`)

**Function**: Generate the `batch` tensor, indicating which protein each residue belongs to.

```
Input: x_num_index = [N0, N1, ..., Nn-1]
Output: batch = [0]*N0 + [1]*N1 + ... + [n-1]*Nn-1   (length T)
```

Example:
```
x_num_index = [3, 4, 2]
batch = [0, 0, 0, 1, 1, 1, 1, 2, 2]   (indices 0-2 → protein 0, 3-6 → protein 1, 7-8 → protein 2)
```

### 3.4 PPI Adjacency Matrix Processing

```python
ppi = np.load(args.ppi_npy)          # [N, N] adjacency matrix
protein_edge = torch.tensor(ppi.T)   # Transpose and convert to tensor
```

Format of the PPI adjacency matrix: `ppi[i][j] = 1` indicates that protein i and protein j interact. The transpose `.T` is used to obtain the correct edge-list format for model input.

### 3.5 Training Loop

```python
for epoch in range(1, epochs+1):         # epochs = 2000
    model.train()
    with torch.amp.autocast('cuda'):     # Mixed-precision training
        F = model(batch, p_x_all, p_edge_all, protein_edge)  # [N, 2000]
        optimizer.zero_grad()
        loss = Loss(protein_edge, F)     # Unsupervised contrastive loss
    scheduler.step()                     # StepLR: step_size=100, gamma=0.7
    scaler.scale(loss).backward()        # AMP gradient scaling
    scaler.step(optimizer)
    scaler.update()
    # Print loss every 25 epochs
    # Log train_loss and lr to TensorBoard at every epoch
    # Save a model checkpoint at every epoch
```

**Hyperparameters**:

| Parameter | Value | Description |
|---|---|---|
| `epochs` | 2000 | Number of training epochs |
| `lr` | 0.001 | Initial learning rate of Adam |
| `weight_decay` | 0.01 | L2 regularization |
| `step_size` | 100 | Learning-rate decay step size |
| `gamma` | 0.7 | Learning-rate decay factor |
| `seed` | 42 | Random seed |
| `feature_num` | 18 | Residue feature dimension |

---

## 4. Inference Workflow (`test.py`)

### 4.1 Inputs

| Parameter | Description |
|---|---|
| `--Protein_name` | Protein name mapping JSON |
| `--Amino_conact_matrix` | Residue contact edge list `.npy` |
| `--x_list_feature` | Residue features `.pt` |
| `--ppi_npy` | PPI adjacency matrix `.npy` |
| `--model_save` | Trained model weights `.ckpt` |
| `--PPI_in_Ground_truth` | PPI edge list appearing in the gold standard `.txt` |
| `--save_top_k` | Directory for saving top-k results |

### 4.2 Inference Process

```python
# 1. Load model weights
model = ppi_model()
model.to('cuda')
model.load_state_dict(torch.load(model_path)['state_dict'])
model.eval()

# 2. Forward inference
F = model(batch, p_x_all, p_edge_all, protein_edge)   # [N, 2000]

# 3. For each protein, select the k proteins with the most similar embeddings
def save_topk(k):
    _, indices = torch.topk(F, k, dim=0, largest=True)  # [k, N]
    # Each column (col) corresponds to the top-k similar proteins for one protein
    for col in indices.t():
        protein_set = set(col.tolist())
        if protein_set not in seen:
            seen.append(protein_set)
            proteins = [idx_2_protein[i] for i in col.tolist()]
            protein_list.append(proteins)
    # Save to file
    df = pd.DataFrame(protein_list)
    df.to_csv(f'top_{k}.txt', sep='\t', header=False, index=False)

# 4. Generate candidate complexes for k=3..34
for i in range(3, 35):
    save_topk(i)
```

**Output**: In each `top_k.txt` file, each row is a candidate complex containing k proteins.

### 4.3 Result Merging

```python
def merge_file2(k):
    # 1. Compute all cliques from the gold-standard PPI network
    G = nx.Graph()
    G.add_edges_from(PPI_in_gt_edges)
    PPI_hyperedge = list(nx.find_cliques(G))
    PPI_hyperedge = [c for c in PPI_hyperedge if 3 <= len(c) <= 4]  # Filter by size

    # 2. Write cliques to the beginning of result_sage_gat.txt
    # 3. Append the contents of top_5.txt ~ top_k.txt
    with open('result_sage_gat.txt', 'w') as outfile:
        for clique in PPI_hyperedge:
            outfile.write("\t".join(clique) + "\n")
    with open('result_sage_gat.txt', 'a') as outfile:
        for filename in filenames:
            with open(filename, 'r') as infile:
                outfile.write(infile.read())
```

---

## 5. Evaluation Workflow (`Select_eva.py`)

### 5.1 Inputs

| Parameter | Description |
|---|---|
| `--PPI_path` | PPI network file, such as `collins.txt` |
| `--Protein_path` | Residue feature directory containing all protein files |
| `--PPI_in_Ground_truth` | PPI edges appearing in the gold standard |
| `--result` | Path to `result_sage_gat.txt` |

### 5.2 Evaluation Steps

```python
# 1. Load gold-standard complexes
gt = Load_txt_list(args.PPI_in_Ground_truth)

# 2. Filter gold-standard complexes that are fully covered by the protein set of the PPI network
gt_in_PPI = [complex for complex in gt
             if all(protein in protein_nodes for protein in complex)]

# 3. Read predicted candidate complexes
test_complex = [line.strip().split() for line in file]

# 4. Filtering: keep only candidate complexes with a PPI edge ratio > 0.2
edges_in_complex = cal_prop_of_link(test_complex, PPI_set)
predict_complex = test_complex[edges_in_complex['prop'] > 0.2]

# 5. Compute evaluation metrics
precision, recall, f1, acc, sn, PPV, msg, ... = get_score(gt_in_PPI, predict_complex)
```

### 5.3 Evaluation Metrics (`evaluation.py`)

#### ACC / Sn / PPV

```python
def acc_score(predicted_complex, reference_complex):
    # Sn (Sensitivity): the proportion of proteins in reference complexes that are correctly predicted
    for ref in reference_complex:
        max_overlap = max(len(set(ref) & set(pred)) for pred in predicted_complex)
        Sn = sum(max_overlap) / N_sum

    # PPV (Positive Predictive Value): the proportion of proteins in predicted complexes that are correct
    for pred in predicted_complex:
        max_overlap = max(len(set(pred) & set(ref)) for ref in reference_complex)
        PPV = sum(max_overlap) / B_sum

    # Acc: geometric mean
    Acc = sqrt(Sn * PPV)
    return Acc, Sn, PPV
```

#### Precision / Recall / F1

These metrics are based on complex-level matching, where a complex pair is considered matched if the Overlap Score is greater than 0.2:

```python
def precision_score(predicted_complex, reference_complex):
    # For each predicted complex, find the best-matched reference complex
    for pred in predicted_complex:
        for ref in reference_complex:
            overlap = set(pred) & set(ref)
            score = |overlap|^2 / (|pred| * |ref|)    # Overlap Score
            if score > 0.2:                           # Matching threshold
                match_count += 1
    precision = match_count / len(predicted_complex)

def recall_score(predicted_complex, reference_complex):
    # For each reference complex, find the best-matched predicted complex
    recall = match_count / len(reference_complex)

def get_score(reference_complex, predicted_complex):
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1, acc, Sn, PPV
```

---

## 6. Complete Data-Flow Summary

### 6.1 Data Format Chain

```
Raw data                                           → Preprocessed data           → Training / inference input
────────                                           ────────────────              ───────────────────────────
Residue contact graph of each protein              → Large matrix of all          → edge [2, E_amino]
(.txt or .npy format)                                residue contact edges
                                                     collins_edge_list_amino.npy

18-dimensional residue features of each protein     → Large matrix of all          → x [T, 18]
(22-dimensional features projected to 18             residue features
 dimensions, or original 18-dimensional features)    collins_x_list.pt

PPI network                                         → PPI adjacency matrix         → protein_edge [2, E_ppi]
(collins.txt: two protein IDs per line)               collins_ppi.npy

Protein names                                       → Name-to-ID mapping           → protein_nodes = N
                                                     protein_collins_name.json
```

### 6.2 Input and Output Shapes at Each Stage

| Stage | Function / component | Input shape | Output shape | Description |
|---|---|---|---|---|
| **Data transformation** | `multi2big_x` | list of `[Ni, 18]` | `[T, 18]` + `[N]` | Concatenate residue features |
| **Data transformation** | `multi2big_edge` | list of `[2, Ei]` | `[2, sum(Ei)]` | Concatenate residue edges |
| **Data transformation** | `multi2big_batch` | `[N]` (number of residues) | `[T]` | Residue-to-protein mapping |
| **SAGE** | `SAGE.forward` | `[T, 18]` + `[2, sum(Ei)]` + `[T]` | `[N, 128]` | Protein embeddings |
| **SAGPooling** | ×3 layers | `[T_l, 128]` | `[T_l*0.5^3, 128]` | Progressive downsampling |
| **global_mean_pool** | Final layer | `[T_last, 128]` | `[N, 128]` | Protein-level aggregation |
| **GAT** | `GAT.forward` | `[N, 128]` + `[2, E_ppi]` | `[N, 2000]` | Final embeddings |
| **Loss** | `Unsupvise_loss` | `[2, E_ppi]` + `[N, 2000]` | scalar | Unsupervised contrastive loss |
| **Inference** | `save_topk(k)` | `[N, 2000]` | list of k proteins | Candidate complexes |
| **Evaluation** | `get_score` | predicted complex list + reference complex list | `precision, recall, f1, acc, Sn, PPV` | Six evaluation metrics |

### 6.3 Complete Training Forward Data Flow

```
Raw Features                                                          Shape
────────────                                                          ─────
p_x_all (list of per-protein residue features)                        [N] each Ni×18
  │
  ▼ multi2big_x()
x (concatenated residue features)                                     [T, 18]          T = sum(Ni)
num_index (number of residues per protein)                            [N]
  │
p_edge_all (list of per-protein residue contact edges)                [N] each 2×Ei
  │
  ▼ multi2big_edge()
edge (concatenated residue contact edges, global indices)             [2, E_amino]      E_amino = sum(Ei)
  │
num_index
  ▼ multi2big_batch()
batch (per-residue protein assignment)                                [T]
  │
ppi (PPI adjacency matrix)                                            [N, N]
  │  ▼ .T
protein_edge (PPI edge list)                                          [2, E_ppi]
  │
  ├─────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  ppi_model.forward()                                             │
  │                                                                  │
  │  ┌─────────────────────────────────────────────────────────┐     │
  │  │  SAGE (BGNN)                                             │     │
  │  │  x: [T, 18] + edge: [2, E_amino] + batch: [T]          │     │
  │  │    │ SAGEConv(18→128) + BatchNorm + ReLU                 │     │
  │  │    │ SAGPooling(0.5) → x_down, edge_down, batch_down     │     │
  │  │    │ SAGEConv(128→128) + BatchNorm + ReLU                │     │
  │  │    │ SAGPooling(0.5) → x_down2, edge_down2, batch_down2  │     │
  │  │    │ SAGEConv(128→128) + BatchNorm + ReLU                │     │
  │  │    │ SAGPooling(0.5) → x_down3, edge_down3, batch_down3  │     │
  │  │    │ global_mean_pool(x_down3, batch_down3)              │     │
  │  │    ▼                                                      │     │
  │  │  embs: [N, 128]    ← Protein-level embeddings            │     │
  │  └─────────────────────────────────────────────────────────┘     │
  │           │                                                      │
  │           ▼                                                      │
  │  ┌─────────────────────────────────────────────────────────┐     │
  │  │  GAT (TGNN)                                             │     │
  │  │  embs: [N, 128] + protein_edge: [2, E_ppi]             │     │
  │  │    │ Linear(128→512) + BatchNorm                        │     │
  │  │    │ GATConv(512→512, heads=4) + ReLU + BatchNorm       │     │
  │  │    │ GATConv(512→512, heads=4) + Linear(512→512) + ReLU│     │
  │  │    │   + BatchNorm                                      │     │
  │  │    │ GATConv(512→512, heads=4) + Linear(512→512) + ReLU│     │
  │  │    │   + BatchNorm                                      │     │
  │  │    │ Linear(512→512) + ReLU + Dropout(0.5)              │     │
  │  │    │ Linear(512→2000)                                   │     │
  │  │    ▼                                                      │     │
  │  │  F: [N, 2000]      ← Final protein embeddings            │     │
  │  └─────────────────────────────────────────────────────────┘     │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
           │
           ▼
  Loss (Unsupvise_loss)
  F: [N, 2000] + protein_edge: [2, E_ppi]
    │ Cosine Similarity → FUV = matmul(F, F.T) / sqrt(2000)
    │ Softmax over dim=-1 (mask diagonal)
    │ edges_loss = -log1p(-exp(-eps - FUV[pos_edges]))
    │ non_edges_loss = sum(FUV * non_edge_mask)
    │ loss = (edges_loss/|E| + non_edges_loss/|non_E|) / 2
    ▼
  scalar (backpropagation)
```

---

## 7. Usage

### 7.1 Training

```bash
python train.py \
  --log "logs/train_logs/collins.log" \
  --Protein_name "data/collins/protein_collins_name.json" \
  --Amino_conact_matrix "data/collins/collins_edge_list_amino.npy" \
  --x_list_feature "data/collins/collins_x_list.pt" \
  --ppi_npy "data/collins/collins_ppi.npy" \
  --model_save "models/collins_sage_gat_tarin.ckpt"
```

### 7.2 Testing / Inference

```bash
python test.py \
  --Protein_name "data/collins/protein_collins_name.json" \
  --Amino_conact_matrix "data/collins/collins_edge_list_amino.npy" \
  --x_list_feature "data/collins/collins_x_list.pt" \
  --ppi_npy "data/collins/collins_ppi.npy" \
  --model_save "models/collins_sage_gat_tarin.ckpt" \
  --PPI_in_Ground_truth "data/collins/collins_ppi_in_gt.txt" \
  --save_top_k "result/collins/save_top_k/"
```

### 7.3 Evaluation

```bash
python Select_eva.py \
  --PPI_path "../PPI_Dataset/collins.txt" \
  --Protein_path "../Residue_Fea_Choose/collins/Residue_feature_22/" \
  --PPI_in_Ground_truth "data/collins/collins_ppi_in_gt.txt" \
  --result "result/collins/save_top_k/result_sage_gat.txt"
```

---

## 8. Supported Datasets

The project provides preprocessed data for the following datasets:

| Dataset | Filename prefix | Description |
|---|---|---|
| Collins | `collins_*` | Collins yeast PPI network |
| BioGRID | `biogrid_*` | BioGRID database PPI |
| Krogan Core | `krogan_core_*` | Krogan core PPI subset |
| Krogan 14K | `krogan14k_*` | Krogan 14K PPI dataset |
| DIP | `dip_*` | DIP database PPI |
| HCT116 | `HCT116_*` | HCT116 human cell-line PPI |

---

## 9. Key Takeaways

1. **Dual graph structure**: SAGE processes residue-level contact graphs, i.e., intra-protein structure, while GAT processes the PPI graph, i.e., inter-protein relationships. The two are connected in series to achieve multi-level information aggregation from amino acids to protein complexes.
2. **Unsupervised training**: No labeled complex data are required. The model uses only the PPI network structure as the training signal. Positive samples, namely connected protein pairs, are encouraged to have similar embeddings, while negative samples, namely non-connected protein pairs, are encouraged to be dissimilar.
3. **SAGPooling downsampling**: The residue graph passes through three layers of 50% pooling, is progressively compressed, and is finally aggregated into a protein-level representation through `global_mean_pool`.
4. **Top-k inference**: For each protein, the k most similar proteins in the embedding space are selected as a candidate complex, with k scanned from 3 to 35.
5. **Clique-detection post-processing**: All cliques, i.e., fully connected subgraphs, are computed from gold-standard PPI edges as a baseline and then merged with model predictions.
6. **Overlap Score evaluation**: The matching strategy is based on `|overlap|^2 / (|pred| * |ref|)`, with 0.2 used as the matching threshold.
