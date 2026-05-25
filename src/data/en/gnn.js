export const gnnEn = [
  {
    title: "Graph Neural Networks: Graph Math & Core Concepts",
    description: "What is a graph, nodes/edges/adjacency matrix, degree matrix, Laplacian, why standard ML fails on graphs — hands-on with NetworkX.",
    date: "২৩ মে, ২০২৬",
    category: "Graph Neural Network",
    readTime: 11,
    slug: "gnn-graph-basics",
    content: `
      <h3>1. What is a Graph?</h3>
      <p>A graph G = (V, E) consists of <strong>nodes (vertices) V</strong> and <strong>edges E</strong>. Graphs naturally model relationships that grids and sequences cannot.</p>
      <table>
        <thead><tr><th>Domain</th><th>Nodes</th><th>Edges</th><th>ML Task</th></tr></thead>
        <tbody>
          <tr><td>Social network</td><td>Users</td><td>Friendships</td><td>Community detection, link prediction</td></tr>
          <tr><td>Molecule</td><td>Atoms</td><td>Chemical bonds</td><td>Property prediction (toxic/safe)</td></tr>
          <tr><td>Citation network</td><td>Papers</td><td>Citations</td><td>Topic classification</td></tr>
          <tr><td>Knowledge graph</td><td>Entities</td><td>Relations</td><td>Question answering</td></tr>
          <tr><td>Road network</td><td>Intersections</td><td>Roads</td><td>Traffic prediction</td></tr>
        </tbody>
      </table>

      <h3>2. Graph Representations</h3>
      <pre><code>import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Create a simple social network
G = nx.Graph()
G.add_nodes_from([0, 1, 2, 3, 4])
G.add_edges_from([(0,1),(0,2),(1,2),(1,3),(3,4)])

# Adjacency matrix A: A[i,j]=1 if edge(i,j) exists
A = nx.to_numpy_array(G)
print("Adjacency matrix A:")
print(A)

# Degree matrix D: diagonal, D[i,i] = degree of node i
degrees = dict(G.degree())
D = np.diag([degrees[i] for i in sorted(G.nodes())])
print("\nDegree matrix D (diagonal):", np.diag(D))

# Graph Laplacian L = D - A
L = D - A
print("\nLaplacian L:\n", L)

# Node feature matrix X (e.g., age and activity score per user)
X = np.array([[25, 0.8], [30, 0.5], [22, 0.9], [35, 0.3], [28, 0.7]])
print(f"\nNode features X: {X.shape}")  # (5 nodes, 2 features)

# Visualize
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='steelblue', node_size=500, font_color='white')
plt.title('Simple Social Network'); plt.show()</code></pre>

      <h3>3. Why Standard ML Fails on Graphs</h3>
      <pre><code"># Problem 1: Variable-size neighborhoods
#   Node 0 has 2 neighbors, Node 3 has 2 neighbors — but different nodes
#   Dense layers require fixed-size input

# Problem 2: No fixed node ordering
#   Relabeling nodes A→B→C→D or D→C→B→A is the same graph
#   Model must be PERMUTATION INVARIANT

# Problem 3: Structural information is in edges
#   Flattening adjacency matrix loses relational structure

# GNN solution: message passing
#   Each node aggregates information from its neighbors
#   Aggregation must be order-invariant (sum, mean, max)
#   Works regardless of graph size or node ordering

# Neighbor feature aggregation (conceptual):
def aggregate_neighbors(A, X):
    # For each node: sum of its neighbors' features
    return A @ X   # (N x N) @ (N x F) = (N x F)

agg = aggregate_neighbors(A, X)
print("Aggregated neighbor features:\n", agg)</code></pre>

      <h3>4. Graph ML Task Types</h3>
      <table>
        <thead><tr><th>Level</th><th>Task</th><th>Example</th><th>Output</th></tr></thead>
        <tbody>
          <tr><td>Node-level</td><td>Node classification</td><td>Classify paper topics in citation network</td><td>Label per node</td></tr>
          <tr><td>Node-level</td><td>Node regression</td><td>Predict user engagement score</td><td>Value per node</td></tr>
          <tr><td>Edge-level</td><td>Link prediction</td><td>Will two users become friends?</td><td>Probability per edge</td></tr>
          <tr><td>Graph-level</td><td>Graph classification</td><td>Is this molecule toxic?</td><td>Label per graph</td></tr>
          <tr><td>Graph-level</td><td>Graph regression</td><td>Predict molecule's boiling point</td><td>Value per graph</td></tr>
        </tbody>
      </table>
    `,
  },
  {
    title: "Graph Convolutional Networks (GCN): Learning from Neighbors",
    description: "Kipf & Welling's GCN — spectral to spatial, normalized adjacency matrix, layer-wise propagation rule, numpy implementation, and PyTorch Geometric on the Cora dataset.",
    date: "২৩ মে, ২০২৬",
    category: "Graph Neural Network",
    readTime: 13,
    slug: "gnn-gcn",
    content: `
      <h3>1. GCN Intuition</h3>
      <p>A GCN layer updates each node's representation by aggregating features from its neighbors — similar to a convolutional filter, but on irregular graph structure instead of a grid.</p>
      <pre><code># GCN propagation rule (Kipf & Welling, 2017):
#
# H^(l+1) = sigma( D_tilde^(-1/2) * A_tilde * D_tilde^(-1/2) * H^(l) * W^(l) )
#
# A_tilde = A + I    (add self-loops: each node includes itself)
# D_tilde = degree matrix of A_tilde
# H^(l)   = node feature matrix at layer l  (H^(0) = X, input features)
# W^(l)   = trainable weight matrix
# sigma   = activation (ReLU for hidden, softmax for output)
#
# D_tilde^(-1/2) * A_tilde * D_tilde^(-1/2) = symmetric normalization
# → prevents exploding/vanishing features from high/low degree nodes</code></pre>

      <h3>2. NumPy Implementation from Scratch</h3>
      <pre><code">import numpy as np

def gcn_layer(A, H, W):
    # Add self-loops
    A_tilde = A + np.eye(A.shape[0])
    # Degree matrix of A_tilde
    D_tilde = np.diag(A_tilde.sum(axis=1))
    # Symmetric normalization: D^(-1/2) A D^(-1/2)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D_tilde)))
    A_norm     = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    # Propagation
    return np.maximum(0, A_norm @ H @ W)   # ReLU

# Toy example: 5 nodes, 3 input features, 4 hidden units
np.random.seed(42)
A = np.array([[0,1,1,0,0],[1,0,1,1,0],[1,1,0,0,0],
              [0,1,0,0,1],[0,0,0,1,0]], dtype=float)
X = np.random.randn(5, 3)   # node features
W = np.random.randn(3, 4) * 0.1

H1 = gcn_layer(A, X, W)
print(f"After 1 GCN layer: {H1.shape}")  # (5, 4)</code></pre>

      <h3>3. PyTorch Geometric — GCN on Cora</h3>
      <pre><code">import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# Load Cora: 2708 papers, 5429 citations, 7 classes, 1433 features
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data    = dataset[0]
print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
print(f"Features: {data.num_node_features}, Classes: {dataset.num_classes}")
print(f"Train: {data.train_mask.sum()}, Val: {data.val_mask.sum()}, Test: {data.test_mask.sum()}")

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)

model     = GCN(dataset.num_features, 64, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out  = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward(); optimizer.step()
    return loss.item()

@torch.no_grad()
def test():
    model.eval()
    out   = model(data.x, data.edge_index).argmax(dim=1)
    accs  = [(out[m] == data.y[m]).float().mean().item() for m in [data.train_mask, data.val_mask, data.test_mask]]
    return accs

for epoch in range(1, 201):
    loss = train()
    if epoch % 50 == 0:
        tr, va, te = test()
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Train: {tr:.4f} | Val: {va:.4f} | Test: {te:.4f}")</code></pre>

      <h3>4. GCN Limitations</h3>
      <table>
        <thead><tr><th>Limitation</th><th>Description</th><th>Solution</th></tr></thead>
        <tbody>
          <tr><td>Transductive</td><td>Needs full graph at training time — can't generalize to new nodes</td><td>GraphSAGE (inductive)</td></tr>
          <tr><td>Oversmoothing</td><td>Many layers → all nodes converge to same representation</td><td>2-3 layers usually optimal</td></tr>
          <tr><td>Fixed graph</td><td>Can't handle dynamic graphs</td><td>Temporal GNNs</td></tr>
          <tr><td>Equal neighbor weights</td><td>All neighbors contribute equally</td><td>GAT (attention)</td></tr>
        </tbody>
      </table>
    `,
  },
  {
    title: "GraphSAGE: Inductive Learning on Unseen Nodes",
    description: "Hamilton et al.'s GraphSAGE — neighborhood sampling, mean/LSTM/pooling aggregators, inductive vs transductive learning, PyG SAGEConv, and mini-batch training with NeighborLoader.",
    date: "২৩ মে, ২০২৬",
    category: "Graph Neural Network",
    readTime: 11,
    slug: "gnn-graphsage",
    content: `
      <h3>1. Transductive vs Inductive</h3>
      <pre><code># Transductive (GCN): must see ALL nodes during training
#   → Can't classify new nodes added after training
#   → Requires full graph in memory
#
# Inductive (GraphSAGE): learns an AGGREGATION FUNCTION
#   → Can generate embeddings for UNSEEN nodes
#   → Works on large graphs via neighborhood sampling
#   → Generalizes across graphs (e.g., train on one graph, test on another)

# GraphSAGE update rule:
#   h_N(v) = AGGREGATE({h_u^(l) : u in N(v)})   sample K neighbors
#   h_v^(l+1) = sigma(W * CONCAT(h_v^(l), h_N(v)))
#   h_v^(l+1) = h_v^(l+1) / ||h_v^(l+1)||       L2 normalize</code></pre>

      <h3>2. Three Aggregators</h3>
      <table>
        <thead><tr><th>Aggregator</th><th>Formula</th><th>Properties</th></tr></thead>
        <tbody>
          <tr><td><strong>Mean</strong></td><td>Element-wise mean of neighbor vectors</td><td>Fast, inductive, good baseline</td></tr>
          <tr><td><strong>LSTM</strong></td><td>Apply LSTM on randomly permuted neighbors</td><td>Expressive but ignores permutation invariance</td></tr>
          <tr><td><strong>Pooling</strong></td><td>MLP on each neighbor → element-wise max/mean</td><td>Permutation invariant, captures individual features</td></tr>
        </tbody>
      </table>

      <h3>3. PyG — SAGEConv on Cora</h3>
      <pre><code">import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data    = dataset[0]

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_ch, hidden, out_ch):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, hidden, aggr='mean')
        self.conv2 = SAGEConv(hidden, out_ch, aggr='mean')

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)

model     = GraphSAGE(dataset.num_features, 64, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    model.train(); optimizer.zero_grad()
    out  = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward(); optimizer.step()

model.eval()
pred = model(data.x, data.edge_index).argmax(dim=1)
acc  = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
print(f"GraphSAGE Test Accuracy: {acc:.4f}")</code></pre>

      <h3>4. Mini-Batch Training with NeighborLoader</h3>
      <pre><code">from torch_geometric.loader import NeighborLoader

# For large graphs: sample 2-hop neighborhoods per batch
loader = NeighborLoader(
    data,
    num_neighbors=[15, 10],  # 15 neighbors at hop-1, 10 at hop-2
    batch_size=64,
    input_nodes=data.train_mask,
)

model.train()
for batch in loader:
    optimizer.zero_grad()
    out  = model(batch.x, batch.edge_index)
    # Only compute loss for seed nodes (first batch_size nodes)
    loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])
    loss.backward(); optimizer.step()
    print(f"Batch loss: {loss.item():.4f}", end='\r')</code></pre>

      <h3>5. GCN vs GraphSAGE</h3>
      <table>
        <thead><tr><th>Aspect</th><th>GCN</th><th>GraphSAGE</th></tr></thead>
        <tbody>
          <tr><td>Learning type</td><td>Transductive</td><td>Inductive</td></tr>
          <tr><td>Unseen nodes</td><td>Requires retraining</td><td>Handled naturally</td></tr>
          <tr><td>Large graphs</td><td>Full graph in memory</td><td>Neighborhood sampling</td></tr>
          <tr><td>Aggregation</td><td>Symmetric normalization</td><td>Flexible (mean/LSTM/pool)</td></tr>
          <tr><td>Normalization</td><td>Spectral (graph-dependent)</td><td>L2 normalization (flexible)</td></tr>
        </tbody>
      </table>
    `,
  },
  {
    title: "Graph Attention Networks (GAT): Learning Neighbor Importance",
    description: "Veličković et al.'s GAT — attention coefficients, multi-head attention, comparison with GCN and GraphSAGE, PyG GATConv, and attention weight visualization on Cora.",
    date: "২৩ মে, ২০২৬",
    category: "Graph Neural Network",
    readTime: 12,
    slug: "gnn-gat",
    content: `
      <h3>1. Why Attention?</h3>
      <p>GCN and GraphSAGE treat all neighbors with equal or degree-based weights. But in a citation network, some references are more relevant than others. GAT <strong>learns</strong> which neighbors to attend to.</p>
      <pre><code># GAT attention mechanism:
#
# Step 1: Linear transformation
#   h'_i = W * h_i   for all nodes i
#
# Step 2: Compute attention coefficient between i and neighbor j
#   e(i,j) = LeakyReLU(a^T [h'_i || h'_j])
#   || = concatenation, a = attention vector (learnable)
#
# Step 3: Normalize with softmax over all neighbors of i
#   alpha(i,j) = exp(e(i,j)) / sum_{k in N(i)} exp(e(i,k))
#   These are the attention weights: how much node j contributes to node i
#
# Step 4: Weighted aggregation
#   h'_i = sigma( sum_{j in N(i)} alpha(i,j) * W * h_j )
#
# Multi-head attention (K heads):
#   Concatenate: h'_i = ||_{k=1}^K sigma(sum_j alpha_k(i,j) * W_k * h_j)
#   Average:     h'_i = sigma( (1/K) * sum_k sum_j alpha_k(i,j) * W_k * h_j )
#   Concatenate in hidden layers, average in final layer</code></pre>

      <h3>2. PyG — GATConv</h3>
      <pre><code">import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data    = dataset[0]

class GAT(torch.nn.Module):
    def __init__(self, in_ch, hidden, out_ch, heads=8):
        super().__init__()
        # heads=8: 8 attention heads, concat → hidden*heads output channels
        self.conv1 = GATConv(in_ch, hidden, heads=heads, dropout=0.6)
        # Last layer: heads=1 with averaging
        self.conv2 = GATConv(hidden * heads, out_ch, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        return self.conv2(x, edge_index)

model     = GAT(dataset.num_features, 8, dataset.num_classes, heads=8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

for epoch in range(200):
    model.train(); optimizer.zero_grad()
    out  = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward(); optimizer.step()

model.eval()
pred = model(data.x, data.edge_index).argmax(dim=1)
acc  = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
print(f"GAT Test Accuracy: {acc:.4f}")</code></pre>

      <h3>3. Visualizing Attention Weights</h3>
      <pre><code"># Get attention weights for a node
class GATWithAttn(torch.nn.Module):
    def __init__(self, in_ch, hidden, out_ch):
        super().__init__()
        self.conv1 = GATConv(in_ch, hidden, heads=8, dropout=0.6, return_attention_weights=True)  # not a parameter
        self.conv2 = GATConv(hidden*8, out_ch, heads=1, concat=False)

    def forward(self, x, edge_index):
        # return_attention_weights via __call__ is done differently in PyG
        x, (edge_idx, alpha) = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        return self.conv2(x, edge_index), (edge_idx, alpha)

# alpha shape: (num_edges, num_heads)
# High alpha(i,j) → node i attends strongly to neighbor j</code></pre>

      <h3>4. Model Comparison on Cora</h3>
      <table>
        <thead><tr><th>Model</th><th>Test Accuracy</th><th>Key Mechanism</th><th>Parameters</th></tr></thead>
        <tbody>
          <tr><td>MLP (no graph)</td><td>~57%</td><td>Node features only</td><td>Low</td></tr>
          <tr><td>GCN</td><td>~81%</td><td>Symmetric normalization</td><td>Low</td></tr>
          <tr><td>GraphSAGE</td><td>~82%</td><td>Neighborhood sampling + aggregation</td><td>Medium</td></tr>
          <tr><td>GAT</td><td>~83%</td><td>Learned attention weights</td><td>Higher</td></tr>
        </tbody>
      </table>
    `,
  },
  {
    title: "GNN Applications: Node, Link & Graph-Level Tasks",
    description: "Three core GNN tasks with complete PyG code — node classification (recap), link prediction with dot-product decoder, and graph classification on the MUTAG molecular dataset using GIN.",
    date: "২৩ মে, ২০২৬",
    category: "Graph Neural Network",
    readTime: 11,
    slug: "gnn-applications",
    content: `
      <h3>1. Task Overview</h3>
      <table>
        <thead><tr><th>Level</th><th>Task</th><th>Real Application</th><th>Readout</th></tr></thead>
        <tbody>
          <tr><td>Node</td><td>Node classification</td><td>Paper topic, user churn, fraud detection</td><td>Node embedding → softmax</td></tr>
          <tr><td>Node</td><td>Node regression</td><td>Traffic speed, user engagement score</td><td>Node embedding → linear</td></tr>
          <tr><td>Edge</td><td>Link prediction</td><td>Friend recommendation, drug-target interaction</td><td>Pair of node embeddings → score</td></tr>
          <tr><td>Graph</td><td>Graph classification</td><td>Molecule toxicity, document category</td><td>Readout pool → softmax</td></tr>
          <tr><td>Graph</td><td>Graph regression</td><td>Molecule boiling point, binding affinity</td><td>Readout pool → linear</td></tr>
        </tbody>
      </table>

      <h3>2. Link Prediction</h3>
      <pre><code">import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling, train_test_split_edges

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data    = train_test_split_edges(dataset[0])

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_ch, hidden):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hidden)
        self.conv2 = GCNConv(hidden, hidden)

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

    def decode(self, z, edge_index):
        # Dot product between source and target node embeddings
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

    def forward(self, x, pos_edge, neg_edge):
        z = self.encode(x, data.train_pos_edge_index)
        pos_score = self.decode(z, pos_edge)
        neg_score = self.decode(z, neg_edge)
        return pos_score, neg_score

model     = LinkPredictor(dataset.num_features, 64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train(); optimizer.zero_grad()
    neg_edge    = negative_sampling(data.train_pos_edge_index, data.num_nodes)
    pos_s, neg_s = model(data.x, data.train_pos_edge_index, neg_edge)
    labels      = torch.cat([torch.ones(pos_s.size(0)), torch.zeros(neg_s.size(0))])
    scores      = torch.cat([pos_s, neg_s])
    loss        = F.binary_cross_entropy_with_logits(scores, labels)
    loss.backward(); optimizer.step()</code></pre>

      <h3>3. Graph Classification with GIN</h3>
      <pre><code">from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
import torch.nn as nn

# MUTAG: 188 molecular graphs, binary label (mutagenic or not)
dataset  = TUDataset(root='/tmp/MUTAG', name='MUTAG').shuffle()
train_ds = dataset[:150]; test_ds = dataset[150:]

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=32)

class GIN(nn.Module):
    def __init__(self, in_ch, hidden, out_ch):
        super().__init__()
        mlp1 = nn.Sequential(nn.Linear(in_ch, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        mlp2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.conv1 = GINConv(mlp1)
        self.conv2 = GINConv(mlp2)
        self.classifier = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, out_ch))

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)    # graph-level readout
        return self.classifier(x)

model     = GIN(dataset.num_features, 64, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(50):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        out  = model(batch.x.float(), batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward(); optimizer.step()

model.eval()
correct = sum((model(b.x.float(), b.edge_index, b.batch).argmax(1) == b.y).sum().item()
              for b in test_loader)
print(f"MUTAG Test Accuracy: {correct / len(test_ds):.4f}")</code></pre>

      <h3>4. Readout Functions</h3>
      <table>
        <thead><tr><th>Readout</th><th>Operation</th><th>Use Case</th></tr></thead>
        <tbody>
          <tr><td>global_mean_pool</td><td>Average of all node embeddings</td><td>Most common, stable</td></tr>
          <tr><td>global_max_pool</td><td>Element-wise max</td><td>Detect presence of a feature</td></tr>
          <tr><td>global_add_pool</td><td>Sum of all node embeddings</td><td>Size-sensitive tasks (GIN)</td></tr>
          <tr><td>DiffPool</td><td>Hierarchical learnable pooling</td><td>Capture multi-scale structure</td></tr>
        </tbody>
      </table>
    `,
  },
  {
    title: "GNN Project: MLP vs GCN vs GAT on Cora Citation Network",
    description: "Full end-to-end project on the Cora dataset — MLP baseline ignoring graph structure, 2-layer GCN, GAT with 8 heads, training with early stopping, t-SNE visualization, and graph structure impact analysis.",
    date: "২৩ মে, ২০২৬",
    category: "Graph Neural Network",
    readTime: 14,
    slug: "gnn-project-node-classification",
    content: `
      <h3>1. Dataset Overview</h3>
      <pre><code">from torch_geometric.datasets import Planetoid
import torch

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data    = dataset[0]

print("=== Cora Dataset ===")
print(f"Nodes:           {data.num_nodes}")
print(f"Edges:           {data.num_edges}")
print(f"Node features:   {data.num_node_features}  (bag-of-words)")
print(f"Classes:         {dataset.num_classes}   (paper topics)")
print(f"Train nodes:     {data.train_mask.sum().item()}")
print(f"Val nodes:       {data.val_mask.sum().item()}")
print(f"Test nodes:      {data.test_mask.sum().item()}")</code></pre>
      <table>
        <thead><tr><th>Property</th><th>Value</th></tr></thead>
        <tbody>
          <tr><td>Nodes (papers)</td><td>2,708</td></tr>
          <tr><td>Edges (citations)</td><td>5,429</td></tr>
          <tr><td>Features per node</td><td>1,433 (bag-of-words)</td></tr>
          <tr><td>Classes</td><td>7 (ML sub-topics)</td></tr>
          <tr><td>Train / Val / Test</td><td>140 / 500 / 1,000 nodes</td></tr>
        </tbody>
      </table>

      <h3>2. Three Models</h3>
      <pre><code">import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

# Model 1: MLP — ignores graph structure
class MLP(nn.Module):
    def __init__(self, in_ch, hidden, out_ch):
        super().__init__()
        self.fc1 = nn.Linear(in_ch, hidden)
        self.fc2 = nn.Linear(hidden, out_ch)
    def forward(self, x, edge_index=None):
        return self.fc2(F.dropout(F.relu(self.fc1(x)), p=0.5, training=self.training))

# Model 2: GCN — uses graph structure
class GCN(nn.Module):
    def __init__(self, in_ch, hidden, out_ch):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hidden)
        self.conv2 = GCNConv(hidden, out_ch)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)

# Model 3: GAT — attention over neighbors
class GAT(nn.Module):
    def __init__(self, in_ch, hidden, out_ch, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_ch, hidden, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden * heads, out_ch, heads=1, concat=False, dropout=0.6)
    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        return self.conv2(x, edge_index)</code></pre>

      <h3>3. Training Loop with Early Stopping</h3>
      <pre><code">def train_model(model, data, lr=0.005, weight_decay=5e-4, epochs=300, patience=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_state, wait = 0, None, 0

    for epoch in range(epochs):
        model.train(); optimizer.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            pred    = out.argmax(dim=1)
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()

        if val_acc > best_val:
            best_val, best_state, wait = val_acc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience: break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred     = model(data.x, data.edge_index).argmax(dim=1)
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    return test_acc

results = {}
for name, ModelClass, kwargs in [
    ('MLP',  MLP,  {}),
    ('GCN',  GCN,  {}),
    ('GAT',  GAT,  {'heads': 8}),
]:
    m   = ModelClass(dataset.num_features, 64, dataset.num_classes, **kwargs)
    acc = train_model(m, data)
    results[name] = acc
    print(f"{name}: Test Accuracy = {acc:.4f}")</code></pre>

      <h3>4. t-SNE Visualization of Embeddings</h3>
      <pre><code">from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Get node embeddings from GCN (before final classification layer)
class GCNEncoder(nn.Module):
    def __init__(self, in_ch, hidden):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hidden)
        self.conv2 = GCNConv(hidden, hidden)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

encoder = GCNEncoder(dataset.num_features, 64)
# (train encoder separately or extract from trained GCN)

gcn_model = GCN(dataset.num_features, 64, dataset.num_classes)
train_model(gcn_model, data)

gcn_model.eval()
with torch.no_grad():
    # Extract penultimate layer embeddings
    h = F.relu(gcn_model.conv1(data.x, data.edge_index))

tsne   = TSNE(n_components=2, random_state=42, perplexity=30)
embed  = tsne.fit_transform(h.numpy())
labels = data.y.numpy()
colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#a65628','#f781bf']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (emb, title) in zip(axes, [(data.x.numpy(), 'Raw Features'), (embed, 'GCN Embeddings')]):
    if emb.shape[1] > 2:
        emb = TSNE(n_components=2, random_state=42).fit_transform(emb)
    for c in range(7):
        mask = labels == c
        ax.scatter(emb[mask, 0], emb[mask, 1], c=colors[c], s=5, alpha=0.7, label=f'Class {c}')
    ax.set_title(title); ax.legend(markerscale=3, fontsize=7)
plt.tight_layout(); plt.show()</code></pre>

      <h3>5. Results Summary</h3>
      <table>
        <thead><tr><th>Model</th><th>Uses Graph?</th><th>Test Accuracy</th><th>Key Insight</th></tr></thead>
        <tbody>
          <tr><td>MLP</td><td>No</td><td>~57%</td><td>Node features alone are insufficient</td></tr>
          <tr><td>GCN</td><td>Yes</td><td>~81%</td><td>Graph structure adds ~24% accuracy</td></tr>
          <tr><td>GAT</td><td>Yes + attention</td><td>~83%</td><td>Attention over GCN gives marginal gain</td></tr>
        </tbody>
      </table>
      <p>The jump from MLP to GCN (~57% → ~81%) demonstrates that <strong>graph structure carries crucial information</strong> beyond node features alone. The citation relationships between papers encode topic similarity that the bag-of-words features miss.</p>
    `,
  },
];
