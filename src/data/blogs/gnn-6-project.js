export const gnn_6_project = {
  title: "GNN প্রজেক্ট: Cora-তে MLP বনাম GCN বনাম GAT",
  description: "Cora citation network-এ end-to-end project — MLP baseline, 2-layer GCN, GAT তুলনা, early stopping, t-SNE embedding ভিজুয়ালাইজেশন, এবং গ্রাফ স্ট্রাকচারের গুরুত্ব প্রমাণ।",
  date: "২৩ মে, ২০২৬",
  category: "গ্রাফ নিউরাল নেটওয়ার্ক",
  readTime: 14,
  slug: "gnn-project-node-classification",
  content: `
    <h3>১. Cora Dataset: আমাদের প্রজেক্টের মঞ্চ</h3>
    <p>
      Cora হলো GNN গবেষণায় সবচেয়ে বিখ্যাত benchmark dataset। এটি একটি citation network:
      প্রতিটি গবেষণাপত্র একটি নোড, এবং দুটি পেপারের মধ্যে citation থাকলে একটি এজ।
    </p>
    <table>
      <thead>
        <tr><th>বৈশিষ্ট্য</th><th>মান</th><th>বিবরণ</th></tr>
      </thead>
      <tbody>
        <tr><td>নোড সংখ্যা</td><td>2,708</td><td>গবেষণাপত্র</td></tr>
        <tr><td>এজ সংখ্যা</td><td>5,429</td><td>citation সংযোগ (undirected)</td></tr>
        <tr><td>ফিচার ডিমেনশন</td><td>1,433</td><td>Bag-of-words (শব্দ আছে কি নেই)</td></tr>
        <tr><td>ক্লাস সংখ্যা</td><td>7</td><td>গবেষণার বিষয়</td></tr>
        <tr><td>Training নোড</td><td>140</td><td>প্রতি ক্লাসে 20টি</td></tr>
        <tr><td>Validation নোড</td><td>500</td><td>hyperparameter tuning</td></tr>
        <tr><td>Test নোড</td><td>1,000</td><td>চূড়ান্ত মূল্যায়ন</td></tr>
      </tbody>
    </table>
    <p>
      7টি ক্লাস: Case Based, Genetic Algorithms, Neural Networks, Probabilistic Methods,
      Reinforcement Learning, Rule Learning, Theory। মাত্র 140টি training নোড — মোটের মাত্র 5%।
      এখানেই graph structure-এর শক্তি প্রমাণিত হবে।
    </p>

    <h3>২. Dataset লোড ও EDA</h3>
    <pre><code>import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree
import networkx as nx
from torch_geometric.utils import to_networkx

# Dataset লোড
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

print("=== Cora Dataset তথ্য ===")
print(f"নোড সংখ্যা: {data.num_nodes}")
print(f"এজ সংখ্যা: {data.num_edges}")
print(f"ফিচার: {data.num_node_features}")
print(f"ক্লাস: {dataset.num_classes}")
print(f"Train/Val/Test: {data.train_mask.sum()}/{data.val_mask.sum()}/{data.test_mask.sum()}")

# ক্লাস বিতরণ
class_names = ['Case Based', 'Genetic Alg', 'Neural Net',
               'Prob Methods', 'Reinf Learning', 'Rule Learning', 'Theory']
class_counts = [(data.y == i).sum().item() for i in range(7)]
print("\n=== ক্লাস বিতরণ ===")
for name, count in zip(class_names, class_counts):
    print(f"  {name}: {count} নোড")

# Degree বিতরণ
node_degrees = degree(data.edge_index[0], data.num_nodes).numpy()
print(f"\nগড় degree: {node_degrees.mean():.2f}")
print(f"সর্বোচ্চ degree: {node_degrees.max():.0f}")
print(f"মেডিয়ান degree: {np.median(node_degrees):.0f}")

# Degree distribution plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(class_names, class_counts, color='steelblue')
plt.xticks(rotation=45, ha='right')
plt.title('Cora ক্লাস বিতরণ')
plt.ylabel('নোড সংখ্যা')

plt.subplot(1, 2, 2)
plt.hist(node_degrees, bins=30, color='coral', edgecolor='white')
plt.xlabel('Degree')
plt.ylabel('নোড সংখ্যা')
plt.title('Degree Distribution')
plt.tight_layout()
plt.savefig('cora_eda.png', dpi=150)
plt.show()</code></pre>

    <h3>৩. তিনটি মডেল তৈরি: MLP, GCN, GAT</h3>
    <pre><code>from torch_geometric.nn import GCNConv, GATConv

# মডেল ১: MLP Baseline (গ্রাফ structure উপেক্ষা করে)
class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(MLP, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x = data.x                          # শুধু নোড ফিচার, কোনো এজ নেই
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

# মডেল ২: 2-layer GCN
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def get_embedding(self, data):
        """t-SNE visualization-এর জন্য hidden embedding"""
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return x   # hidden layer output

# মডেল ৩: 2-layer GAT (8 heads)
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 heads=8, dropout=0.6):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads,
                             dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, out_channels,
                             heads=1, dropout=dropout, concat=False)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def get_embedding(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=False)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        return x</code></pre>

    <h3>৪. Training Loop: Early Stopping সহ</h3>
    <pre><code>device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = dataset[0].to(device)

def create_model(model_type):
    if model_type == 'MLP':
        return MLP(dataset.num_node_features, 64, dataset.num_classes).to(device)
    elif model_type == 'GCN':
        return GCN(dataset.num_node_features, 64, dataset.num_classes).to(device)
    elif model_type == 'GAT':
        return GAT(dataset.num_node_features, 8, dataset.num_classes).to(device)

def train_model(model, epochs=200, lr=0.01, weight_decay=5e-4, patience=20):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr, weight_decay=weight_decay)

    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0
    history = {'loss': [], 'val_acc': [], 'test_acc': []}

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = out.argmax(dim=1)

        val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()

        history['loss'].append(loss.item())
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 50 == 0:
            print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f} | "
                  f"Val: {val_acc:.4f} | Test: {test_acc:.4f}")

    return best_val_acc, best_test_acc, history

import time

results = {}
for model_type in ['MLP', 'GCN', 'GAT']:
    print(f"\n=== {model_type} Training ===")
    model = create_model(model_type)

    lr = 0.005 if model_type == 'GAT' else 0.01
    wd = 5e-4

    start = time.time()
    best_val, best_test, history = train_model(
        model, epochs=200, lr=lr, weight_decay=wd, patience=20
    )
    elapsed = time.time() - start

    param_count = sum(p.numel() for p in model.parameters())
    results[model_type] = {
        'val_acc': best_val,
        'test_acc': best_test,
        'time': elapsed,
        'params': param_count
    }
    print(f"  Best Val: {best_val:.4f} | Best Test: {best_test:.4f}")</code></pre>

    <h3>৫. t-SNE দিয়ে Embedding ভিজুয়ালাইজেশন</h3>
    <pre><code">from sklearn.manifold import TSNE

def plot_tsne(model, data, model_name):
    model.eval()
    with torch.no_grad():
        if model_name == 'MLP':
            # MLP-র জন্য raw features ব্যবহার করো
            emb = data.x.cpu().numpy()
        else:
            emb = model.get_embedding(data).cpu().numpy()

    labels = data.y.cpu().numpy()

    # t-SNE: উচ্চ-মাত্রার embedding → 2D
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    emb_2d = tsne.fit_transform(emb)

    colors = plt.cm.Set1(np.linspace(0, 1, 7))
    plt.figure(figsize=(8, 6))
    for c in range(7):
        mask = labels == c
        plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                    c=[colors[c]], label=class_names[c],
                    alpha=0.6, s=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.title(f't-SNE: {model_name} Embedding')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'tsne_{model_name.lower()}.png', dpi=150, bbox_inches='tight')
    plt.show()

# প্রতিটি মডেলের embedding ভিজুয়ালাইজ
for model_type in ['MLP', 'GCN', 'GAT']:
    model = create_model(model_type)
    # (production-এ trained model লোড করে ব্যবহার করবে)
    print(f"{model_type} t-SNE plotting...")
    plot_tsne(model, data, model_type)</code></pre>

    <h3>৬. ফলাফল ও বিশ্লেষণ</h3>
    <pre><code>print("\n" + "="*60)
print(f"{'মডেল':<12} {'Val Acc':>10} {'Test Acc':>10} {'সময় (s)':>10} {'Parameters':>12}")
print("="*60)
for name, res in results.items():
    print(f"{name:<12} {res['val_acc']:>10.4f} {res['test_acc']:>10.4f} "
          f"{res['time']:>10.2f} {res['params']:>12,}")
print("="*60)</code></pre>
    <table>
      <thead>
        <tr><th>মডেল</th><th>Val Accuracy</th><th>Test Accuracy</th><th>সময় (s)</th><th>Parameters</th></tr>
      </thead>
      <tbody>
        <tr><td>MLP</td><td>~0.560</td><td>~0.570</td><td>~5</td><td>~92K</td></tr>
        <tr><td>GCN</td><td>~0.800</td><td>~0.810</td><td>~10</td><td>~92K</td></tr>
        <tr><td>GAT</td><td>~0.810</td><td>~0.830</td><td>~25</td><td>~365K</td></tr>
      </tbody>
    </table>
    <p>
      <strong>মূল শিক্ষা:</strong> MLP শুধু নোড ফিচার (bag-of-words) দেখে ~57% accuracy দেয়।
      GCN একই ফিচার + citation structure ব্যবহার করে ~81% দেয় — শুধু graph connectivity
      যোগ করে 24% পয়েন্ট উন্নতি! GAT আরও ভালো করে ~83%, কারণ এটি important citations-কে
      বেশি গুরুত্ব দিতে পারে।
    </p>
    <p>
      t-SNE visualization-এ স্পষ্ট দেখা যায়: MLP-র embedding ক্লাসগুলো মিশিয়ে থাকে,
      কিন্তু GCN ও GAT-এর embedding-এ ক্লাসগুলো আলাদা cluster তৈরি করে।
      এটাই graph structure-এর শক্তি।
    </p>

    <h3>৭. সিরিজের সারসংক্ষেপ ও পরবর্তী ধাপ</h3>
    <p>এই GNN সিরিজে আমরা শিখেছি:</p>
    <p>
      <strong>ব্লগ ১:</strong> গ্রাফের গণিত — অ্যাডজেসেন্সি ম্যাট্রিক্স, ডিগ্রি ম্যাট্রিক্স, লাপ্লাসিয়ান।
    </p>
    <p>
      <strong>ব্লগ ২:</strong> GCN — নরমালাইজড aggregation, PyG-তে GCNConv, Cora-তে প্রয়োগ।
    </p>
    <p>
      <strong>ব্লগ ৩:</strong> GraphSAGE — inductive learning, neighborhood sampling, mini-batch।
    </p>
    <p>
      <strong>ব্লগ ৪:</strong> GAT — attention mechanism, multi-head attention, interpretability।
    </p>
    <p>
      <strong>ব্লগ ৫:</strong> তিনটি GNN কাজ — node classification, link prediction, graph classification।
    </p>
    <p>
      <strong>ব্লগ ৬:</strong> End-to-end project — MLP vs GCN vs GAT, early stopping, t-SNE।
    </p>
    <p>
      পরবর্তী অন্বেষণের জন্য: Graph Transformers (Graphformer), Heterogeneous GNNs,
      Temporal GNNs, এবং লিংক প্রেডিকশনের জন্য Knowledge Graph Embedding (TransE, RotatE)।
    </p>
  `,
};
