export const gnn_5_applications = {
  title: "GNN-এর তিনটি প্রধান কাজ: Node, Link ও Graph",
  description: "Node classification, link prediction এবং graph classification — তিনটি প্রধান GNN কাজ, PyG কোড, এবং MUTAG molecular dataset-এ GIN দিয়ে graph classification।",
  date: "২৩ মে, ২০২৬",
  category: "গ্রাফ নিউরাল নেটওয়ার্ক",
  readTime: 11,
  slug: "gnn-applications",
  content: `
    <h3>১. তিনটি প্রধান GNN কাজের সারসংক্ষেপ</h3>
    <p>
      GNN দিয়ে তিন স্তরে prediction করা যায়: নোড স্তরে, এজ স্তরে, এবং সম্পূর্ণ গ্রাফ স্তরে।
      প্রতিটি কাজের জন্য আলাদা আর্কিটেকচার কৌশল দরকার।
    </p>
    <table>
      <thead>
        <tr><th>কাজের ধরন</th><th>ইনপুট</th><th>আউটপুট</th><th>মূল্যায়ন</th><th>বাস্তব উদাহরণ</th></tr>
      </thead>
      <tbody>
        <tr>
          <td>Node Classification</td>
          <td>গ্রাফ + কিছু labeled নোড</td>
          <td>প্রতিটি নোডের class</td>
          <td>Accuracy, F1</td>
          <td>Cora পেপার বিষয়, ফেক অ্যাকাউন্ট সনাক্ত</td>
        </tr>
        <tr>
          <td>Link Prediction</td>
          <td>গ্রাফ + নোড embedding</td>
          <td>এজ থাকবে কি? (0/1)</td>
          <td>AUC-ROC, AP</td>
          <td>বন্ধু সুপারিশ, drug interaction</td>
        </tr>
        <tr>
          <td>Graph Classification</td>
          <td>একটি গ্রাফ</td>
          <td>পুরো গ্রাফের class</td>
          <td>Accuracy</td>
          <td>অণুর বিষাক্ততা, প্রোটিন ফাংশন</td>
        </tr>
      </tbody>
    </table>

    <h3>২. Node Classification: Semi-supervised শেখা</h3>
    <p>
      Node classification-এ মাত্র কিছু নোডের লেবেল জানা (labeled), বাকিগুলো unlabeled।
      GNN এই কিছু labeled নোড থেকে শিখে বাকিগুলোর লেবেল predict করে।
    </p>
    <p>
      এটাই <strong>semi-supervised learning</strong>: graph structure ব্যবহার করে labeled থেকে
      unlabeled-এ জ্ঞান "প্রবাহিত" করা হয়।
    </p>
    <p>
      Cora-তে মাত্র 140টি training node (7 ক্লাস × 20 প্রতি ক্লাস) — মোট 2708 নোডের মাত্র 5%।
      তবু GCN ~81% accuracy দেয় কারণ citation structure থেকে প্রচুর তথ্য পায়।
    </p>
    <p>আমরা ইতিমধ্যে GCN ও GAT দিয়ে এটি করেছি। এখন link prediction ও graph classification দেখি।</p>

    <h3>৩. Link Prediction: দুই নোড কি সংযুক্ত হবে?</h3>
    <p>
      Link prediction-এ প্রশ্ন: দুটি নোড u এবং v-এর মধ্যে এজ থাকবে কিনা?
      উদাহরণ: ফেসবুকে "হয়তো চেনো" সুপারিশ, knowledge graph-এ missing relation পূরণ।
    </p>
    <p><strong>পদ্ধতি:</strong></p>
    <p>
      GNN encoder দিয়ে প্রতিটি নোডের embedding তৈরি করো।
      তারপর দুটি নোডের embedding-এর similarity score বের করো।
    </p>
    <pre><code>import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import negative_sampling, train_test_split_edges

# Dataset লোড ও edge split
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Training/validation/test edge split
data = train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1)

print(f"Train edges: {data.train_pos_edge_index.shape[1]}")
print(f"Val positive edges: {data.val_pos_edge_index.shape[1]}")
print(f"Test positive edges: {data.test_pos_edge_index.shape[1]}")

# GCN Encoder: নোড embedding তৈরি করে
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x   # shape: (num_nodes, out_channels)

# Dot Product Decoder: দুই নোডের similarity
def decode(z, edge_index):
    # dot product: sigmoid(z_i · z_j)
    src = z[edge_index[0]]
    dst = z[edge_index[1]]
    return torch.sigmoid((src * dst).sum(dim=1))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCNEncoder(dataset.num_node_features, 64).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train_link():
    model.train()
    optimizer.zero_grad()
    z = model(data.x, data.train_pos_edge_index)

    # Positive edges
    pos_pred = decode(z, data.train_pos_edge_index)
    pos_labels = torch.ones(pos_pred.size(0), device=device)

    # Negative sampling: এজ নেই এমন pair
    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1)
    )
    neg_pred = decode(z, neg_edge_index)
    neg_labels = torch.zeros(neg_pred.size(0), device=device)

    # Binary cross-entropy loss
    preds = torch.cat([pos_pred, neg_pred])
    labels = torch.cat([pos_labels, neg_labels])
    loss = F.binary_cross_entropy(preds, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

from sklearn.metrics import roc_auc_score

def test_link():
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.train_pos_edge_index)

    # Test positive
    pos_pred = decode(z, data.test_pos_edge_index).cpu()
    # Test negative
    neg_pred = decode(z, data.test_neg_edge_index).cpu()

    y_true = torch.cat([torch.ones(pos_pred.size(0)),
                        torch.zeros(neg_pred.size(0))]).numpy()
    y_pred = torch.cat([pos_pred, neg_pred]).numpy()
    return roc_auc_score(y_true, y_pred)

for epoch in range(1, 101):
    loss = train_link()
    if epoch % 20 == 0:
        auc = test_link()
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Test AUC: {auc:.4f}")</code></pre>

    <h3>৪. Graph Classification: পুরো গ্রাফকে ক্লাসিফাই করো</h3>
    <p>
      Graph classification-এ একটি সম্পূর্ণ গ্রাফকে একটি class দিতে হয়।
      চ্যালেঞ্জ: বিভিন্ন গ্রাফে নোড সংখ্যা ভিন্ন। তাই সব নোডের embedding একটি
      fixed-size গ্রাফ embedding-এ পরিণত করতে হবে — এটাকে বলে <strong>graph readout/pooling</strong>।
    </p>
    <p><strong>Global Pooling পদ্ধতি:</strong></p>
    <pre><code># Global Mean Pool: সব নোডের embedding-এর গড়
h_G = (1/n) Σ h_v

# Global Max Pool: element-wise maximum
h_G = max({h_v : v ∈ V})

# Global Sum Pool: sum (size-sensitive)
h_G = Σ h_v</code></pre>
    <p>
      <strong>GIN (Graph Isomorphism Network):</strong> Xu et al. (2019) প্রমাণ করেছেন GIN
      গ্রাফের গঠন আলাদা করতে GCN-এর চেয়ে বেশি powerful। GIN-এর update rule:
    </p>
    <pre><code>h_v^(l+1) = MLP((1 + ε) · h_v^(l) + Σ_{u ∈ N(v)} h_u^(l))</code></pre>

    <h3>৫. GIN দিয়ে MUTAG Graph Classification</h3>
    <pre><code>import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

# MUTAG: 188টি molecular graph, 2 ক্লাস (toxic / non-toxic)
dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
print(f"মোট গ্রাফ: {len(dataset)}")
print(f"ক্লাস সংখ্যা: {dataset.num_classes}")
print(f"নোড ফিচার: {dataset.num_node_features}")

# Train/test split
torch.manual_seed(42)
dataset = dataset.shuffle()
train_dataset = dataset[:150]
test_dataset = dataset[150:]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# GIN মডেল
class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GIN, self).__init__()

        # GINConv: MLP দিয়ে aggregation
        mlp1 = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        mlp2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )

        self.conv1 = GINConv(mlp1, train_eps=True)
        self.conv2 = GINConv(mlp2, train_eps=True)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # Node-level embedding
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Graph-level readout: global mean pooling
        x = global_mean_pool(x, batch)   # batch: কোন নোড কোন গ্রাফের

        # Classification
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GIN(
    in_channels=dataset.num_node_features,
    hidden_channels=64,
    out_channels=dataset.num_classes
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train_graph():
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.nll_loss(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test_graph(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    return correct / total

for epoch in range(1, 101):
    loss = train_graph()
    if epoch % 20 == 0:
        train_acc = test_graph(train_loader)
        test_acc = test_graph(test_loader)
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | "
              f"Train: {train_acc:.4f} | Test: {test_acc:.4f}")</code></pre>

    <h3>৬. তিনটি কাজের চূড়ান্ত তুলনা</h3>
    <table>
      <thead>
        <tr><th>কাজ</th><th>Readout দরকার?</th><th>Loss</th><th>মূল্যায়ন মেট্রিক</th><th>বাস্তব প্রয়োগ</th></tr>
      </thead>
      <tbody>
        <tr>
          <td>Node Classification</td>
          <td>না (নোড-স্তরে prediction)</td>
          <td>Cross-entropy</td>
          <td>Accuracy, Macro-F1</td>
          <td>কমিউনিটি ডিটেকশন, ফেক অ্যাকাউন্ট</td>
        </tr>
        <tr>
          <td>Link Prediction</td>
          <td>না (এজ-স্তরে)</td>
          <td>Binary cross-entropy</td>
          <td>AUC-ROC, Average Precision</td>
          <td>বন্ধু সুপারিশ, knowledge graph</td>
        </tr>
        <tr>
          <td>Graph Classification</td>
          <td>হ্যাঁ (global pooling)</td>
          <td>Cross-entropy</td>
          <td>Accuracy</td>
          <td>ড্রাগ আবিষ্কার, প্রোটিন ফাংশন</td>
        </tr>
      </tbody>
    </table>
    <p>
      পরের ব্লগে আমরা একটি সম্পূর্ণ end-to-end project করব: Cora dataset-এ MLP, GCN এবং GAT
      তুলনা করব এবং t-SNE দিয়ে embedding ভিজুয়ালাইজ করব।
    </p>
  `,
};
