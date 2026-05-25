export const gnn_2_gcn = {
  title: "গ্রাফ কনভোলিউশনাল নেটওয়ার্ক (GCN): প্রতিবেশী থেকে শেখা",
  description: "Kipf ও Welling-এর GCN: স্পেকট্রাল থেকে স্প্যাশিয়াল পদ্ধতি, নরমালাইজড অ্যাডজেসেন্সি, numpy থেকে PyTorch Geometric পর্যন্ত বাস্তব কোড উদাহরণ।",
  date: "২৩ মে, ২০২৬",
  category: "গ্রাফ নিউরাল নেটওয়ার্ক",
  readTime: 13,
  slug: "gnn-gcn",
  content: `
    <h3>১. GCN-এর মূল স্বজ্ঞা: প্রতিবেশীর কাছ থেকে শেখো</h3>
    <p>
      মনে করো তুমি একটি বিশ্ববিদ্যালয়ের নতুন ছাত্র। তুমি কোন বিষয়ে পড়ো তা বোঝার সহজ উপায় হলো —
      তোমার বন্ধুরা (প্রতিবেশী নোড) কোন বিষয়ে পড়ে। গবেষণাপত্রের সাইটেশন নেটওয়ার্কেও একই ব্যাপার:
      একটি Machine Learning পেপার সাধারণত অন্য ML পেপারই cite করে।
    </p>
    <p>
      <strong>Graph Convolutional Networks (GCN)</strong> — Kipf ও Welling (2017) — ঠিক এই ধারণাটি
      ব্যবহার করে। প্রতিটি নোড তার নিজের ফিচার এবং প্রতিবেশীদের ফিচার একত্রিত (aggregate) করে
      একটি নতুন representation তৈরি করে।
    </p>
    <p>
      প্রথমে <strong>স্পেকট্রাল পদ্ধতি</strong> সম্পর্কে সংক্ষেপে বলি: GCN-এর তাত্ত্বিক ভিত্তি
      গ্রাফ সিগন্যাল প্রসেসিং থেকে এসেছে। লাপ্লাসিয়ান ম্যাট্রিক্সের eigendecomposition ব্যবহার করে
      গ্রাফে ফ্রিকোয়েন্সি ডোমেইনে ফিল্টারিং করা যায়। কিন্তু এটি computationally ব্যয়বহুল।
      Kipf ও Welling এটি সরলীকরণ করে একটি স্প্যাশিয়াল সূত্রে পরিণত করেছেন।
    </p>

    <h3>২. GCN সূত্র: ধাপে ধাপে বোঝো</h3>
    <p>GCN-এর layer-wise propagation rule:</p>
    <pre><code>H^(l+1) = σ( D̃^(-1/2) · Ã · D̃^(-1/2) · H^(l) · W^(l) )

যেখানে:
  Ã = A + I         (self-loop যোগ করা: প্রতিটি নোড নিজেও প্রতিবেশী)
  D̃ = Ã-এর degree matrix (diagonal)
  H^(l) = l-তম layer-এর নোড এমবেডিং (প্রথমে H^(0) = X, ফিচার ম্যাট্রিক্স)
  W^(l) = learnable weight matrix
  σ = activation function (ReLU)</code></pre>
    <p>
      <strong>ধাপ ১ — Self-loop যোগ:</strong> Ã = A + I। প্রতিটি নোড নিজের ফিচারও aggregation-এ রাখে।
    </p>
    <p>
      <strong>ধাপ ২ — Symmetric normalization:</strong> D̃^(-1/2) Ã D̃^(-1/2)। এটি ডিগ্রির
      পার্থক্যের কারণে যে সমস্যা হয় তা সমাধান করে — বড় ডিগ্রির নোড ছোট ডিগ্রির নোডকে dominate করবে না।
    </p>
    <p>
      <strong>ধাপ ৩ — Feature transformation:</strong> নরমালাইজড অ্যাডজেসেন্সি × H^(l) × W^(l)।
      প্রতিবেশীদের ফিচার গড় করে weight matrix দিয়ে transform করা।
    </p>
    <p>
      <strong>ধাপ ৪ — Activation:</strong> ReLU বা softmax দিয়ে nonlinearity।
    </p>

    <h3>৩. NumPy দিয়ে GCN: শূন্য থেকে তৈরি</h3>
    <pre><code>import numpy as np

# ছোট গ্রাফ: 4টি নোড
# এজ: 0-1, 1-2, 2-3, 0-3
A = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
], dtype=float)

# নোড ফিচার (4 নোড, 3 ফিচার)
X = np.array([
    [1.0, 0.0, 0.5],
    [0.0, 1.0, 0.2],
    [0.5, 0.5, 0.8],
    [0.2, 0.3, 1.0]
])

# ধাপ ১: Self-loop যোগ করো
I = np.eye(A.shape[0])
A_tilde = A + I
print("Ã (A + I):")
print(A_tilde)

# ধাপ ২: Degree matrix of Ã
d_tilde = A_tilde.sum(axis=1)
D_tilde_inv_sqrt = np.diag(1.0 / np.sqrt(d_tilde))
print("\nD̃^(-1/2):")
print(np.round(D_tilde_inv_sqrt, 3))

# ধাপ ৩: Normalized adjacency
A_norm = D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt
print("\nনরমালাইজড A:")
print(np.round(A_norm, 3))

# ধাপ ৪: Weight matrix (randomly initialized)
np.random.seed(42)
W1 = np.random.randn(3, 4) * 0.1   # 3 input features → 4 hidden
W2 = np.random.randn(4, 2) * 0.1   # 4 hidden → 2 output classes

# ধাপ ৫: Forward pass (2-layer GCN)
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - x.max(axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

# Layer 1
H1 = relu(A_norm @ X @ W1)
print("\nLayer 1 output H1:")
print(np.round(H1, 3))

# Layer 2
H2 = softmax(A_norm @ H1 @ W2)
print("\nLayer 2 output (class probabilities):")
print(np.round(H2, 3))</code></pre>

    <h3>৪. PyTorch Geometric দিয়ে GCN: Cora Dataset</h3>
    <p>
      বাস্তব কাজে আমরা PyTorch Geometric (PyG) ব্যবহার করি। Cora হলো একটি বিখ্যাত citation network
      dataset — 2708টি গবেষণাপত্র (নোড), 5429টি citation (এজ), 7টি ক্লাস।
    </p>
    <pre><code>import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

# Dataset লোড করো
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

print(f"নোড সংখ্যা: {data.num_nodes}")
print(f"এজ সংখ্যা: {data.num_edges}")
print(f"ফিচার ডিমেনশন: {data.num_node_features}")
print(f"ক্লাস সংখ্যা: {dataset.num_classes}")
print(f"Training নোড: {data.train_mask.sum().item()}")
print(f"Validation নোড: {data.val_mask.sum().item()}")
print(f"Test নোড: {data.test_mask.sum().item()}")

# GCN মডেল তৈরি
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Layer 1: GCN + ReLU + Dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2: GCN (output layer)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# মডেল, optimizer সেটআপ
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(
    num_features=dataset.num_node_features,
    hidden_dim=64,
    num_classes=dataset.num_classes,
    dropout=0.5
).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training loop
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(mask):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        correct = (pred[mask] == data.y[mask]).sum()
        acc = correct / mask.sum()
    return acc.item()

# ট্রেনিং চালাও
for epoch in range(1, 201):
    loss = train()
    if epoch % 20 == 0:
        val_acc = evaluate(data.val_mask)
        test_acc = evaluate(data.test_mask)
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}")

print(f"\nFinal Test Accuracy: {evaluate(data.test_mask):.4f}")</code></pre>

    <h3>৫. GCN-এর সীমাবদ্ধতা</h3>
    <p>GCN চমৎকার হলেও কিছু গুরুত্বপূর্ণ সীমাবদ্ধতা আছে:</p>
    <table>
      <thead>
        <tr><th>সীমাবদ্ধতা</th><th>কারণ</th><th>সমাধান</th></tr>
      </thead>
      <tbody>
        <tr>
          <td>Transductive শুধু</td>
          <td>পুরো গ্রাফ একবারে দরকার, নতুন নোড handle করতে পারে না</td>
          <td>GraphSAGE (পরের ব্লগ)</td>
        </tr>
        <tr>
          <td>Oversmoothing</td>
          <td>অনেক layer হলে সব নোডের embedding একই হয়ে যায়</td>
          <td>2-3 layer রাখো, residual connection</td>
        </tr>
        <tr>
          <td>সমান গুরুত্ব</td>
          <td>সব প্রতিবেশীকে সমান গুরুত্ব দেয়</td>
          <td>Graph Attention Network (GAT)</td>
        </tr>
        <tr>
          <td>Memory সমস্যা</td>
          <td>বড় গ্রাফে পুরো A ম্যাট্রিক্স মেমোরিতে রাখা কঠিন</td>
          <td>Mini-batch + NeighborLoader</td>
        </tr>
      </tbody>
    </table>
    <table>
      <thead>
        <tr><th>Hyperparameter</th><th>সাধারণ মান</th><th>প্রভাব</th></tr>
      </thead>
      <tbody>
        <tr><td>num_layers</td><td>2-3</td><td>বেশি হলে oversmoothing</td></tr>
        <tr><td>hidden_dim</td><td>64-256</td><td>বড় হলে বেশি expressive কিন্তু overfit</td></tr>
        <tr><td>dropout</td><td>0.5</td><td>regularization, overfitting কমায়</td></tr>
        <tr><td>learning_rate</td><td>0.01</td><td>খুব বড় হলে অস্থির, খুব ছোট হলে ধীর</td></tr>
        <tr><td>weight_decay</td><td>5e-4</td><td>L2 regularization</td></tr>
      </tbody>
    </table>

    <h3>৬. সারসংক্ষেপ</h3>
    <p>
      GCN গ্রাফ ডেটায় deep learning প্রয়োগের একটি মার্জিত সমাধান। মূল ধারণা সহজ: প্রতিটি নোড
      তার প্রতিবেশীদের ফিচার normalize করে aggregate করে, তারপর একটি learnable weight matrix দিয়ে
      transform করে। Cora-তে এটি ~81% accuracy দেয় — শুধু নোড ফিচার ব্যবহার করা MLP (~57%) থেকে
      অনেক বেশি।
    </p>
    <p>
      পরের ব্লগে আমরা দেখব <strong>GraphSAGE</strong> — যেটি GCN-এর transductive সীমাবদ্ধতা দূর করে
      নতুন (unseen) নোডেও কাজ করতে পারে।
    </p>
  `,
};
