export const gnn_3_graphsage = {
  title: "GraphSAGE: নতুন নোডেও কাজ করে — Inductive Learning",
  description: "Hamilton et al.-এর GraphSAGE: neighborhood sampling, mean/LSTM/pooling aggregator, inductive vs transductive পার্থক্য, PyG-তে SAGEConv এবং mini-batch ট্রেনিং।",
  date: "২৩ মে, ২০২৬",
  category: "গ্রাফ নিউরাল নেটওয়ার্ক",
  readTime: 11,
  slug: "gnn-graphsage",
  content: `
    <h3>১. GCN-এর সমস্যা: নতুন নোড চিনতে পারে না</h3>
    <p>
      ধরো তুমি একটি ই-কমার্স সাইটের পণ্য সুপারিশ সিস্টেম বানাচ্ছ। প্রতিদিন হাজার হাজার নতুন পণ্য
      (নোড) যোগ হচ্ছে। GCN দিয়ে প্রতিদিন পুরো গ্রাফ নিয়ে আবার ট্রেন করা সম্ভব নয়।
    </p>
    <p>
      এটাই GCN-এর <strong>transductive</strong> সমস্যা:
    </p>
    <p>
      <strong>Transductive:</strong> ট্রেনিংয়ের সময় সব নোড (labeled + unlabeled) দেখতে হয়।
      নতুন নোড এলে model আবার train করতে হবে।
    </p>
    <p>
      <strong>Inductive:</strong> ট্রেনিংয়ের সময় শুধু কিছু নোড দেখেছে। তবু নতুন, অদেখা নোডেও
      কাজ করতে পারে — কারণ এটি neighborhood-এর pattern শিখেছে, নির্দিষ্ট নোড ID নয়।
    </p>
    <p>
      <strong>GraphSAGE</strong> (Hamilton et al., 2017) — SAmple and aggreGatE — এই সমস্যার সমাধান করে।
    </p>

    <h3>২. GraphSAGE-এর মূল ধারণা: Sample করো, Aggregate করো</h3>
    <p>GCN পুরো প্রতিবেশিতা ব্যবহার করে। GraphSAGE দুটি চালাকি করে:</p>
    <p>
      <strong>Neighborhood Sampling:</strong> প্রতিটি নোডের সব প্রতিবেশী নেওয়ার বদলে
      fixed-size sample নেওয়া হয়। যেমন, hop 1 থেকে 25টি, hop 2 থেকে 10টি।
      এতে বড় গ্রাফেও memory নিয়ন্ত্রণে থাকে।
    </p>
    <p>
      <strong>Aggregator Function:</strong> নমুনা নেওয়া প্রতিবেশীদের ফিচার একটি নির্দিষ্ট
      function দিয়ে একত্রিত করা হয়।
    </p>
    <p>Update rule:</p>
    <pre><code>h_N(v) = AGGREGATE({h_u : u ∈ sampled_N(v)})
h_v^(l+1) = σ(W · CONCAT(h_v^(l), h_N(v)))
h_v^(l+1) = h_v^(l+1) / ||h_v^(l+1)||_2    (L2 normalization)</code></pre>
    <p>
      লক্ষ্য করো: GCN-এ শুধু প্রতিবেশী aggregate হয়, কিন্তু GraphSAGE-এ নিজের পুরোনো
      embedding আর প্রতিবেশীর aggregate <strong>CONCAT</strong> করা হয়।
    </p>

    <h3>৩. তিনটি Aggregator: কোনটি কীভাবে কাজ করে</h3>
    <p><strong>১. Mean Aggregator:</strong> সবচেয়ে সহজ। প্রতিবেশীদের ফিচারের গড় নেওয়া।</p>
    <pre><code>h_N(v) = mean({h_u : u ∈ N(v)})</code></pre>
    <p>
      এটি GCN-এর নরমালাইজড aggregation-এর মতো। দ্রুত ও কার্যকর।
    </p>
    <p><strong>২. LSTM Aggregator:</strong> প্রতিবেশীদের একটি sequence হিসেবে LSTM-এ দেওয়া।</p>
    <pre><code>h_N(v) = LSTM([h_u : u ∈ shuffled_N(v)])</code></pre>
    <p>
      গ্রাফে প্রতিবেশীদের কোনো order নেই, তাই random shuffle করা হয়। বেশি expressive কিন্তু ধীর।
    </p>
    <p><strong>৩. Pooling Aggregator:</strong> প্রতিটি প্রতিবেশীকে MLP দিয়ে transform করে max/mean pool।</p>
    <pre><code>h_N(v) = max({ReLU(W_pool · h_u + b) : u ∈ N(v)})</code></pre>
    <p>
      Max pooling সবচেয়ে গুরুত্বপূর্ণ প্রতিবেশীর বৈশিষ্ট্য ধরে রাখে।
    </p>

    <h3>৪. PyTorch Geometric দিয়ে GraphSAGE</h3>
    <pre><code>import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Planetoid

# Dataset লোড
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# GraphSAGE মডেল
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GraphSAGE, self).__init__()
        # SAGEConv: mean aggregator by default
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')
        self.dropout = dropout

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGE(
    in_channels=dataset.num_node_features,
    hidden_channels=64,
    out_channels=dataset.num_classes
).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# Training & Evaluation
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append((pred[mask] == data.y[mask]).float().mean().item())
    return accs

for epoch in range(1, 201):
    loss = train()
    if epoch % 20 == 0:
        train_acc, val_acc, test_acc = test()
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | "
              f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}")</code></pre>

    <h3>৫. Mini-batch Training: বড় গ্রাফের জন্য</h3>
    <p>
      বড় গ্রাফে পুরো গ্রাফ একবারে মেমোরিতে রাখা সম্ভব নয়। PyG-এর NeighborLoader
      mini-batch training সম্ভব করে:
    </p>
    <pre><code>from torch_geometric.loader import NeighborLoader

# NeighborLoader: প্রতি hop-এ নির্দিষ্ট সংখ্যক প্রতিবেশী sample করে
train_loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],    # hop 1: 25 প্রতিবেশী, hop 2: 10 প্রতিবেশী
    batch_size=64,
    input_nodes=data.train_mask,
    shuffle=True
)

# Mini-batch training loop
model.train()
total_loss = 0
for batch in train_loader:
    batch = batch.to(device)
    optimizer.zero_grad()
    out = model(batch.x, batch.edge_index)
    # শুধু batch-এর seed নোডগুলোর loss নাও
    loss = F.nll_loss(out[:batch.batch_size], batch.y[:batch.batch_size])
    loss.backward()
    optimizer.step()
    total_loss += loss.item()

print(f"Average Loss: {total_loss / len(train_loader):.4f}")</code></pre>

    <h3>৬. GCN বনাম GraphSAGE তুলনা</h3>
    <table>
      <thead>
        <tr><th>বৈশিষ্ট্য</th><th>GCN</th><th>GraphSAGE</th></tr>
      </thead>
      <tbody>
        <tr><td>Learning type</td><td>Transductive</td><td>Inductive</td></tr>
        <tr><td>নতুন নোড handle</td><td>না</td><td>হ্যাঁ</td></tr>
        <tr><td>Aggregation</td><td>Symmetric normalized sum</td><td>Mean / LSTM / Pooling</td></tr>
        <tr><td>Self-connection</td><td>Ã = A + I</td><td>CONCAT(self, neighbor_agg)</td></tr>
        <tr><td>Neighborhood</td><td>সম্পূর্ণ প্রতিবেশিতা</td><td>Fixed-size sampling</td></tr>
        <tr><td>Mini-batch</td><td>কঠিন</td><td>সহজ (NeighborLoader)</td></tr>
        <tr><td>Cora Accuracy</td><td>~81%</td><td>~81-82%</td></tr>
        <tr><td>বড় গ্রাফ</td><td>কঠিন</td><td>সহজ</td></tr>
      </tbody>
    </table>
    <p>
      মূল শিক্ষা: GraphSAGE শুধু accuracy-তে নয়, <em>scalability</em> এবং
      <em>flexibility</em>-তে GCN-কে ছাড়িয়ে যায়। বাস্তব পণ্য পরিবেশে (Pinterest, Uber Eats)
      GraphSAGE-এর মতো inductive পদ্ধতি ব্যবহার হয়।
    </p>

    <h3>৭. পরবর্তী ধাপ</h3>
    <p>
      GraphSAGE সব প্রতিবেশীকে সমান গুরুত্ব দেয় (mean aggregator)। কিন্তু বাস্তবে কিছু প্রতিবেশী
      বেশি গুরুত্বপূর্ণ। পরের ব্লগে আমরা দেখব <strong>Graph Attention Networks (GAT)</strong> —
      যেটি প্রতিটি প্রতিবেশীর গুরুত্ব <em>শিখে নেয়</em>।
    </p>
  `,
};
