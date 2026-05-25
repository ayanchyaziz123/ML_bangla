export const gnn_4_gat = {
  title: "গ্রাফ অ্যাটেনশন নেটওয়ার্ক (GAT): প্রতিবেশীর গুরুত্ব শিখে নাও",
  description: "Veličković et al.-এর GAT: attention coefficient, multi-head attention, GCN ও GraphSAGE-এর সাথে তুলনা, PyG-এ GATConv এবং attention weight ভিজুয়ালাইজেশন।",
  date: "২৩ মে, ২০২৬",
  category: "গ্রাফ নিউরাল নেটওয়ার্ক",
  readTime: 12,
  slug: "gnn-gat",
  content: `
    <h3>১. সমস্যা: সব প্রতিবেশী কি সমান গুরুত্বপূর্ণ?</h3>
    <p>
      ধরো তুমি একজন গবেষক। তোমার 10 জন সহকর্মী আছে — তাদের মধ্যে 2 জন তোমার সাথে একই বিষয়ে কাজ করে,
      বাকি 8 জন সম্পূর্ণ ভিন্ন বিষয়ে। তোমার নিজের গবেষণার বিষয় চেনাতে কারা বেশি গুরুত্বপূর্ণ?
      অবশ্যই ঐ 2 জন।
    </p>
    <p>
      GCN এবং GraphSAGE-এর Mean aggregator সব প্রতিবেশীকে সমান গুরুত্ব দেয়। এটি অনেক সময়
      suboptimal।
    </p>
    <p>
      <strong>Graph Attention Networks (GAT)</strong> — Veličković et al. (2018) — এই সমস্যা
      সমাধান করে। এটি প্রতিটি প্রতিবেশীর জন্য একটি <strong>attention weight</strong> শিখে নেয়:
      কোন প্রতিবেশী কতটা গুরুত্বপূর্ণ।
    </p>

    <h3>২. Attention Mechanism: কীভাবে গুরুত্ব নির্ধারণ হয়</h3>
    <p>GAT-এর attention তিনটি ধাপে কাজ করে:</p>
    <p><strong>ধাপ ১ — Linear Transformation:</strong></p>
    <pre><code># প্রতিটি নোডের ফিচার transform করো
z_i = W · h_i
z_j = W · h_j
# W: shared weight matrix</code></pre>
    <p><strong>ধাপ ২ — Attention Coefficient:</strong></p>
    <pre><code># i এবং j-এর মধ্যে attention score
e(i, j) = LeakyReLU(a^T · CONCAT(z_i, z_j))

# a: learnable attention vector
# LeakyReLU: নেগেটিভ মানও কিছুটা পাস করতে দেয়</code></pre>
    <p><strong>ধাপ ৩ — Softmax Normalization:</strong></p>
    <pre><code># Normalize করো (i-এর সব প্রতিবেশীর মধ্যে)
α(i, j) = softmax_j(e(i, j))
         = exp(e(i,j)) / Σ_{k ∈ N(i)} exp(e(i,k))

# Final aggregation
h'_i = σ(Σ_{j ∈ N(i)} α(i,j) · W · h_j)</code></pre>
    <p>
      α(i,j) হলো নোড i-এর জন্য প্রতিবেশী j-এর গুরুত্ব। এটি 0-1 এর মধ্যে থাকে এবং সব প্রতিবেশীর
      জন্য যোগফল 1।
    </p>

    <h3>৩. Multi-Head Attention: আরও স্থিতিশীল শেখা</h3>
    <p>
      NLP-র Transformer-এর মতো, GAT-ও multi-head attention ব্যবহার করে। K টি আলাদা attention
      head একসাথে চলে:
    </p>
    <pre><code># K টি independent attention head
# প্রতিটি head আলাদা W_k এবং a_k শেখে

# Intermediate layers: CONCATENATE
h'_i = CONCAT_{k=1}^{K} σ(Σ_{j ∈ N(i)} α_k(i,j) · W_k · h_j)
# Output dimension: K × out_features

# Final layer: AVERAGE (dimensionality নিয়ন্ত্রণ)
h'_i = σ(1/K · Σ_{k=1}^{K} Σ_{j ∈ N(i)} α_k(i,j) · W_k · h_j)</code></pre>
    <p>
      Multi-head attention এর সুবিধা: বিভিন্ন head বিভিন্ন ধরনের সম্পর্ক শিখতে পারে।
      একটি head হয়তো গবেষণার বিষয় দেখে, আরেকটি author affiliation দেখে।
    </p>

    <h3>৪. PyTorch Geometric দিয়ে GAT</h3>
    <pre><code>import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
import numpy as np

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 heads=8, dropout=0.6):
        super(GAT, self).__init__()

        # Layer 1: 8 heads, concatenate → hidden_channels * heads output
        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True          # intermediate layer: concatenate
        )

        # Layer 2: 1 head, average → out_channels output
        self.conv2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            dropout=dropout,
            concat=False         # final layer: average
        )
        self.dropout = dropout

    def forward(self, x, edge_index, return_attention=False):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)             # ELU activation (original paper)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if return_attention:
            # attention weights ফেরত দাও
            x, (edge_index_att, alpha) = self.conv2(
                x, edge_index, return_attention_weights=True
            )
            return F.log_softmax(x, dim=1), edge_index_att, alpha
        else:
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAT(
    in_channels=dataset.num_node_features,
    hidden_channels=8,
    out_channels=dataset.num_classes,
    heads=8,
    dropout=0.6
).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

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

best_val = 0
for epoch in range(1, 201):
    loss = train()
    if epoch % 20 == 0:
        train_acc, val_acc, test_acc = test()
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | "
              f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}")</code></pre>
    <pre><code># Attention weight ভিজুয়ালাইজ করো
model.eval()
with torch.no_grad():
    out, edge_idx, alpha = model(data.x, data.edge_index, return_attention=True)

# alpha shape: (num_edges,) — প্রতিটি এজের attention weight
alpha_np = alpha.squeeze().cpu().numpy()

plt.figure(figsize=(8, 4))
plt.hist(alpha_np, bins=50, color='steelblue', edgecolor='white')
plt.xlabel('Attention Weight α')
plt.ylabel('এজ সংখ্যা')
plt.title('GAT Attention Weight Distribution (Cora)')
plt.tight_layout()
plt.savefig('gat_attention.png', dpi=150)
plt.show()

print(f"গড় attention: {alpha_np.mean():.4f}")
print(f"সর্বোচ্চ attention: {alpha_np.max():.4f}")
print(f"High-attention এজ (>0.5): {(alpha_np > 0.5).sum()}")</code></pre>

    <h3>৫. GCN, GraphSAGE এবং GAT-এর তুলনা</h3>
    <table>
      <thead>
        <tr><th>বৈশিষ্ট্য</th><th>GCN</th><th>GraphSAGE</th><th>GAT</th></tr>
      </thead>
      <tbody>
        <tr><td>Aggregation</td><td>Normalized sum</td><td>Mean/LSTM/Pool</td><td>Weighted sum (learned)</td></tr>
        <tr><td>Neighbor weight</td><td>Fixed (degree-based)</td><td>Equal (mean)</td><td>Dynamic (attention)</td></tr>
        <tr><td>Inductive</td><td>না</td><td>হ্যাঁ</td><td>হ্যাঁ</td></tr>
        <tr><td>Expressiveness</td><td>কম</td><td>মাঝারি</td><td>বেশি</td></tr>
        <tr><td>Parameters</td><td>কম</td><td>মাঝারি</td><td>বেশি (attention vector)</td></tr>
        <tr><td>Interpretability</td><td>কম</td><td>কম</td><td>বেশি (α দেখা যায়)</td></tr>
        <tr><td>Cora Test Acc</td><td>~81%</td><td>~81%</td><td>~83%</td></tr>
        <tr><td>Training Speed</td><td>দ্রুত</td><td>দ্রুত</td><td>তুলনামূলক ধীর</td></tr>
      </tbody>
    </table>
    <table>
      <thead>
        <tr><th>Attention Variant</th><th>বৈশিষ্ট্য</th><th>কোথায় ব্যবহার</th></tr>
      </thead>
      <tbody>
        <tr><td>GAT (original)</td><td>additive attention, LeakyReLU</td><td>Node classification</td></tr>
        <tr><td>GATv2</td><td>dynamic attention (more expressive)</td><td>বেশি জটিল গ্রাফ</td></tr>
        <tr><td>Transformer-based</td><td>dot-product attention</td><td>Large-scale graphs</td></tr>
      </tbody>
    </table>

    <h3>৬. GAT-এর সুবিধা ও সীমাবদ্ধতা</h3>
    <p><strong>সুবিধা:</strong></p>
    <p>
      Attention weight interpretable — কোন প্রতিবেশী কেন গুরুত্বপূর্ণ তা দেখা যায়।
      ভিন্ন প্রতিবেশীকে ভিন্ন গুরুত্ব দিতে পারে, তাই বেশি expressive।
      Noisy neighbor থাকলে কম attention দিয়ে তাদের প্রভাব কমাতে পারে।
    </p>
    <p><strong>সীমাবদ্ধতা:</strong></p>
    <p>
      GCN বা GraphSAGE-এর চেয়ে বেশি parameter এবং computation।
      খুব বড় গ্রাফে memory সমস্যা হতে পারে।
      Attention নিজেই overfitting-এর শিকার হতে পারে।
    </p>

    <h3>৭. পরবর্তী ধাপ</h3>
    <p>
      এখন পর্যন্ত আমরা GNN-এর তিনটি প্রধান আর্কিটেকচার দেখেছি: GCN, GraphSAGE, GAT।
      পরের ব্লগে আমরা দেখব এই আর্কিটেকচারগুলো <strong>বাস্তব কাজে</strong> কীভাবে ব্যবহার হয় —
      node classification, link prediction, এবং graph classification।
    </p>
  `,
};
