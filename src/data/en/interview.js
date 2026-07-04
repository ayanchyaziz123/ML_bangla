export const interviewEn = [
  {
    slug: 'interview-1-100',
    title: 'ML & Deep Learning Interview Questions (Intermediate → Hard) — 100 Q&A with Examples',
    description: '100 intermediate-to-hard ML/DL interview questions with real-failure-case answers, PyTorch code examples for deep learning topics, and mock-interview follow-up reasoning.',
    category: 'ML Interview Questions',
    readTime: 40,
    content: `
<h3>Core ML Concepts (Q1–Q20)</h3>

<p><strong>Q1. Explain the bias-variance tradeoff with a real failure case.</strong><br/>
Bias = error from overly simple assumptions (underfitting). Variance = error from being too sensitive to training data (overfitting). Total error = Bias² + Variance + irreducible noise.<br/>
<em>Example:</em> A linear regression predicting house price from only square footage underfits (high bias) — it ignores location/age. A depth-20 decision tree on the same 200 rows gets training R²=0.99 but test R²=0.4 (high variance) — it memorized noise.</p>

<p><strong>Q2. When does adding more data not improve model performance?</strong><br/>
When the model is bias-limited (too simple regardless of data volume), when new data comes from a different distribution, or when it's dominated by label noise.<br/>
<em>Example:</em> Adding 10× more mislabeled tweets to a sentiment classifier reinforces noise instead of signal — accuracy can even drop.</p>

<p><strong>Q3. Compare L1 vs L2 regularization beyond definitions.</strong><br/>
L1 (Lasso) adds a |w| penalty — its diamond-shaped constraint region has corners on the axes, so optimization often lands exactly on zero, giving built-in feature selection. L2 (Ridge) adds a w² penalty — its circular constraint region has no corners, so weights shrink smoothly but rarely hit exact zero.<br/>
<em>Example:</em> In a genomics dataset with 20,000 genes but ~50 relevant ones, L1 zeroes out ~19,950 irrelevant coefficients automatically.
<pre><code>import torch.nn as nn
l1_penalty = sum(p.abs().sum() for p in model.parameters())
l2_penalty = sum((p ** 2).sum() for p in model.parameters())
loss = criterion(output, target) + 1e-4 * l1_penalty + 1e-4 * l2_penalty</code></pre></p>

<p><strong>Q4. Why does overfitting still happen with cross-validation?</strong><br/>
CV protects against overfitting to one validation split, but running hundreds of hyperparameter trials against the same CV score leaks information through the search process itself — you overfit to the folds.<br/>
<em>Example:</em> 500 Optuna trials optimizing 5-fold CV AUC, then reporting that same CV AUC as final performance — a held-out test set usually scores lower.</p>

<p><strong>Q5. Explain situations where accuracy is misleading.</strong><br/>
Imbalanced classes make accuracy trivially high by always predicting the majority class.<br/>
<em>Example:</em> A fraud model on data with 0.5% fraud gets 99.5% accuracy predicting "not fraud" for everyone — 0% recall on the class that matters.</p>

<p><strong>Q6. How do you detect data leakage in a pipeline?</strong><br/>
Suspiciously high validation scores, a feature near-perfectly correlated with the target, or a feature that wouldn't exist at prediction time.<br/>
<em>Example:</em> A churn model using "account_closed_date" as a feature — that column only gets populated after the churn event, so it's leaking the label.</p>

<p><strong>Q7. Why can a high AUC still give poor predictions?</strong><br/>
AUC measures ranking quality across all thresholds, not calibration at the specific threshold you actually deploy.<br/>
<em>Example:</em> A model with AUC=0.95 but poorly calibrated probabilities (everything squeezed between 0.4–0.6) makes bad decisions at a fixed 0.5 cutoff even though it ranks correctly.</p>

<p><strong>Q8. What happens when features are highly correlated?</strong><br/>
Multicollinearity makes linear coefficients unstable and hard to interpret (sign/magnitude flips with small data changes), though it usually doesn't hurt predictive accuracy much.<br/>
<em>Example:</em> A regression with both "square_feet" and "square_meters" (perfectly correlated) causes wildly fluctuating coefficient estimates between runs.</p>

<p><strong>Q9. Why do tree models not require feature scaling?</strong><br/>
Trees split on threshold comparisons (feature &gt; x), which are invariant to monotonic transforms like scaling — the split point just shifts.<br/>
<em>Example:</em> Splitting on "income &gt; 50000" vs "income_scaled &gt; 0.3" produces the identical partition of data either way.</p>

<p><strong>Q10. When would you prefer a simple model over a complex one?</strong><br/>
Small/noisy data, need for interpretability (regulatory, medical), tight latency budgets, or when the complex model's accuracy gain is marginal.<br/>
<em>Example:</em> A loan-approval model must be explainable to regulators — logistic regression with clear coefficients beats a black-box gradient-boosted ensemble even with 2% higher AUC.</p>

<p><strong>Q11. How do you debug a model that performs well in training but poorly in production?</strong><br/>
Check train/production feature distribution mismatch, verify preprocessing consistency, check for train-only leakage, confirm production data quality — and in PyTorch specifically, confirm the model is actually in eval mode.<br/>
<em>Example:</em> A very common real bug: forgetting <code>model.eval()</code> before inference leaves Dropout and BatchNorm running in training mode, so predictions become noisy and unstable in production.
<pre><code>model.eval()          # disables dropout, freezes BatchNorm running stats
with torch.no_grad():
    preds = model(x_batch)</code></pre></p>

<p><strong>Q12. Explain the curse of dimensionality with intuition.</strong><br/>
As dimensions grow, data points become nearly equidistant and space becomes overwhelmingly sparse — "nearest neighbor" stops being meaningfully near.<br/>
<em>Example:</em> In 2D, a unit square is mostly interior; in 100D, almost all the volume of a hypercube sits near its corners — random points end up nearly equidistant from each other.</p>

<p><strong>Q13. What are the risks of using PCA before splitting data?</strong><br/>
Fitting PCA on the full dataset before the train/test split leaks test-set variance/structure into the training-time transformation — a form of data leakage.<br/>
<em>Example:</em> PCA fit on all 10,000 rows, then split 80/20 — the components already "saw" the test rows' distribution, inflating validation scores.</p>

<p><strong>Q14. How do you choose evaluation metrics for imbalanced datasets?</strong><br/>
Prefer PR-AUC, F1, precision/recall over accuracy or plain ROC-AUC; pick based on the relative cost of false positives vs false negatives.<br/>
<em>Example:</em> Cancer screening (FN costly) → optimize recall/sensitivity even at the cost of precision.</p>

<p><strong>Q15. Why can increasing model complexity reduce performance?</strong><br/>
Past a point, added complexity fits noise instead of signal — variance grows faster than bias shrinks.<br/>
<em>Example:</em> Fitting a degree-15 polynomial to 30 data points memorizes noise, producing wild oscillations between training points.</p>

<p><strong>Q16. What happens if training and test distributions differ?</strong><br/>
Covariate shift — the learned decision boundary no longer matches where test data actually lives, so performance degrades even if train metrics look great.<br/>
<em>Example:</em> A model trained on daytime traffic-camera images fails on nighttime images because the pixel-intensity distribution shifted.</p>

<p><strong>Q17. Explain stratified sampling and when it fails.</strong><br/>
Stratified sampling preserves the class ratio in every split. It fails when strata are too fine-grained (too few samples per stratum) or when the chosen stratification variable isn't the one that actually matters.<br/>
<em>Example:</em> Stratifying a multi-label dataset by only the primary label can still leave rare label combinations completely absent from the validation fold.</p>

<p><strong>Q18. What is concept drift and how do you detect it?</strong><br/>
The relationship between features and target changes over time (P(y|x) changes) even if P(x) stays the same. Detect via live performance monitoring, statistical tests (KS test) on prediction/error distributions, or drift algorithms (ADWIN, DDM).<br/>
<em>Example:</em> A spam classifier trained in 2023 degrades by 2026 because spammers changed tactics — identical email features now map to a different label.</p>

<p><strong>Q19. Why do ensemble models often outperform single models?</strong><br/>
Combining diverse, decorrelated models reduces variance (bagging) and can reduce bias (boosting) — errors uncorrelated across models cancel out.<br/>
<em>Example:</em> The Netflix Prize winning solution blended 100+ models; no single model matched the blend's accuracy.</p>

<p><strong>Q20. When does cross-validation fail to give reliable estimates?</strong><br/>
With time series (random folds leak future info into training), tiny datasets (high variance across folds), or grouped data (multiple rows per patient) where the split ignores the grouping.<br/>
<em>Example:</em> K-fold CV on patient-level medical records where the same patient's rows appear in both train and test folds leaks patient-specific signal — GroupKFold is needed instead.</p>

<h3>Supervised Learning Deep Dive (Q21–Q40)</h3>

<p><strong>Q21. Derive the cost function for logistic regression.</strong><br/>
Modeling P(y|x)=σ(wx+b) as a Bernoulli likelihood and taking the negative log-likelihood gives binary cross-entropy: J = -1/n Σ[yᵢlog(ŷᵢ) + (1-yᵢ)log(1-ŷᵢ)]. Unlike squared error on a sigmoid, this loss is convex in w.<br/>
<em>Example:</em> For a point with true y=1 and predicted ŷ=0.01, the loss -log(0.01)≈4.6 is large — confident wrong predictions are penalized sharply.
<pre><code>import torch.nn as nn
criterion = nn.BCEWithLogitsLoss()   # combines sigmoid + BCE, numerically stable
loss = criterion(logits, targets.float())</code></pre></p>

<p><strong>Q22. Why does gradient descent sometimes not converge?</strong><br/>
Learning rate too high → oscillation/divergence. Too low → painfully slow progress. Non-convex surfaces have saddle points/local minima that stall progress. Poor feature scaling distorts the loss landscape into narrow ravines.<br/>
<em>Example:</em> Training with lr=10 causes the loss to explode to NaN within a few steps — a classic sign the learning rate is far too high.</p>

<p><strong>Q23. What causes vanishing gradients in simple neural networks?</strong><br/>
Sigmoid/tanh derivatives are ≤0.25, and the chain rule multiplies these small numbers layer after layer, shrinking gradients exponentially toward the input layers.<br/>
<em>Example:</em> In a 10-layer sigmoid network, a per-layer gradient factor of 0.2 becomes 0.2¹⁰ ≈ 1×10⁻⁷ at layer 1 — effectively zero, so early layers stop learning.</p>

<p><strong>Q24. Compare decision trees vs random forests in edge cases.</strong><br/>
A single tree is unstable — small data changes can produce very different trees (high variance) — but fully interpretable. Random forest averages many trees over bootstrap samples and random feature subsets, cutting variance, but you lose the single clean decision path.<br/>
<em>Example:</em> Removing 5 rows can flip a single tree's root split entirely, but barely changes a 500-tree forest's predictions.</p>

<p><strong>Q25. Why does boosting reduce bias but risk overfitting?</strong><br/>
Each new weak learner fits the ensemble's current residual errors, progressively correcting systematic bias — but with enough rounds it starts fitting noise in those residuals too.<br/>
<em>Example:</em> XGBoost with 5000 rounds and no early stopping on noisy data reaches near-zero training error, but validation error creeps back up after round ~300 — the classic overfitting curve.</p>

<p><strong>Q26. Explain how XGBoost handles missing values.</strong><br/>
At each split, XGBoost learns a "default direction" for missing values by trying both directions and picking whichever minimizes loss — no imputation needed.<br/>
<em>Example:</em> An "income" feature with 15% missing values is automatically routed left or right at each split, based on whichever direction improved that split's gain during training.</p>

<p><strong>Q27. When does SVM fail even with kernel tricks?</strong><br/>
Very large datasets (kernel SVM is O(n²)-O(n³)), heavily overlapping classes with no clear margin, or high-dimensional sparse data where the kernel doesn't capture the right structure.<br/>
<em>Example:</em> Training an RBF-kernel SVM on 1M rows takes hours or runs out of memory — gradient boosting scales far better.</p>

<p><strong>Q28. Why is hinge loss used instead of log loss in some cases?</strong><br/>
Hinge loss (max(0, 1-y·f(x))) only penalizes misclassified or margin-violating points, giving zero gradient to confidently correct points — producing the sparse "support vector" solution and directly optimizing margin, not calibrated probability.<br/>
<em>Example:</em> A point safely past the decision margin contributes 0 to hinge loss, but log loss would still nudge it further, unnecessarily.</p>

<p><strong>Q29. Explain margin maximization intuitively.</strong><br/>
SVM doesn't just find any separating boundary — it finds the one with the largest buffer zone between classes, which tends to generalize better to unseen points near the boundary.<br/>
<em>Example:</em> Of two possible separating lines between two clusters, the one exactly centered (max margin) misclassifies fewer new points sampled near the boundary.</p>

<p><strong>Q30. Why do k-NN models struggle with high dimensions?</strong><br/>
The same curse-of-dimensionality issue — in high-D space, distances between all points converge, so "nearest" neighbors aren't meaningfully closer than "far" ones.<br/>
<em>Example:</em> In 500-dim word-embedding space, cosine distance between a truly similar and a random dissimilar document can differ by only a few percent.</p>

<p><strong>Q31. What happens if k=1 vs very large k in k-NN?</strong><br/>
k=1 → zero training error but very high variance (each prediction governed by a single, possibly noisy, neighbor). Very large k (k=n) → high bias, predictions approach the global majority class, ignoring local structure.<br/>
<em>Example:</em> k=1 on noisy labeled data creates a jagged, overfit decision boundary; k=n just predicts the same majority class everywhere.</p>

<p><strong>Q32. Explain why Naive Bayes works even when assumptions are violated.</strong><br/>
NB only needs to rank the correct class highest, not produce accurate probabilities — the independence assumption's errors often bias all classes similarly, preserving the correct argmax ranking.<br/>
<em>Example:</em> "New" and "York" are clearly dependent, not independent, yet Naive Bayes still classifies "New York Times" articles correctly because the mis-estimation is systematic across classes.</p>

<p><strong>Q33. When is linear regression inappropriate even for linear-looking data?</strong><br/>
When errors are heteroscedastic, non-normal, when there's severe autocorrelation (time series), or when outliers dominate — OLS assumptions are violated even if a scatter plot looks linear.<br/>
<em>Example:</em> Stock returns vs a linear predictor can look linear on a scatter plot but have fat-tailed, autocorrelated residuals, invalidating OLS confidence intervals.</p>

<p><strong>Q34. Why do polynomial features sometimes hurt performance?</strong><br/>
High-degree polynomial terms overfit rapidly (Runge's phenomenon) and create multicollinearity between terms (x, x², x³ are highly correlated), destabilizing coefficient estimates.<br/>
<em>Example:</em> A degree-9 polynomial through 10 points passes through every point exactly but oscillates wildly between them — terrible interpolation.</p>

<p><strong>Q35. How do you interpret coefficients in regularized regression?</strong><br/>
Regularized coefficients are shrunk toward zero based on the fit-vs-penalty tradeoff — their magnitude isn't directly comparable to unregularized OLS coefficients and shouldn't be read as unbiased "effect sizes."<br/>
<em>Example:</em> A Ridge coefficient of 0.3 for a feature might correspond to an OLS coefficient of 0.8 for the same feature — the shrinkage depends on λ, not just the true effect.</p>

<p><strong>Q36. What happens if learning rate is too small?</strong><br/>
Training converges extremely slowly, can look stuck on a plateau for so many steps it appears to have stopped learning, and wastes compute budget.
<pre><code>import torch.optim as optim
optimizer = optim.SGD(model.parameters(), lr=1e-8)  # far too small
# loss barely moves even after thousands of steps</code></pre></p>

<p><strong>Q37. Why can early stopping act as regularization?</strong><br/>
Stopping before the model fully converges on training loss prevents it from reaching the point where it memorizes noise — it implicitly limits effective model capacity, similar to shrinking weights.
<pre><code>best_val, patience, wait = float('inf'), 5, 0
for epoch in range(epochs):
    train_one_epoch(model, train_loader, optimizer)
    val_loss = evaluate(model, val_loader)
    if val_loss < best_val:
        best_val, wait = val_loss, 0
        torch.save(model.state_dict(), 'best.pt')
    else:
        wait += 1
        if wait >= patience:
            break   # stop before the model starts memorizing noise</code></pre></p>

<p><strong>Q38. Compare bagging vs boosting with examples.</strong><br/>
Bagging trains models in parallel on bootstrap samples and averages them to reduce variance (Random Forest). Boosting trains models sequentially, each correcting the previous ensemble's errors, reducing bias (AdaBoost, XGBoost).<br/>
<em>Example:</em> Random Forest on noisy data is robust to outliers because averaging cancels noise; XGBoost on the same data may overfit to outliers since later rounds chase their large residuals.</p>

<p><strong>Q39. Why does random forest reduce variance but not bias?</strong><br/>
Averaging many high-variance, low-bias trees cancels out their independent noise, but each tree shares the same structural limitation from the algorithm — averaging can't fix systematic underfitting.<br/>
<em>Example:</em> If every tree is capped at depth=2 (systematically too shallow for a complex boundary), averaging 500 of them still can't represent that boundary — bias remains high.</p>

<p><strong>Q40. Explain calibration of probabilities.</strong><br/>
A calibrated model's predicted probability matches the true empirical frequency — among all predictions of 0.7, about 70% should actually be positive. Poorly calibrated models (raw SVM scores, overconfident deep nets) need Platt scaling or isotonic regression.<br/>
<em>Example:</em> A weather model predicting "70% chance of rain" is calibrated if it actually rains on ~70% of such days — not 95% or 40%.</p>

<h3>Optimization &amp; Training (Q41–Q55)</h3>

<p><strong>Q41. Compare SGD, Momentum, RMSProp, Adam.</strong><br/>
SGD updates using the raw gradient each step (simple, noisy). Momentum accumulates a moving average of past gradients to smooth updates and accelerate through ravines. RMSProp adapts the learning rate per-parameter using a moving average of squared gradients. Adam combines momentum + RMSProp's adaptive scaling.
<pre><code>import torch.optim as optim
sgd      = optim.SGD(model.parameters(), lr=0.01)
momentum = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
rmsprop  = optim.RMSprop(model.parameters(), lr=0.001)
adam     = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))</code></pre>
On a narrow ravine, plain SGD oscillates across the walls; momentum dampens that oscillation and accelerates along the ravine floor.</p>

<p><strong>Q42. Why does Adam sometimes generalize worse than SGD?</strong><br/>
Adam's adaptive per-parameter learning rates can converge to sharp minima that fit training data precisely but generalize poorly, whereas SGD's noisier updates tend to find flatter minima.<br/>
<em>Example:</em> Many vision models (ResNet) are still trained with SGD+momentum rather than Adam specifically because final test accuracy is measurably better, despite Adam converging faster early on.</p>

<p><strong>Q43. What is the saddle point problem in optimization?</strong><br/>
In high-dimensional non-convex surfaces, saddle points (gradient zero, minimum along some directions, maximum along others) vastly outnumber local minima — gradient descent can slow to a crawl near them since the gradient magnitude shrinks.<br/>
<em>Example:</em> Near a saddle point, the loss curve looks flat along one axis (looks like convergence) but is still descending along another axis if you look further.</p>

<p><strong>Q44. How do you escape local minima in deep networks?</strong><br/>
Use momentum/Adam (carries velocity through flat regions), rely on mini-batch SGD's inherent noise, use learning-rate warm restarts, or increase model width (over-parameterization empirically avoids bad local minima).
<pre><code>scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2)   # periodically bumps LR back up
for epoch in range(epochs):
    train_one_epoch(...)
    scheduler.step()</code></pre></p>

<p><strong>Q45. Why is weight initialization important?</strong><br/>
Bad initialization can cause activations/gradients to vanish or explode from the very first forward/backward pass; good initialization (Xavier/He) keeps activation variance roughly constant across layers.
<pre><code>import torch.nn as nn
linear = nn.Linear(256, 256)
nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')  # He init for ReLU nets
nn.init.xavier_normal_(linear.weight)                        # for tanh/sigmoid nets</code></pre></p>

<p><strong>Q46. What happens if all weights are initialized to zero?</strong><br/>
Every neuron in a layer computes the identical output and receives the identical gradient — symmetry is never broken, so all neurons stay identical forever, collapsing the layer to one neuron's worth of capacity.
<pre><code>for p in model.parameters():
    nn.init.zeros_(p)
# every neuron in a layer now updates identically — 512 neurons behave like 1</code></pre></p>

<p><strong>Q47. Explain exploding gradients and mitigation strategies.</strong><br/>
Gradients grow exponentially through backprop when layer derivatives/weights exceed 1, causing huge unstable updates (or NaN). Mitigate with gradient clipping, better initialization, BatchNorm, or smaller learning rates.
<pre><code>loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
optimizer.step()</code></pre>
An RNN unrolled over 100 timesteps with weight-matrix spectral norm 1.5 amplifies gradients by 1.5¹⁰⁰ — clipping the norm to 5 keeps training stable.</p>

<p><strong>Q48. Why does batch normalization speed up training?</strong><br/>
It normalizes each layer's activations to zero mean/unit variance per mini-batch, smoothing the loss landscape and allowing higher learning rates.
<pre><code>import torch.nn as nn
layer = nn.Sequential(
    nn.Linear(256, 256),
    nn.BatchNorm1d(256),
    nn.ReLU()
)</code></pre>
A CNN that needs lr=0.001 without BatchNorm can often train stably at lr=0.01+ with it, cutting training time significantly.</p>

<p><strong>Q49. When can batch normalization hurt performance?</strong><br/>
With very small batch sizes (batch statistics become noisy/unrepresentative), across RNN timesteps (breaks the recurrent computation), or when train/inference batch statistics differ significantly.<br/>
<em>Example:</em> Object detection models trained with batch size 2 (GPU memory limits) show unstable BatchNorm statistics — <code>nn.GroupNorm</code> is often substituted instead.</p>

<p><strong>Q50. Explain gradient clipping and its use cases.</strong><br/>
Caps the gradient's norm (or individual values) before the weight update, preventing one large gradient spike from destabilizing training — essential in RNN/LSTM and Transformer training.
<pre><code># clip by norm (rescales the whole gradient vector)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# clip by value (clamps each gradient element independently)
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)</code></pre></p>

<p><strong>Q51. Why does mini-batch training outperform full-batch?</strong><br/>
Mini-batches give noisy-but-frequent gradient updates (many per epoch vs one), the noise helps escape sharp minima/saddle points, and it fits within GPU memory for large datasets.
<pre><code>from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=256, shuffle=True)  # not batch_size=len(dataset)</code></pre></p>

<p><strong>Q52. What is the effect of batch size on generalization?</strong><br/>
Smaller batches → noisier gradients → flatter minima → better generalization. Larger batches → smoother gradients → sharper minima → sometimes worse test performance (the "generalization gap").<br/>
<em>Example:</em> The same ResNet trained with batch size 32 vs 8192 (same epochs) — batch=8192 often shows a noticeable test-accuracy drop unless the learning rate is linearly scaled and warmup is used.</p>

<p><strong>Q53. How does learning rate scheduling help?</strong><br/>
A larger LR early speeds up progress; decaying it later allows precise convergence into a minimum without overshooting.
<pre><code>scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# lr = 0.1 for epochs 1-30, 0.01 for 31-60, 0.001 for 61-90</code></pre></p>

<p><strong>Q54. Explain warm restarts in training.</strong><br/>
Periodically reset the LR back to a high value mid-training (e.g., cosine annealing schedule), "restarting" the optimizer's exploration — can escape sharp minima and act like an implicit ensemble (snapshot ensembling).
<pre><code>scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
# LR resets to max every 10 epochs; snapshots near each restart can be ensembled</code></pre></p>

<p><strong>Q55. Why does dropout work?</strong><br/>
Randomly zeroing a fraction of neurons each training step prevents co-adaptation and approximates training an ensemble of exponentially many thinned sub-networks, averaged at test time.
<pre><code>import torch.nn as nn
layer = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Dropout(p=0.5))
model.train()   # dropout active — random ~50% of neurons zeroed each step
model.eval()    # dropout disabled — all neurons active, outputs auto-scaled</code></pre></p>

<h3>Neural Networks &amp; Deep Learning (Q56–Q75)</h3>

<p><strong>Q56. Why are deep networks more powerful than shallow ones?</strong><br/>
Depth allows hierarchical composition of features (edges → textures → parts → objects), representing complex functions with exponentially fewer neurons than an equally expressive shallow (wide) network.<br/>
<em>Example:</em> A CNN's first layer detects edges, middle layers detect textures/shapes, and final layers detect whole objects — each layer reuses the previous layer's features.</p>

<p><strong>Q57. Explain universal approximation theorem limitations.</strong><br/>
UAT guarantees a wide-enough single hidden layer can approximate any continuous function, but says nothing about how many neurons are needed (could be exponential) or whether such a network is actually learnable via gradient descent from random init.<br/>
<em>Example:</em> Approximating a moderately complex function with one hidden layer might theoretically need millions of neurons, while a deep network with far fewer total parameters matches that accuracy in practice.</p>

<p><strong>Q58. What makes ReLU better than sigmoid/tanh?</strong><br/>
ReLU's gradient is exactly 1 for positive inputs (no vanishing gradient in the active region), it's computationally cheap, and it produces sparse activations.
<pre><code>import torch
x = torch.linspace(-5, 5, 5)
print(torch.sigmoid(x).detach())      # gradient shrinks toward 0 at the tails
print(torch.relu(x).detach())          # gradient is exactly 1 for x > 0, flat 0 otherwise</code></pre></p>

<p><strong>Q59. When does ReLU fail?</strong><br/>
"Dying ReLU" — if a neuron's weights update such that its input is always negative, its output and gradient become permanently zero, and it never recovers (gradient is 0 for negative input).<br/>
<em>Example:</em> A high learning rate pushes a neuron's bias very negative early in training; that neuron outputs 0 forever after, wasting model capacity.</p>

<p><strong>Q60. Compare Leaky ReLU, ELU, GELU.</strong><br/>
Leaky ReLU allows a small negative slope (e.g., 0.01x for x&lt;0) to prevent dying neurons. ELU smoothly saturates to a negative constant for x&lt;0, giving activations closer to zero mean. GELU weights inputs by their percentile under a Gaussian, giving smooth probabilistic gating — used in BERT/GPT.
<pre><code>import torch.nn as nn
leaky = nn.LeakyReLU(negative_slope=0.01)
elu   = nn.ELU(alpha=1.0)
gelu  = nn.GELU()   # near-universal choice in modern Transformer blocks</code></pre></p>

<p><strong>Q61. Why do deeper networks become harder to train?</strong><br/>
Vanishing/exploding gradients compound with depth, and without skip connections the network must learn increasingly complex identity-like mappings just to pass useful information through — gradient descent struggles to find this directly.<br/>
<em>Example:</em> In the original ResNet paper, a plain (non-residual) 56-layer CNN had higher training error than a plain 20-layer CNN — pure depth without shortcuts actively hurt optimization, not just generalization.</p>

<p><strong>Q62. Explain residual connections and why they help.</strong><br/>
y = F(x) + x lets the network learn the residual F(x) ≈ y − x (often close to zero) instead of the full mapping, and gives a direct identity gradient path back to earlier layers, bypassing vanishing gradients.
<pre><code>import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return self.relu(out + x)   # skip connection: F(x) + x</code></pre></p>

<p><strong>Q63. What is vanishing gradient in RNNs?</strong><br/>
Backpropagation through time repeatedly multiplies the same recurrent weight matrix across timesteps; if its eigenvalues are &lt;1, gradients shrink exponentially with sequence length, so the network can't learn long-range dependencies.<br/>
<em>Example:</em> An RNN connecting a subject at timestep 1 to a verb at timestep 50 effectively gets zero gradient signal for that dependency after 50 steps of backprop.</p>

<p><strong>Q64. Why do LSTMs solve long-term dependency issues?</strong><br/>
The cell state has a near-linear, additive update path (governed by the forget gate) instead of repeated matrix multiplication, so gradients flow largely unchanged across many timesteps when the forget gate stays near 1.
<pre><code>import torch.nn as nn
lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True)
output, (h_n, c_n) = lstm(x)   # c_n carries long-range signal via the forget gate</code></pre></p>

<p><strong>Q65. Compare LSTM vs GRU in practical scenarios.</strong><br/>
LSTM has 3 gates (input/forget/output) and a separate cell state — more expressive, more parameters. GRU merges cell and hidden state with 2 gates (reset/update) — fewer parameters, faster to train, often matches LSTM on smaller datasets.
<pre><code>import torch.nn as nn
lstm = nn.LSTM(128, 256, batch_first=True)
gru  = nn.GRU(128, 256, batch_first=True)
print(sum(p.numel() for p in lstm.parameters()))  # ~4x hidden_size*(input+hidden) params
print(sum(p.numel() for p in gru.parameters()))   # ~3x — noticeably fewer</code></pre></p>

<p><strong>Q66. Why do CNNs work well for images?</strong><br/>
Convolution exploits two image-specific priors: local spatial correlation and translation invariance, achieved via local receptive fields and weight sharing.<br/>
<em>Example:</em> A single 3×3 edge-detecting filter is reused across every position in the image instead of learning a separate filter per pixel location, cutting parameters from millions to a few dozen.</p>

<p><strong>Q67. Explain the convolution operation intuitively.</strong><br/>
A small filter (kernel) slides over the image, computing a weighted sum at each position — producing a feature map that highlights where the pattern the filter represents (e.g., a vertical edge) occurs.
<pre><code>import torch.nn as nn
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
feature_map = conv(image_batch)   # 16 learned filters slide over the image</code></pre>
A 3×3 vertical-edge kernel [[1,0,-1],[1,0,-1],[1,0,-1]] responds strongly wherever intensity jumps from bright to dark left-to-right.</p>

<p><strong>Q68. What is the role of pooling layers?</strong><br/>
Downsamples feature maps (max-pooling takes the max in each window), reducing spatial resolution/computation and adding translation invariance.
<pre><code>import torch.nn as nn
pool = nn.MaxPool2d(kernel_size=2, stride=2)
out = pool(feature_map)   # 224x224 → 112x112, keeps "was this feature present here"</code></pre></p>

<p><strong>Q69. When should pooling be avoided?</strong><br/>
When precise spatial/location information matters, like semantic segmentation or pose estimation — pooling's information loss directly hurts pixel-level output quality; strided or dilated convolutions are used instead.<br/>
<em>Example:</em> U-Net for medical image segmentation uses skip connections specifically to recover the fine spatial detail that pooling in the encoder destroyed.</p>

<p><strong>Q70. Explain padding and stride impact.</strong><br/>
Padding adds border pixels so edge pixels get equal treatment and controls output size ("same" padding keeps output size = input size). Stride controls how far the filter moves each step — stride 2 roughly halves output resolution.
<pre><code>import torch.nn as nn
conv_same   = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1)  # 5x5 -> 5x5
conv_strided = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2)  # 5x5 -> 3x3</code></pre></p>

<p><strong>Q71. Why do transformers replace RNNs in many tasks?</strong><br/>
Transformers process all sequence positions in parallel (no sequential dependency), enabling much faster training, and self-attention directly connects any two positions regardless of distance, avoiding RNNs' vanishing-gradient-limited long-range dependencies.<br/>
<em>Example:</em> Training on a sequence of length 512 — an RNN must process 512 sequential steps; a Transformer computes all attention scores in parallel matrix ops, drastically cutting wall-clock training time on GPUs.</p>

<p><strong>Q72. What is the attention mechanism conceptually?</strong><br/>
For each query, attention computes a weighted average of all value vectors, where weights (from softmax(QKᵀ/√d)) reflect how relevant each key is to that query — letting the model dynamically focus on the most relevant input parts.
<pre><code>import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / d_k ** 0.5
    weights = F.softmax(scores, dim=-1)
    return weights @ V

# or simply: nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)</code></pre></p>

<p><strong>Q73. Why is self-attention powerful?</strong><br/>
It directly models pairwise interactions between every pair of tokens in a single layer (O(1) path length), regardless of distance in the sequence, unlike RNNs where distant tokens need many sequential steps to interact.<br/>
<em>Example:</em> In a 1000-word document, self-attention lets word 5 and word 995 interact directly in one layer, while an RNN would need to propagate through 990 intermediate hidden states.</p>

<p><strong>Q74. Explain positional encoding necessity.</strong><br/>
Self-attention is permutation-invariant — it produces the same output regardless of token order unless position information is injected, so positional encodings (sinusoidal or learned) are added to embeddings.
<pre><code>import torch

def sinusoidal_positional_encoding(seq_len, d_model):
    pos = torch.arange(seq_len).unsqueeze(1)
    i = torch.arange(d_model).unsqueeze(0)
    angle = pos / (10000 ** (2 * (i // 2) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(angle[:, 0::2])
    pe[:, 1::2] = torch.cos(angle[:, 1::2])
    return pe</code></pre>
Without positional encoding, "dog bites man" and "man bites dog" would produce identical attention computations since attention only sees the unordered set of token embeddings.</p>

<p><strong>Q75. When do deep networks overfit despite regularization?</strong><br/>
When the model is massively overparameterized relative to a small/noisy dataset, regularization strength is insufficient, or regularization targets weights but not architecture-specific overfitting sources.<br/>
<em>Example:</em> A 100M-parameter transformer fine-tuned on only 500 labeled examples can still overfit noticeably even with dropout=0.3 and weight decay, because parameter count vastly outnumbers examples.</p>

<h3>Advanced Topics (Q76–Q90)</h3>

<p><strong>Q76. Explain transfer learning and when it fails.</strong><br/>
Reuse features/weights learned on a large source task, usually fine-tuning only the last layers (or all layers with a small LR). It fails when source and target domains are too dissimilar (negative transfer) or the target task needs features never present in the source domain.
<pre><code>import torchvision.models as models
backbone = models.resnet50(weights='IMAGENET1K_V2')
for param in backbone.parameters():
    param.requires_grad = False        # freeze pretrained backbone
backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)  # train new head only</code></pre></p>

<p><strong>Q77. Why does fine-tuning sometimes degrade performance?</strong><br/>
Catastrophic forgetting (new task's gradients overwrite pretrained features), too high a learning rate destroying pretrained weights early, or too few target examples causing severe overfitting.<br/>
<em>Example:</em> Fine-tuning a pretrained BERT with lr=1e-3 (far too high) for 2 epochs often collapses language understanding entirely — recommended fine-tuning LR is closer to 2e-5.</p>

<p><strong>Q78. What is catastrophic forgetting?</strong><br/>
A model trained sequentially on task B forgets task A because gradient updates for B overwrite the weights important for A, with no mechanism protecting that earlier knowledge.<br/>
<em>Example:</em> A chatbot fine-tuned only on customer-support conversations gradually loses the general conversational ability it had after initial pretraining.</p>

<p><strong>Q79. Explain domain adaptation challenges.</strong><br/>
Source and target domains have different input/label distributions — a model must adapt without labeled target data (unsupervised domain adaptation) or with very few labels, violating the usual train/test-same-distribution assumption.<br/>
<em>Example:</em> A sentiment classifier trained on movie reviews applied to restaurant reviews — "cheesy" is negative for movies but the vocabulary/cues differ enough to need adaptation.</p>

<p><strong>Q80. What is zero-shot learning?</strong><br/>
The model performs a task or recognizes a class it never saw labeled examples of, typically leveraging auxiliary semantic information (class descriptions, embeddings, a shared representation space) learned from related tasks.<br/>
<em>Example:</em> CLIP can classify an image as "a photo of a zebra" even without zebra images in its classification training set, because it learned a joint image-text embedding space from massive web data.</p>

<p><strong>Q81. Why does class imbalance affect gradient updates?</strong><br/>
With standard cross-entropy, the majority class's gradient contribution dominates the total gradient simply because there are far more majority examples, so the model optimizes mostly for majority-class accuracy.
<pre><code>import torch.nn as nn
class_weights = torch.tensor([1.0, 99.0])   # inverse of class frequency (99:1 imbalance)
criterion = nn.CrossEntropyLoss(weight=class_weights)</code></pre></p>

<p><strong>Q82. Explain focal loss and its advantage.</strong><br/>
Focal loss adds a (1-p)^γ modulating factor to cross-entropy, down-weighting easy, well-classified examples and focusing training on hard/misclassified ones — useful for extreme class imbalance (e.g., object detection background vs foreground).
<pre><code>import torch
import torch.nn.functional as F

def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    return (alpha * (1 - p_t) ** gamma * ce).mean()</code></pre></p>

<p><strong>Q83. What are adversarial examples?</strong><br/>
Inputs deliberately perturbed by a small, often imperceptible amount specifically to fool a model into misclassification, exploiting locally linear/fragile decision boundaries.
<pre><code>x.requires_grad = True
loss = criterion(model(x), true_label)
loss.backward()
x_adv = x + epsilon * x.grad.sign()   # FGSM attack: nudge along the gradient sign</code></pre></p>

<p><strong>Q84. Why are neural networks vulnerable to small perturbations?</strong><br/>
High-dimensional linear-ish decision boundaries mean a tiny perturbation aligned with the gradient direction can accumulate across many dimensions into a large change in the output, even though each individual input change is imperceptible.<br/>
<em>Example:</em> An FGSM attack perturbs every pixel by just ε=0.01 in the loss-gradient direction — individually invisible, but summed across 150,000 pixels, enough to flip the prediction.</p>

<p><strong>Q85. Explain interpretability challenges in deep models.</strong><br/>
Millions of parameters interact nonlinearly across many layers, so there's no simple mapping from "this weight" to "this decision" — unlike linear models, deep model explanations are approximations that can be unfaithful.<br/>
<em>Example:</em> Two explanation methods (Grad-CAM vs saliency maps) on the same image classification can highlight completely different regions as "important."</p>

<p><strong>Q86. What is SHAP or LIME conceptually?</strong><br/>
LIME approximates a complex model locally around one prediction with an interpretable linear model. SHAP assigns each feature a contribution based on Shapley values from game theory, guaranteeing contributions sum exactly to the prediction, with consistency guarantees LIME lacks.<br/>
<em>Example:</em> For a loan denial, SHAP might show: base rate 60% approval, −25% low credit score, −10% short employment history, +5% high income — summing exactly to the actual 30% predicted probability.</p>

<p><strong>Q87. Why do embeddings capture semantic meaning?</strong><br/>
Embeddings are trained so words/items appearing in similar contexts end up with similar vectors (distributional hypothesis) — proximity in embedding space then reflects semantic similarity.
<pre><code>import torch.nn as nn
import torch.nn.functional as F

embedding = nn.Embedding(num_embeddings=50000, embedding_dim=128)
king, man, woman = embedding(torch.tensor([101, 205, 310]))
similarity = F.cosine_similarity((king - man + woman).unsqueeze(0), king.unsqueeze(0))</code></pre></p>

<p><strong>Q88. What happens when embedding size is too large?</strong><br/>
Overfitting risk increases (more parameters to memorize training-specific patterns, especially for rare items), training slows, and memory grows — with diminishing or even negative returns on downstream accuracy past a certain size.<br/>
<em>Example:</em> A 2048-dim embedding for a recommender with only 5,000 unique items likely overfits and wastes memory compared to a well-tuned 64 or 128-dim embedding.</p>

<p><strong>Q89. Explain multi-task learning benefits and risks.</strong><br/>
Sharing representations across related tasks can improve generalization (implicit regularization) — but risks negative transfer if tasks conflict, and needs careful loss-weighting since one task's gradient can dominate and hurt the others.
<pre><code>total_loss = w1 * detection_loss + w2 * depth_loss + w3 * segmentation_loss
# if w1 is too large relative to w2, w3, those tasks get starved of useful gradient</code></pre></p>

<p><strong>Q90. Why do some models memorize instead of generalize?</strong><br/>
When model capacity vastly exceeds what's needed to represent the true pattern and training data is limited/noisy, the optimizer can drive training loss to near zero by literally storing training examples rather than learning generalizable rules.<br/>
<em>Example:</em> A large network trained on CIFAR-10 with completely randomized labels can still reach ~100% training accuracy — memorizing arbitrary label-image pairs (Zhang et al., "Understanding Deep Learning Requires Rethinking Generalization").</p>

<h3>Real-World &amp; System Design (Q91–Q100)</h3>

<p><strong>Q91. How would you design a scalable training pipeline?</strong><br/>
Ingest data via a scalable store → feature engineering with a versioned feature store → distributed training (data/model parallelism) with checkpointing → experiment tracking (MLflow/W&amp;B) → automated evaluation gate before promotion to production.<br/>
<em>Example:</em> A pipeline using Spark for feature computation, S3 as the data lake, PyTorch DDP across 8 GPUs for training, and MLflow logging every run's hyperparameters/metrics so any past run is reproducible.</p>

<p><strong>Q92. How do you handle missing data in production systems?</strong><br/>
Reuse the exact imputation logic fit during training (never recompute stats on live data), add missing-value indicator features, set sensible defaults, and monitor the missing-rate — a spike usually signals an upstream pipeline bug, not organic missingness.<br/>
<em>Example:</em> If "age" is missing for 40% of live requests when training data had only 2% missing, that signals the production feature pipeline broke, not that users stopped reporting age.</p>

<p><strong>Q93. Explain feature drift vs data drift.</strong><br/>
Data drift (covariate shift) is any change in input feature distribution P(x). Feature drift often refers to individual feature distributions shifting, sometimes due to a change in how the feature is even defined/computed. Both can occur without changing P(y|x) — that's concept drift.<br/>
<em>Example:</em> If "user age" suddenly shows values in months instead of years due to an upstream unit change, that's feature drift — a definitional break, not a real demographic shift.</p>

<p><strong>Q94. How would you monitor model performance post-deployment?</strong><br/>
Track live prediction distributions, input feature drift (PSI/KS tests), delayed ground-truth labels when available, business KPIs tied to the model's decisions, and alerting thresholds on all of the above.<br/>
<em>Example:</em> A fraud model dashboard tracks daily flagged-transaction rate, feature PSI scores, and (once true labels arrive weeks later) actual precision/recall — alerting if the flagged rate suddenly jumps 5× with no external driver.</p>

<p><strong>Q95. What steps ensure reproducibility in training?</strong><br/>
Fix random seeds (data shuffling, weight init, dropout), pin exact library/dependency versions, version the training data snapshot, log all hyperparameters, and use deterministic operations where supported.
<pre><code>import torch, random, numpy as np
torch.manual_seed(42); random.seed(42); np.random.seed(42)
torch.use_deterministic_algorithms(True)</code></pre></p>

<p><strong>Q96. How do you reduce latency in model inference?</strong><br/>
Quantization (fp32→int8), knowledge distillation to a smaller student model, pruning, batching requests, optimized runtimes (ONNX Runtime, TensorRT), and caching frequent predictions.
<pre><code>import torch
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)   # typically ~2-4x faster inference, small accuracy drop</code></pre></p>

<p><strong>Q97. Explain trade-offs between accuracy and speed.</strong><br/>
More accurate models are usually larger/deeper/ensembled, directly increasing inference latency and serving cost; the right operating point depends on the SLA and how much marginal accuracy actually moves the business metric.<br/>
<em>Example:</em> Going from a 2-model ensemble (AUC 0.91, 200ms) to a single distilled model (AUC 0.89, 20ms) is the right trade for a real-time bidding system with a strict 30ms budget, even though it's "less accurate" on paper.</p>

<p><strong>Q98. How do you debug a model that suddenly degrades?</strong><br/>
Check for upstream data pipeline breakage first (most common real cause), compare current vs historical feature distributions, verify no silent schema/library version change, check for concept drift, and confirm serving code matches training code (training-serving skew).<br/>
<em>Example:</em> A recommendation model's CTR crashes overnight — the investigation finds an upstream ETL job started sending null instead of 0 for a "click_count" feature after a schema migration.</p>

<p><strong>Q99. What are common pitfalls in deploying deep models?</strong><br/>
Training-serving skew (different preprocessing code paths), ignoring latency/memory constraints of the serving environment, no plan for model versioning/rollback, insufficient monitoring, and not testing edge cases before launch.<br/>
<em>Example:</em> A model trained with images resized via PIL's bicubic interpolation gets served with OpenCV's default resizing — a subtle preprocessing mismatch that silently degrades every prediction in production.</p>

<p><strong>Q100. How would you redesign a failing ML system end-to-end?</strong><br/>
(1) Re-anchor on the actual business metric, not just the offline ML metric. (2) Audit the data pipeline for leakage/drift/quality issues. (3) Re-evaluate whether the model architecture/objective matches the real problem. (4) Rebuild evaluation to reflect production conditions. (5) Add continuous monitoring + a feedback loop so failures are caught early next time.<br/>
<em>Example:</em> A "successful" churn model (AUC 0.85) that never reduced actual churn gets redesigned around the real KPI (retained revenue), re-trained on properly time-split data, and shipped with a shadow-mode test before it's allowed to trigger real retention offers.</p>
`
  },
];
