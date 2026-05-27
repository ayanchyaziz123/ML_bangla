export const interview_1_100 = {
  slug: 'interview-1-100',
  title: '১০০টি ML ইন্টারভিউ প্রশ্ন: সবচেয়ে বেশি জিজ্ঞেস করা হয়',
  description: 'Google, Meta, Amazon, Microsoft-এর ML ইন্টারভিউতে সবচেয়ে বেশি জিজ্ঞেস করা ১০০টি প্রশ্ন এবং সংক্ষিপ্ত উত্তর — ML Theory, Statistics, DL, Coding, System Design।',
  date: 'মে ২০২৬',
  category: 'এমএল ইন্টারভিউ প্রশ্ন',
  readTime: 25,
  content: `
    <h3>ML Fundamentals (Q1–Q25)</h3>

    <p><strong>Q1. Bias-Variance Tradeoff কী?</strong><br/>
    Bias = model assumption থেকে error (underfitting)। Variance = data-এর noise-এর প্রতি sensitivity (overfitting)। Total Error = Bias² + Variance + Irreducible Noise। Complex model → low bias, high variance; Simple model → high bias, low variance।</p>

    <p><strong>Q2. Overfitting কীভাবে prevent করবেন?</strong><br/>
    Regularization (L1/L2), Dropout, Early stopping, Data augmentation, Cross-validation, Simpler model, More training data।</p>

    <p><strong>Q3. L1 vs L2 Regularization-এর পার্থক্য?</strong><br/>
    L1 (Lasso): |w| penalty → sparse weights, feature selection করে। L2 (Ridge): w² penalty → weights ছোট কিন্তু zero হয় না। Elastic Net = L1 + L2 combination।</p>

    <p><strong>Q4. Cross-Validation কী এবং কেন দরকার?</strong><br/>
    Training data-কে k folds-এ ভাগ করে k বার train/validate করা। Test set-এ performance estimate করার unbiased পদ্ধতি। Prevents overfitting to validation set। K=5 বা K=10 common।</p>

    <p><strong>Q5. Precision vs Recall vs F1?</strong><br/>
    Precision = TP/(TP+FP) — যা positive predict করেছি তার কতটা সত্যিই positive।<br/>
    Recall = TP/(TP+FN) — সব actual positive-এর কতটা detect করেছি।<br/>
    F1 = 2×P×R/(P+R) — harmonic mean। Medical diagnosis-এ high recall critical।</p>

    <p><strong>Q6. Class Imbalance কীভাবে handle করবেন?</strong><br/>
    Oversampling (SMOTE), Undersampling, Class weights, Threshold tuning, Anomaly detection approach, precision-recall curve ব্যবহার (ROC-AUC নয়)।</p>

    <p><strong>Q7. Feature Scaling কেন দরকার?</strong><br/>
    Distance-based (KNN, SVM) এবং gradient descent algorithms (LR, NN) scale sensitive। Tree-based methods (RF, XGBoost) scaling-এর দরকার নেই। StandardScaler (z-score) বা MinMaxScaler ব্যবহার করুন।</p>

    <p><strong>Q8. Missing Values কীভাবে handle করবেন?</strong><br/>
    Remove rows/columns (data কম হলে), Mean/Median/Mode imputation, KNN imputation, Model-based imputation (IterativeImputer), Missing indicator feature যোগ করা, Tree models naturally handle করে।</p>

    <p><strong>Q9. Gradient Descent-এর variants কী কী?</strong><br/>
    Batch GD: সব data দিয়ে update (slow, stable)। SGD: একটি sample দিয়ে (fast, noisy)। Mini-batch: batch_size samples দিয়ে (best of both)। Adam = momentum + adaptive learning rates।</p>

    <p><strong>Q10. ROC-AUC কী?</strong><br/>
    Receiver Operating Characteristic curve: TPR vs FPR at all thresholds। AUC = Area Under Curve। AUC=0.5: random; AUC=1.0: perfect। Class imbalance-এ Precision-Recall AUC বেশি informative।</p>

    <p><strong>Q11. Ensemble Methods কী কী?</strong><br/>
    Bagging: parallel models + average (Random Forest)। Boosting: sequential, each corrects previous (XGBoost, AdaBoost)। Stacking: meta-model learns from base models। Boosting সাধারণত better performance দেয়।</p>

    <p><strong>Q12. Decision Tree-এ splitting criterion কী কী?</strong><br/>
    Gini Impurity: 1 - Σpᵢ² (classification, faster)। Information Gain / Entropy: -Σpᵢlog(pᵢ) (classification)। MSE / MAE (regression)। Gini এবং Entropy সাধারণত same results দেয়।</p>

    <p><strong>Q13. Random Forest vs XGBoost?</strong><br/>
    Random Forest: bagging + feature randomness, parallel training, robust to outliers, less hyperparameters। XGBoost: gradient boosting, sequential, regularization built-in, better performance, more hyperparameters। Tabular data-এ XGBoost often wins।</p>

    <p><strong>Q14. Curse of Dimensionality কী?</strong><br/>
    High dimensional space-এ data points sparse হয়ে যায়। Distance metrics অর্থহীন হয়। KNN, k-means degrade করে। Feature selection/reduction (PCA, UMAP) দিয়ে মোকাবেলা করা হয়।</p>

    <p><strong>Q15. PCA কীভাবে কাজ করে?</strong><br/>
    Covariance matrix-এর eigenvectors বের করে। Maximum variance direction = first principal component। Orthogonal components → redundancy remove করে। n_components দিয়ে explained variance ≥ 95% রাখুন।</p>

    <p><strong>Q16. SVM-এর kernel trick কী?</strong><br/>
    Non-linearly separable data-কে higher dimensional space-এ project করে linear separation সম্ভব করে। RBF kernel most common। C = margin vs misclassification tradeoff। γ = decision boundary smoothness।</p>

    <p><strong>Q17. Naive Bayes কেন "Naive"?</strong><br/>
    Features conditional independence assume করে given class label। বাস্তবে প্রায়ই false, তবুও text classification-এ surprisingly effective। Fast, works well with small data।</p>

    <p><strong>Q18. Train/Validation/Test split-এর best practice?</strong><br/>
    80/10/10 বা 70/15/15। Test set শুধু final evaluation-এ একবার। Time series-এ temporal split করুন। Stratified split করুন class imbalance থাকলে। Test set কখনো hyperparameter tuning-এ use করবেন না।</p>

    <p><strong>Q19. Data Leakage কী? কীভাবে prevent করবেন?</strong><br/>
    Test/future data information training-এ চলে আসা। Types: target leakage (future feature), train-test contamination (scaling before split)। Fix: সবসময় split করার পরে preprocessing। Pipeline ব্যবহার করুন।</p>

    <p><strong>Q20. Model selection-এ AIC/BIC কী?</strong><br/>
    AIC = 2k - 2ln(L): Akaike Information Criterion। BIC = k×ln(n) - 2ln(L): Bayesian IC। উভয়ই model complexity penalize করে। Lower = better। BIC larger penalty for complex models।</p>

    <p><strong>Q21. Clustering-এ optimal K কীভাবে বেছে নেবেন?</strong><br/>
    Elbow method: inertia vs K plot। Silhouette score: cohesion vs separation (-1 to 1, higher better)। Gap statistic। Domain knowledge। DBSCAN density-based → K specify করতে হয় না।</p>

    <p><strong>Q22. Anomaly detection-এর approaches?</strong><br/>
    Statistical (z-score, IQR)। Distance-based (LOF)। Tree-based (Isolation Forest)। Autoencoder (reconstruction error)। One-class SVM। Supervised (যদি labeled anomaly data থাকে)।</p>

    <p><strong>Q23. Feature Importance কীভাবে পাবেন?</strong><br/>
    Tree-based: Gini impurity decrease (fast, biased toward high-cardinality)। Permutation importance (model-agnostic, test set-এ, unbiased)। SHAP values (additive, consistent, gold standard)। LIME (local explanation)।</p>

    <p><strong>Q24. Hyperparameter tuning-এর approaches?</strong><br/>
    Grid Search (exhaustive, slow)। Random Search (efficient, Bergstra & Bengio 2012)। Bayesian Optimization (smart, Gaussian Process surrogate)। Optuna/Hyperopt (TPE algorithm)। Random search often beats grid search।</p>

    <p><strong>Q25. Model evaluation metric কীভাবে বেছে নেবেন?</strong><br/>
    Classification binary: AUC-ROC (balanced), F1 (imbalanced), Precision (FP costly), Recall (FN costly)। Multi-class: Macro/Weighted F1। Regression: RMSE (outlier sensitive), MAE (robust), MAPE (relative error)।</p>

    <h3>Deep Learning (Q26–Q50)</h3>

    <p><strong>Q26. Backpropagation কীভাবে কাজ করে?</strong><br/>
    Chain rule দিয়ে loss-এর gradient প্রতিটি weight-এর সাথে সম্পর্কে calculate করা। Forward pass: prediction। Backward pass: chain rule দিয়ে gradients। Weight update: w = w - lr × ∂L/∂w।</p>

    <p><strong>Q27. Vanishing/Exploding Gradient সমস্যা?</strong><br/>
    Deep networks-এ gradients exponentially ছোট (vanish) বা বড় (explode) হয়। Solutions: ReLU activation, Batch Normalization, Residual connections (ResNet), Gradient clipping, LSTM gating।</p>

    <p><strong>Q28. Activation functions তুলনা করুন।</strong><br/>
    Sigmoid: (0,1) range, vanishing gradient সমস্যা। Tanh: (-1,1), zero-centered। ReLU: fast, dying ReLU সমস্যা (negative → dead neuron)। Leaky ReLU: dying ReLU fix। GELU: Transformer-এ popular।</p>

    <p><strong>Q29. Batch Normalization কেন কাজ করে?</strong><br/>
    প্রতিটি mini-batch-এর activation normalize করে। Internal covariate shift কমায়। Higher learning rates সম্ভব করে। Slight regularization effect। Dropout-এর সাথে একসাথে ব্যবহারে সতর্ক থাকুন।</p>

    <p><strong>Q30. Dropout কীভাবে কাজ করে?</strong><br/>
    Training-এ random neurons zero করা (p probability)। Inference-এ সব neurons active, weights × (1-p)। Ensemble effect: many sub-networks। Overfitting prevent করে। Typically p=0.2-0.5।</p>

    <p><strong>Q31. CNN-এ Convolution vs Pooling?</strong><br/>
    Convolution: learned filters দিয়ে feature extraction (edge, texture)। Pooling: spatial dimension কমানো। Max pooling: dominant feature preserve। Average pooling: global context। Stride convolution pooling replace করতে পারে।</p>

    <p><strong>Q32. Transfer Learning কী?</strong><br/>
    Pre-trained model (ImageNet, BERT)-এর weights নিজের task-এ reuse করা। Feature extraction: backbone frozen, শুধু head train। Fine-tuning: সব weights train (more data দরকার)। Few-shot learning-এর জন্য ideal।</p>

    <p><strong>Q33. Attention Mechanism কীভাবে কাজ করে?</strong><br/>
    Query, Key, Value vectors। Score = softmax(QKᵀ/√d_k)। Output = Score × V। Self-attention: একই sequence-এর tokens interact করে। Multi-head: multiple attention subspaces simultaneously।</p>

    <p><strong>Q34. Transformer vs LSTM?</strong><br/>
    LSTM: sequential, captures order naturally, slower training। Transformer: parallel, global attention, better for long sequences, needs more data, scales better। Modern NLP mostly Transformer।</p>

    <p><strong>Q35. BERT vs GPT?</strong><br/>
    BERT: Encoder-only, bidirectional, masked language modeling → good for classification, NER। GPT: Decoder-only, autoregressive, causal → good for generation। T5: Encoder-Decoder, text-to-text।</p>

    <p><strong>Q36. Positional Encoding কেন দরকার?</strong><br/>
    Transformer-এর attention position-invariant। Positional encoding token-এর position সম্পর্কে তথ্য যোগ করে। Sine/cosine (original) বা learned embeddings। RoPE (rotary) modern models-এ popular।</p>

    <p><strong>Q37. ResNet-এর Residual Connection কী করে?</strong><br/>
    Skip connection: F(x) + x। Gradient highway তৈরি করে — deeper layers-ও gradient পায়। Identity mapping সহজ — network শুধু residual শেখে। 100+ layer training সম্ভব করে।</p>

    <p><strong>Q38. GAN training কেন unstable?</strong><br/>
    Generator ও Discriminator-এর competing objectives। Mode collapse: generator একটি ভালো sample-ই repeat করে। Vanishing gradient যখন discriminator too good। Fix: WGAN, spectral normalization, progressive training।</p>

    <p><strong>Q39. Autoencoder এবং VAE-র পার্থক্য?</strong><br/>
    Autoencoder: deterministic latent space, generation নয়। VAE: latent space = Gaussian distribution (μ, σ)। Reparameterization trick। KL divergence regularizer। VAE smooth latent space → generation সম্ভব।</p>

    <p><strong>Q40. Learning Rate Scheduling?</strong><br/>
    Step decay, Cosine annealing, Warmup + decay (Transformers)। Cyclical LR (CLR)। OneCycleLR। Too high LR → diverge। Too low → slow convergence। Adam often less sensitive than SGD।</p>

    <p><strong>Q41. Weight Initialization কেন গুরুত্বপূর্ণ?</strong><br/>
    All zeros → symmetry breaking হয় না (useless)। Xavier/Glorot: tanh/sigmoid-এর জন্য। He initialization: ReLU-র জন্য (variance × 2/fan_in)। Bad init → vanishing/exploding gradients।</p>

    <p><strong>Q42. LoRA কীভাবে কাজ করে?</strong><br/>
    Weight update ΔW = BA (low-rank)। B (d×r) এবং A (r×d), r &lt;&lt; d। Full W frozen, শুধু A ও B train হয়। VRAM ~3x কম। Multiple adapters swap করা যায়। QLoRA = 4-bit quantization + LoRA।</p>

    <p><strong>Q43. Tokenization-এর types?</strong><br/>
    Word-level: OOV problem। Character-level: long sequences। Subword (BPE, WordPiece, SentencePiece): best of both। BPE: merge frequent pairs। WordPiece: maximize language model likelihood।</p>

    <p><strong>Q44. Embeddings vs One-Hot Encoding?</strong><br/>
    One-hot: sparse, high-dimensional, no semantic relationship। Embedding: dense, low-dimensional, semantic similarity captured। Word2Vec, GloVe, contextual (BERT)। Pretrained embeddings transfer learning-এ ব্যবহার।</p>

    <p><strong>Q45. Contrastive Learning কী?</strong><br/>
    Similar samples-কে embedding space-এ কাছে, dissimilar-কে দূরে আনা। SimCLR: augmented views positive pairs। NT-Xent loss। Self-supervised: labels ছাড়া শেখা। CLIP: image-text pairs।</p>

    <p><strong>Q46. Knowledge Distillation কী?</strong><br/>
    Large "teacher" model-এর knowledge ছোট "student" model-এ transfer করা। Soft labels (temperature-scaled probabilities) use হয়। Student learns teacher-এর generalization। Production deployment-এ efficient।</p>

    <p><strong>Q47. Sequence-to-Sequence model?</strong><br/>
    Encoder processes input sequence → context vector। Decoder generates output sequence। Attention added (Neural Machine Translation)। Transformer এটি parallel করে। T5, BART এই architecture।</p>

    <p><strong>Q48. Object Detection-এর approaches?</strong><br/>
    Two-stage: R-CNN family (Region Proposal + Classification) → accurate, slow। One-stage: YOLO, SSD (direct prediction) → fast, less accurate। Anchor-free: FCOS, CenterNet। YOLO সাধারণত production favorite।</p>

    <p><strong>Q49. Graph Neural Networks কী করে?</strong><br/>
    Node features + graph structure থেকে শেখে। Message passing: neighbors-এর information aggregate করে। GCN, GraphSAGE, GAT (attention)। Application: social network, molecular properties, recommendation।</p>

    <p><strong>Q50. Diffusion Models কীভাবে কাজ করে?</strong><br/>
    Forward process: data-তে gradually noise add করা (T steps)। Reverse process: noise থেকে data reconstruct করতে শেখা। DDPM, Stable Diffusion। Score matching দিয়ে gradient of log-density শেখা।</p>

    <h3>Statistics & Math (Q51–Q65)</h3>

    <p><strong>Q51. p-value কী? Misinterpretation?</strong><br/>
    H₀ সত্য ধরলে observed data পাওয়ার probability। Common mistake: "model correct হওয়ার probability" নয়। p &lt; 0.05 মানে effect "significant", কিন্তু practically meaningful নাও হতে পারে।</p>

    <p><strong>Q52. Confidence Interval vs Credible Interval?</strong><br/>
    95% CI: যদি experiment বারবার করি, 95% CI-তে true parameter থাকবে (frequentist)। Credible interval: "true parameter এই range-এ আছে probability 95%" (Bayesian)। Credible interval more intuitive।</p>

    <p><strong>Q53. CLT কী এবং ML-এ কোথায় ব্যবহার হয়?</strong><br/>
    যেকোনো distribution থেকে sample mean large n-এ normal হয়। Hypothesis tests, confidence intervals, SGD convergence analysis-এর ভিত্তি।</p>

    <p><strong>Q54. Expected Value ও Variance-এর properties?</strong><br/>
    E[aX+b] = aE[X]+b। Var[aX] = a²Var[X]। Independent হলে: E[XY]=E[X]E[Y], Var[X+Y]=Var[X]+Var[Y]।</p>

    <p><strong>Q55. Covariance vs Correlation?</strong><br/>
    Covariance: E[(X-μx)(Y-μy)] — scale dependent। Correlation: Cov(X,Y)/(σxσy) — scale-free, [-1,1]। Correlation শুধু linear relationship। Pearson (linear), Spearman (rank-based, non-linear)।</p>

    <p><strong>Q56. MLE কী?</strong><br/>
    Data-কে সবচেয়ে ভালো explain করে এমন parameters খোঁজা: argmax P(data|θ)। Log-likelihood maximize করা। Linear regression MSE = Gaussian MLE। Logistic regression cross-entropy = Bernoulli MLE।</p>

    <p><strong>Q57. Overfitting-এর statistical explanation?</strong><br/>
    High variance model: training data-এর noise fit করে। Bias-variance decomposition: E[(y-ŷ)²] = Bias² + Variance + Noise। Regularization variance কমায় (bias বাড়িয়ে)।</p>

    <p><strong>Q58. Type I vs Type II Error tradeoff?</strong><br/>
    α কমালে (stricter) → Type I error কমে, Type II বাড়ে। Medical testing-এ Type II costly (miss disease)। Spam filter-এ Type I costly (block legit email)। Task অনুযায়ী threshold adjust।</p>

    <p><strong>Q59. Bootstrap কী?</strong><br/>
    Replacement-সহ resample করে statistics-এর distribution estimate। Small sample-এ powerful। Confidence intervals, hypothesis tests। Parametric assumptions ছাড়াই কাজ করে।</p>

    <p><strong>Q60. Information Theory: Entropy ও KL Divergence?</strong><br/>
    Entropy H(p) = -Σp log p: uncertainty/surprise। Cross-entropy H(p,q) = -Σp log q: classification loss। KL Divergence = H(p,q) - H(p): distributions-এর পার্থক্য। VAE-তে KL divergence regularizer।</p>

    <p><strong>Q61. Eigenvalues ও PCA-র সম্পর্ক?</strong><br/>
    Covariance matrix-এর eigenvectors = principal components। Eigenvalues = explained variance। Top k eigenvectors → k principal components। SVD দিয়েও efficiently compute করা যায়।</p>

    <p><strong>Q62. Gradient ও Hessian?</strong><br/>
    Gradient: first-order partial derivatives (direction of steepest ascent)। Hessian: second-order partial derivatives (curvature)। Newton's method: Hessian inverse দিয়ে update → faster convergence, expensive।</p>

    <p><strong>Q63. Convexity কেন ML-এ গুরুত্বপূর্ণ?</strong><br/>
    Convex function-এ local minimum = global minimum। Logistic regression loss convex। Neural network loss non-convex। SGD often finds good solutions despite non-convexity (overparameterization helps)।</p>

    <p><strong>Q64. Monte Carlo Methods?</strong><br/>
    Random sampling দিয়ে numerical integration/estimation। MCMC (Markov Chain Monte Carlo): complex posterior sampling। Bayesian inference, RL (policy evaluation)। Law of large numbers: accuracy বাড়ে sample দিয়ে।</p>

    <p><strong>Q65. Mutual Information?</strong><br/>
    I(X;Y) = H(X) + H(Y) - H(X,Y): X ও Y-র shared information। Feature selection-এ ব্যবহার। Zero = independent। MI কে I(X;Y) = KL(P(X,Y) || P(X)P(Y)) হিসেবেও লেখা যায়।</p>

    <h3>Data & Coding (Q66–Q80)</h3>

    <p><strong>Q66. Pandas-এ large dataset efficiently handle?</strong><br/>
    chunksize দিয়ে read। dtype specify (int32 vs int64)। category dtype for low-cardinality strings। Dask for out-of-memory। query() faster than boolean indexing। avoid apply() loops, use vectorized ops।</p>

    <p><strong>Q67. SQL vs NoSQL কোনটি ML pipeline-এ?</strong><br/>
    SQL (PostgreSQL, BigQuery): structured data, complex queries, ACID। NoSQL (MongoDB, Cassandra): flexible schema, horizontal scale। Feature store (Feast, Tecton) ML-specific। Data warehouse (Snowflake) analytics-এ।</p>

    <p><strong>Q68. Python-এ memory-efficient code লেখার tips?</strong><br/>
    Generators yield করুন (list এর বদলে)। numpy arrays prefer করুন। del + gc.collect()। Context managers। Float32 vs Float64। Sparse matrices (scipy.sparse) for sparse data।</p>

    <p><strong>Q69. Vectorization কী?</strong><br/>
    Loops-এর বদলে numpy/pandas array operations। SIMD CPU instructions ব্যবহার। 10-100x faster। np.dot() vs manual loop। আপনার custom loops avoid করুন — numpy/scipy functions ব্যবহার।</p>

    <p><strong>Q70. Feature Engineering best practices?</strong><br/>
    Datetime → year, month, day, hour, day_of_week, is_weekend। Categorical → one-hot, target encoding, embedding। Numeric → log transform (skewed), binning, interactions। Text → TF-IDF, embeddings।</p>

    <p><strong>Q71. EDA (Exploratory Data Analysis) workflow?</strong><br/>
    shape, dtypes, describe(), isnull(). Distribution plots। Correlation matrix। Target distribution। Outlier detection। Feature vs target relationship। Class balance check।</p>

    <p><strong>Q72. Target Encoding vs One-Hot Encoding?</strong><br/>
    One-hot: high cardinality-তে dimensionality explodes। Target encoding: category → mean of target। Leakage সম্ভব → use cross-validation target encoding। Leave-one-out বা Bayesian smoothing use করুন।</p>

    <p><strong>Q73. Time Series data-এ train/test split?</strong><br/>
    Random split করবেন না — temporal leakage হবে। Walk-forward validation: expanding window বা sliding window। Test = last N days। Feature: lag features আগের values থেকে create।</p>

    <p><strong>Q74. Data versioning কেন দরকার?</strong><br/>
    Reproducibility: same data → same results। Debugging: কোন data version-এ performance drop হয়েছিল। DVC (Data Version Control), Delta Lake, MLflow artifacts।</p>

    <p><strong>Q75. Efficient model serialization?</strong><br/>
    sklearn: joblib (pickle-এর চেয়ে দ্রুত numpy arrays-এ)। PyTorch: torch.save state_dict (model এর বদলে)। ONNX: framework-agnostic format। TensorRT: GPU inference optimization।</p>

    <p><strong>Q76. Confusion Matrix interpret করুন।</strong><br/>
    TP: correctly predicted positive। TN: correctly predicted negative। FP: false alarm (Type I)। FN: missed detection (Type II)। High FP costly? → raise threshold। High FN costly? → lower threshold।</p>

    <p><strong>Q77. SMOTE কীভাবে কাজ করে?</strong><br/>
    Minority class-এর synthetic samples তৈরি করে। KNN দিয়ে nearest neighbors খোঁজে। দুটি real minority samples-এর মধ্যে interpolate করে নতুন sample তৈরি। Oversampling-এর smarter version।</p>

    <p><strong>Q78. Stratified sampling কী?</strong><br/>
    Class distribution maintain রেখে random sampling। train_test_split(stratify=y)। Class imbalance-এ বিশেষভাবে important। Cross-validation-এও stratify করুন।</p>

    <p><strong>Q79. Pipeline কেন ব্যবহার করবেন?</strong><br/>
    Data leakage prevent করে (fit শুধু training data-তে)। Code reuse। cross_val_score ও GridSearchCV সাথে সহজে কাজ করে। Deployment-এ training-inference consistency নিশ্চিত করে।</p>

    <p><strong>Q80. Feature Store কী?</strong><br/>
    Centralized repository for ML features। Online store: low-latency inference। Offline store: batch training। Feature reuse across teams। Feast, Tecton, Vertex AI Feature Store popular options।</p>

    <h3>MLOps & System Design (Q81–Q100)</h3>

    <p><strong>Q81. ML System Design-এ কী কী consider করবেন?</strong><br/>
    Scale (traffic, data size), Latency requirement, Consistency vs availability, Online vs offline inference, Model update frequency, Training infrastructure, Monitoring strategy, A/B testing plan।</p>

    <p><strong>Q82. Model Monitoring কেন দরকার?</strong><br/>
    Data drift: input distribution পরিবর্তন। Concept drift: target relationship পরিবর্তন। Performance degradation। Evidently, WhyLabs, Fiddler monitoring tools। KS test, PSI (Population Stability Index)।</p>

    <p><strong>Q83. Online vs Batch inference?</strong><br/>
    Online: real-time (recommendations, fraud detection), low latency crucial, scalability challenge। Batch: periodic (email campaigns, reports), higher throughput, cost efficient। Streaming: near real-time (Kafka, Flink)।</p>

    <p><strong>Q84. Model versioning best practices?</strong><br/>
    MLflow, Weights &amp; Biases। Track: model file, parameters, metrics, data version, code version (git hash)। Staging → Production promotion। Rollback capability critical।</p>

    <p><strong>Q85. Canary deployment কী?</strong><br/>
    New model-কে small % traffic (1-5%) দিয়ে test করা। Metrics monitor করা। Gradually ramp up। Problem হলে rollback। Blue-green deployment: instant switch। Shadow mode: parallel run without serving.</p>

    <p><strong>Q86. Recommendation System design করুন।</strong><br/>
    Two-stage: Retrieval (Two-Tower + FAISS) → Ranking (Wide&amp;Deep/DCN)। Features: user history, item attributes, context। Cold start: popularity, content-based। Evaluation: NDCG@K, Precision@K।</p>

    <p><strong>Q87. ML Pipeline-এ CI/CD?</strong><br/>
    Code tests (unit, integration)। Data validation (Great Expectations)। Model validation (performance threshold)। Auto-deployment if tests pass। GitHub Actions, Jenkins, Kubeflow Pipelines।</p>

    <p><strong>Q88. Kubernetes-এ ML model serve করার challenge?</strong><br/>
    GPU scheduling। Model loading time। Memory management (multiple models)। Auto-scaling based on queue depth। KServe (formerly KFServing) standardizes serving। gRPC vs REST tradeoffs।</p>

    <p><strong>Q89. Fairness in ML?</strong><br/>
    Demographic parity, Equal opportunity, Equalized odds। Protected attributes (race, gender)। Disparate impact testing। Fairness-accuracy tradeoff। IBM AI Fairness 360, Google's What-If Tool।</p>

    <p><strong>Q90. Explainability requirements?</strong><br/>
    EU AI Act: high-risk AI must be explainable। SHAP global ও local explanations। LIME model-agnostic। Counterfactual explanations ("what would change the decision?")। Inherently interpretable models (logistic regression, decision tree)।</p>

    <p><strong>Q91. Cold Start Problem?</strong><br/>
    New user: no history → popularity-based বা demographic-based। New item: no interactions → content-based features। Matrix factorization fails। Solution: hybrid, side information (metadata), exploration-exploitation।</p>

    <p><strong>Q92. Feature Store Online/Offline consistency?</strong><br/>
    Training-serving skew: training এবং inference-এ different feature values। Fix: same feature computation logic। Lambda architecture: batch + stream। Point-in-time correct features (no future leakage)।</p>

    <p><strong>Q93. Distributed training কীভাবে কাজ করে?</strong><br/>
    Data parallelism: same model, different data shards। Model parallelism: different layers, different GPUs। Parameter Server: central gradient aggregation। AllReduce (Horovod, NCCL): peer-to-peer efficient।</p>

    <p><strong>Q94. Quantization কী?</strong><br/>
    Float32 → Int8 বা Float16 (less bits)। Model size ও inference speed improve। Post-training quantization vs quantization-aware training। Edge deployment-এ important। TensorRT, ONNX Runtime।</p>

    <p><strong>Q95. A/B testing এর limitations?</strong><br/>
    Network effects (social platforms)। Long-term effects miss হয়। Novelty effect। Multi-armed bandit: exploration/exploitation balance করে। Holdout groups for long-term measurement।</p>

    <p><strong>Q96. Shadow Mode Testing?</strong><br/>
    New model production traffic দেখে কিন্তু serve করে না। Real distribution-এ test। Zero user impact। Latency ও resource ব্যবহার measure। Deploy করার আগে এটি করুন।</p>

    <p><strong>Q97. Multi-armed Bandit vs A/B testing?</strong><br/>
    A/B testing: fixed exploration period, then exploit winner (slow)। Bandit: simultaneously explore ও exploit। Thompson Sampling, UCB algorithms। Faster convergence, less regret।</p>

    <p><strong>Q98. Embeddings কীভাবে serve করবেন?</strong><br/>
    Precompute ও store in vector DB (Pinecone, Weaviate, pgvector)। Approximate Nearest Neighbor (FAISS, ScaNN, HNSW)। Exact search too slow at scale। Cache popular queries।</p>

    <p><strong>Q99. Real-time fraud detection system design?</strong><br/>
    Low latency (&lt;100ms)। Features: transaction velocity, amount z-score, location change, merchant category। Streaming feature computation (Flink/Kafka)। Model: gradient boosting + neural network ensemble। Threshold tuning for precision/recall।</p>

    <p><strong>Q100. ML-এর নৈতিক দিক?</strong><br/>
    Bias amplification (historical data-এ bias থাকলে model তা শেখে)। Privacy (differential privacy, federated learning)। Transparency ও explainability। Environmental impact (training carbon footprint)। Human oversight for high-stakes decisions।</p>
  `
};
