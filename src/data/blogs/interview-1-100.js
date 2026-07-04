export const interview_1_100 = {
  slug: 'interview-1-100',
  title: '১০০টি ML ও Deep Learning ইন্টারভিউ প্রশ্ন (Intermediate → Hard) — উদাহরণ ও PyTorch code সহ',
  description: 'Intermediate থেকে Hard লেভেলের ১০০টি ML/DL ইন্টারভিউ প্রশ্ন — real failure case, PyTorch code example এবং system design reasoning সহ বিস্তারিত উত্তর।',
  date: 'জুলাই ২০২৬',
  category: 'এমএল ইন্টারভিউ প্রশ্ন',
  readTime: 40,
  content: `
    <h3>Core ML Concepts (Q1–Q20)</h3>

    <p><strong>Q1. Bias-Variance Tradeoff একটা real failure case দিয়ে explain করুন।</strong><br/>
    Bias = খুব simple assumption থেকে আসা error (underfitting)। Variance = training data-র প্রতি বেশি sensitive হওয়ার error (overfitting)। Total Error = Bias² + Variance + irreducible noise।<br/>
    উদাহরণ: শুধু square footage দিয়ে house price predict করা linear regression underfit করে (high bias) — location/age ignore করে। একই 200 rows-এ depth-20 decision tree training-এ R²=0.99 পায় কিন্তু test-এ R²=0.4 (high variance) — এটা noise মুখস্থ করে ফেলেছে।</p>

    <p><strong>Q2. কখন বেশি data যোগ করলেও model performance improve হয় না?</strong><br/>
    যখন model bias-limited (data যতই থাকুক, খুব simple), যখন নতুন data ভিন্ন distribution থেকে আসে, অথবা যখন data label noise-এ dominated।<br/>
    উদাহরণ: একটা sentiment classifier-এ ১০ গুণ বেশি mislabeled tweet যোগ করলে signal নয়, noise-ই reinforce হয় — accuracy কমেও যেতে পারে।</p>

    <p><strong>Q3. L1 vs L2 regularization definition-এর বাইরে গিয়ে compare করুন।</strong><br/>
    L1 (Lasso) |w| penalty যোগ করে — এর diamond-shaped constraint region-এর axis-এ corner থাকে, তাই optimization প্রায়ই exactly zero-তে গিয়ে থামে — built-in feature selection। L2 (Ridge) w² penalty যোগ করে — circular constraint region-এ কোনো corner নেই, তাই weight smoothly ছোট হয় কিন্তু exact zero হয় না।<br/>
    উদাহরণ: ২০,০০০ gene-এর genomics dataset-এ মাত্র ~৫০টা relevant হলে, L1 automatically বাকি ~১৯,৯৫০টা coefficient zero করে দেয়।
    <pre><code>import torch.nn as nn
l1_penalty = sum(p.abs().sum() for p in model.parameters())
l2_penalty = sum((p ** 2).sum() for p in model.parameters())
loss = criterion(output, target) + 1e-4 * l1_penalty + 1e-4 * l2_penalty</code></pre></p>

    <p><strong>Q4. Cross-validation থাকা সত্ত্বেও কেন overfitting হয়?</strong><br/>
    CV একটা single validation split-এ overfit হওয়া থেকে রক্ষা করে, কিন্তু একই CV score-এর বিরুদ্ধে শত শত hyperparameter trial চালালে search process-এর মধ্য দিয়েই information leak হয় — আপনি folds-এর প্রতিই overfit করেন।<br/>
    উদাহরণ: 5-fold CV AUC optimize করতে 500 Optuna trial চালিয়ে, সেই একই CV AUC-কে final performance হিসেবে report করা — held-out test set-এ score সাধারণত কম আসে।</p>

    <p><strong>Q5. এমন পরিস্থিতি explain করুন যেখানে accuracy misleading।</strong><br/>
    Imbalanced class-এ সবসময় majority class predict করেই accuracy trivially high হয়ে যায়।<br/>
    উদাহরণ: 0.5% fraud থাকা data-তে সবাইকে "not fraud" predict করে একটা fraud model 99.5% accuracy পায় — যে class matter করে তাতে 0% recall।</p>

    <p><strong>Q6. Pipeline-এ data leakage কীভাবে detect করবেন?</strong><br/>
    সন্দেহজনকভাবে বেশি validation score, target-এর সাথে প্রায় perfect correlation থাকা কোনো feature, অথবা prediction time-এ যেটা available থাকবে না এমন feature।<br/>
    উদাহরণ: churn model-এ "account_closed_date" feature হিসেবে ব্যবহার করা — এই column শুধু churn event-এর পরেই populate হয়, তাই এটা label leak করছে।</p>

    <p><strong>Q7. High AUC থাকা সত্ত্বেও কেন prediction খারাপ হতে পারে?</strong><br/>
    AUC সব threshold জুড়ে ranking quality measure করে, আপনি আসলে deploy করা নির্দিষ্ট threshold-এর calibration নয়।<br/>
    উদাহরণ: AUC=0.95 কিন্তু poorly calibrated probability (সব 0.4–0.6-এর মধ্যে squeeze করা) থাকা model fixed 0.5 cutoff-এ ঠিকমতো rank করলেও খারাপ decision নেয়।</p>

    <p><strong>Q8. Feature গুলো highly correlated হলে কী হয়?</strong><br/>
    Multicollinearity linear coefficient-কে unstable ও interpret করা কঠিন করে তোলে (ছোট data change-এ sign/magnitude flip করে), যদিও এটা সাধারণত predictive accuracy-তে খুব একটা প্রভাব ফেলে না।<br/>
    উদাহরণ: "square_feet" এবং "square_meters" (perfectly correlated) দুটোই থাকা regression-এ run-এর মধ্যে coefficient estimate wildly fluctuate করে।</p>

    <p><strong>Q9. Tree model-এ কেন feature scaling দরকার হয় না?</strong><br/>
    Tree threshold comparison (feature &gt; x) দিয়ে split করে, যা scaling-এর মতো monotonic transform-এ invariant — split point শুধু shift করে।<br/>
    উদাহরণ: "income &gt; 50000" বনাম "income_scaled &gt; 0.3"-এ split করলে data-র identical partition হয়।</p>

    <p><strong>Q10. Complex model-এর বদলে কখন simple model prefer করবেন?</strong><br/>
    ছোট/noisy data, interpretability দরকার (regulatory, medical), tight latency budget, অথবা complex model-এর accuracy gain marginal হলে।<br/>
    উদাহরণ: loan-approval model regulator-দের কাছে explainable হতে হবে — clear coefficient-সহ logistic regression 2% বেশি AUC থাকা black-box gradient-boosted ensemble-কেও হার মানায়।</p>

    <p><strong>Q11. Training-এ ভালো কিন্তু production-এ খারাপ performance করা model কীভাবে debug করবেন?</strong><br/>
    Train/production feature distribution mismatch check করুন, preprocessing consistency verify করুন, train-only leakage check করুন, production data quality নিশ্চিত করুন — এবং PyTorch-এ specifically, model আসলে eval mode-এ আছে কিনা confirm করুন।<br/>
    উদাহরণ: খুবই common real bug: inference-এর আগে model.eval() call করতে ভুলে গেলে Dropout ও BatchNorm training mode-এ চলতে থাকে, ফলে production-এ prediction noisy ও unstable হয়ে যায়।
    <pre><code>model.eval()          # dropout বন্ধ করে, BatchNorm running stats freeze করে
with torch.no_grad():
    preds = model(x_batch)</code></pre></p>

    <p><strong>Q12. Curse of dimensionality intuition দিয়ে explain করুন।</strong><br/>
    Dimension বাড়ার সাথে সাথে data point গুলো প্রায় equidistant হয়ে যায় এবং space overwhelmingly sparse হয়ে পড়ে — "nearest neighbor" আর meaningfully near থাকে না।<br/>
    উদাহরণ: 2D-তে একটা unit square-এর বেশিরভাগই interior; 100D-তে hypercube-এর প্রায় সব volume corner-এর কাছে থাকে — random point-গুলো প্রায় equidistant হয়ে যায়।</p>

    <p><strong>Q13. Data split করার আগে PCA ব্যবহার করার risk কী?</strong><br/>
    পুরো dataset-এ train/test split-এর আগে PCA fit করলে test-set-এর variance/structure training-time transformation-এ leak হয়ে যায় — এক ধরনের data leakage।<br/>
    উদাহরণ: 10,000 rows-এ PCA fit করে তারপর 80/20 split করলে — components ইতিমধ্যেই test rows-এর distribution "দেখে ফেলেছে", validation score inflate করে।</p>

    <p><strong>Q14. Imbalanced dataset-এর জন্য evaluation metric কীভাবে বেছে নেবেন?</strong><br/>
    Accuracy বা plain ROC-AUC-এর বদলে PR-AUC, F1, precision/recall prefer করুন; false positive বনাম false negative-এর relative cost অনুযায়ী বেছে নিন।<br/>
    উদাহরণ: Cancer screening-এ (FN costly) → precision-এর কিছুটা খরচ করে হলেও recall/sensitivity optimize করুন।</p>

    <p><strong>Q15. Model complexity বাড়ালে কেন performance কমতে পারে?</strong><br/>
    একটা point-এর পরে, বাড়তি complexity signal নয়, noise fit করতে শুরু করে — bias কমার চেয়ে variance দ্রুত বাড়ে।<br/>
    উদাহরণ: 30টা data point-এ degree-15 polynomial fit করলে noise মুখস্থ হয়ে যায়, training point-গুলোর মাঝে wild oscillation তৈরি হয়।</p>

    <p><strong>Q16. Training ও test distribution আলাদা হলে কী হয়?</strong><br/>
    Covariate shift — learned decision boundary আর test data যেখানে আসলে থাকে তার সাথে মেলে না, তাই train metric ভালো দেখালেও performance কমে যায়।<br/>
    উদাহরণ: দিনের বেলার traffic-camera image-এ train করা model রাতের image-এ fail করে কারণ pixel-intensity distribution shift করেছে।</p>

    <p><strong>Q17. Stratified sampling explain করুন এবং কখন এটা fail করে।</strong><br/>
    Stratified sampling প্রতিটা split-এ class ratio বজায় রাখে। এটা fail করে যখন strata খুব fine-grained হয় (প্রতিটাতে খুব কম sample) অথবা যখন বেছে নেওয়া stratification variable-ই আসল গুরুত্বপূর্ণ variable নয়।<br/>
    উদাহরণ: শুধু primary label দিয়ে multi-label dataset stratify করলে validation fold-এ rare label combination সম্পূর্ণ absent থেকে যেতে পারে।</p>

    <p><strong>Q18. Concept drift কী এবং কীভাবে detect করবেন?</strong><br/>
    Feature ও target-এর মধ্যে সম্পর্ক সময়ের সাথে বদলায় (P(y|x) বদলায়) যদিও P(x) একই থাকে। Live performance monitoring, prediction/error distribution-এ statistical test (KS test), অথবা drift algorithm (ADWIN, DDM) দিয়ে detect করুন।<br/>
    উদাহরণ: 2023-এ train করা spam classifier 2026-এ degrade করে কারণ spammer-রা tactic বদলেছে — একই email feature এখন ভিন্ন label-এ map করে।</p>

    <p><strong>Q19. Ensemble model কেন প্রায়ই single model-এর চেয়ে ভালো করে?</strong><br/>
    Diverse, decorrelated model combine করলে variance কমে (bagging) এবং bias-ও কমতে পারে (boosting) — model-গুলোর মধ্যে uncorrelated error একে অপরকে cancel করে।<br/>
    উদাহরণ: Netflix Prize জেতা solution 100+ model blend করেছিল; একটা single model সেই blend-এর accuracy-র ধারেকাছেও যেতে পারেনি।</p>

    <p><strong>Q20. Cross-validation কখন reliable estimate দিতে ব্যর্থ হয়?</strong><br/>
    Time series-এ (random fold future info training-এ leak করে), খুব ছোট dataset-এ (fold জুড়ে high variance), অথবা grouped data-তে (এক patient-এর multiple row) যেখানে split সেই grouping ignore করে।<br/>
    উদাহরণ: patient-level medical record-এ K-fold CV যেখানে একই patient-এর row train ও test দুই fold-এই আছে — patient-specific signal leak করে score inflate করে; এখানে GroupKFold দরকার।</p>

    <h3>Supervised Learning Deep Dive (Q21–Q40)</h3>

    <p><strong>Q21. Logistic regression-এর cost function derive করুন।</strong><br/>
    P(y|x)=σ(wx+b)-কে Bernoulli likelihood হিসেবে model করে negative log-likelihood নিলে binary cross-entropy পাওয়া যায়: J = -1/n Σ[yᵢlog(ŷᵢ) + (1-yᵢ)log(1-ŷᵢ)]। Sigmoid-এর উপর squared error-এর বিপরীতে, এই loss w-তে convex।<br/>
    উদাহরণ: true y=1 এবং predicted ŷ=0.01 হলে loss -log(0.01)≈4.6 অনেক বড় — confidently ভুল prediction-কে sharply penalize করা হয়।
    <pre><code>import torch.nn as nn
criterion = nn.BCEWithLogitsLoss()   # sigmoid + BCE একসাথে, numerically stable
loss = criterion(logits, targets.float())</code></pre></p>

    <p><strong>Q22. Gradient descent কখনো কখনো কেন converge করে না?</strong><br/>
    Learning rate খুব বেশি হলে → oscillation/divergence। খুব কম হলে → খুবই ধীর progress। Non-convex surface-এ saddle point/local minima থাকে যা progress আটকে দেয়। খারাপ feature scaling loss landscape-কে narrow ravine বানিয়ে দেয়।<br/>
    উদাহরণ: lr=10 দিয়ে train করলে কয়েক step-এই loss explode করে NaN হয়ে যায় — learning rate অনেক বেশি হওয়ার classic sign।</p>

    <p><strong>Q23. Simple neural network-এ vanishing gradient কেন হয়?</strong><br/>
    Sigmoid/tanh-এর derivative ≤0.25, আর chain rule layer-এর পর layer এই ছোট সংখ্যাগুলো multiply করতে থাকে, ফলে input layer-এর দিকে gradient exponentially ছোট হয়ে যায়।<br/>
    উদাহরণ: 10-layer sigmoid network-এ প্রতি layer-এ 0.2 factor হলে layer 1-এ তা 0.2¹⁰ ≈ 1×10⁻⁷ হয়ে যায় — কার্যত zero, তাই প্রথম দিকের layer শেখা বন্ধ করে দেয়।</p>

    <p><strong>Q24. Edge case-এ decision tree বনাম random forest compare করুন।</strong><br/>
    একটা single tree unstable — ছোট data change খুব ভিন্ন tree তৈরি করতে পারে (high variance) — কিন্তু পুরোপুরি interpretable। Random forest bootstrap sample ও random feature subset-এর উপর অনেকগুলো tree-র average নেয়, variance কমায়, কিন্তু single clean decision path হারিয়ে যায়।<br/>
    উদাহরণ: 5টা row সরালে একটা single tree-র root split সম্পূর্ণ বদলে যেতে পারে, কিন্তু 500-tree forest-এর prediction প্রায় একই থাকে।</p>

    <p><strong>Q25. Boosting কেন bias কমায় কিন্তু overfitting-এর risk রাখে?</strong><br/>
    প্রতিটা নতুন weak learner ensemble-এর এখন পর্যন্ত residual error-এর উপর fit হয়, ধীরে ধীরে systematic bias সংশোধন করে — কিন্তু যথেষ্ট round-এর পর সেই residual-এর noise-ও fit করা শুরু করে।<br/>
    উদাহরণ: noisy data-তে early stopping ছাড়া 5000 round-এর XGBoost training error প্রায় zero করে ফেলে, কিন্তু validation error round ~300-এর পর আবার বাড়তে শুরু করে — classic overfitting curve।</p>

    <p><strong>Q26. XGBoost missing value কীভাবে handle করে explain করুন।</strong><br/>
    প্রতিটা split-এ XGBoost missing value-র জন্য একটা "default direction" শেখে — দুই দিকেই try করে যেটা loss কমায় সেটা বেছে নেয় — কোনো imputation দরকার হয় না।<br/>
    উদাহরণ: 15% missing value থাকা "income" feature training-এর সময় যেদিকে split gain ভালো হয় সেই দিকে automatically route হয়।</p>

    <p><strong>Q27. Kernel trick থাকা সত্ত্বেও SVM কখন fail করে?</strong><br/>
    খুব বড় dataset-এ (kernel SVM O(n²)-O(n³)), heavily overlapping class-এ যেখানে clear margin নেই, অথবা high-dimensional sparse data-তে যেখানে kernel সঠিক structure capture করে না।<br/>
    উদাহরণ: 1M rows-এ RBF-kernel SVM train করতে ঘণ্টার পর ঘণ্টা লাগে বা memory শেষ হয়ে যায় — gradient boosting অনেক ভালো scale করে।</p>

    <p><strong>Q28. কিছু ক্ষেত্রে log loss-এর বদলে কেন hinge loss ব্যবহার করা হয়?</strong><br/>
    Hinge loss (max(0, 1-y·f(x))) শুধু misclassified বা margin-violating point-কে penalize করে, confidently correct point-এ zero gradient দেয় — sparse "support vector" solution তৈরি করে এবং calibrated probability নয়, সরাসরি margin optimize করে।<br/>
    উদাহরণ: decision margin-এর অনেক পেরিয়ে থাকা একটা point hinge loss-এ 0 contribute করে, কিন্তু log loss সেটাকে আরও ঠেলে দিত, অপ্রয়োজনীয়ভাবে।</p>

    <p><strong>Q29. Margin maximization intuitively explain করুন।</strong><br/>
    SVM যেকোনো separating boundary খোঁজে না — এটা class-গুলোর মাঝে সবচেয়ে বড় buffer zone থাকা boundary খোঁজে, যা boundary-র কাছের unseen point-এ ভালো generalize করে।<br/>
    উদাহরণ: দুটো cluster-এর মাঝে সম্ভাব্য দুটো separating line-এর মধ্যে, exactly মাঝখানে থাকা (max margin) line boundary-র কাছের নতুন point-এ কম misclassify করে।</p>

    <p><strong>Q30. High dimension-এ k-NN model কেন struggle করে?</strong><br/>
    একই curse-of-dimensionality সমস্যা — high-D space-এ সব point-এর মধ্যে distance converge করে, তাই "nearest" neighbor "দূরের" neighbor-এর চেয়ে meaningfully কাছে থাকে না।<br/>
    উদাহরণ: 500-dim word-embedding space-এ সত্যিকারের similar আর random dissimilar document-এর cosine distance মাত্র কয়েক percent আলাদা হতে পারে।</p>

    <p><strong>Q31. k-NN-এ k=1 বনাম খুব বড় k হলে কী হয়?</strong><br/>
    k=1 → training error zero কিন্তু খুব high variance (প্রতিটা prediction একটা মাত্র, সম্ভবত noisy, neighbor দ্বারা নির্ধারিত)। খুব বড় k (k=n) → high bias, prediction local structure ignore করে global majority class-এর দিকে চলে যায়।<br/>
    উদাহরণ: noisy labeled data-তে k=1 একটা jagged, overfit decision boundary তৈরি করে; k=n সব জায়গায় একই majority class predict করে।</p>

    <p><strong>Q32. Assumption violate হলেও Naive Bayes কেন কাজ করে explain করুন।</strong><br/>
    NB-কে শুধু সঠিক class-কে সবচেয়ে উপরে rank করতে হয়, accurate probability দিতে হয় না — independence assumption-এর error প্রায়ই সব class-কে একইভাবে bias করে, ফলে সঠিক argmax ranking টিকে থাকে।<br/>
    উদাহরণ: "New" এবং "York" স্পষ্টতই dependent, independent নয়, তবুও Naive Bayes "New York Times" article ঠিকমতোই classify করে কারণ mis-estimation সব class জুড়ে systematic।</p>

    <p><strong>Q33. Linear-looking data হলেও কখন linear regression অনুপযুক্ত?</strong><br/>
    যখন error heteroscedastic, non-normal, যখন severe autocorrelation থাকে (time series), অথবা outlier dominate করে — scatter plot linear দেখালেও OLS assumption violate হয়।<br/>
    উদাহরণ: linear predictor-এর বিপরীতে stock return scatter plot-এ linear দেখাতে পারে কিন্তু fat-tailed, autocorrelated residual থাকতে পারে, যা OLS confidence interval invalid করে।</p>

    <p><strong>Q34. Polynomial feature কখনো কখনো কেন performance খারাপ করে?</strong><br/>
    High-degree polynomial term দ্রুত overfit করে (Runge's phenomenon) এবং term-গুলোর মধ্যে (x, x², x³ highly correlated) multicollinearity তৈরি করে, coefficient estimate-কে unstable করে।<br/>
    উদাহরণ: 10 point-এর মধ্য দিয়ে degree-9 polynomial প্রতিটা point দিয়ে exactly যায় কিন্তু মাঝখানে wildly oscillate করে — খারাপ interpolation।</p>

    <p><strong>Q35. Regularized regression-এ coefficient কীভাবে interpret করবেন?</strong><br/>
    Regularized coefficient fit-বনাম-penalty tradeoff অনুযায়ী zero-এর দিকে shrink হয় — এর magnitude unregularized OLS coefficient-এর সাথে সরাসরি comparable নয় এবং unbiased "effect size" হিসেবে পড়া উচিত নয়।<br/>
    উদাহরণ: একটা feature-এর Ridge coefficient 0.3 হতে পারে যেখানে একই feature-এর OLS coefficient 0.8 — shrinkage আসল effect নয়, λ-এর উপর নির্ভর করে।</p>

    <p><strong>Q36. Learning rate খুব ছোট হলে কী হয়?</strong><br/>
    Training অত্যন্ত ধীরে converge করে, এত বেশি step ধরে plateau-তে আটকে থাকে যে মনে হয় শেখা বন্ধ হয়ে গেছে, এবং compute budget নষ্ট হয়।
    <pre><code>import torch.optim as optim
optimizer = optim.SGD(model.parameters(), lr=1e-8)  # অনেক ছোট
# হাজার হাজার step-এর পরেও loss প্রায় নড়ে না</code></pre></p>

    <p><strong>Q37. Early stopping কেন regularization হিসেবে কাজ করে?</strong><br/>
    Training loss সম্পূর্ণ converge হওয়ার আগেই থামালে model noise মুখস্থ করার point-এ পৌঁছায় না — এটা implicit-ভাবে effective model capacity সীমিত করে, weight shrink করার মতোই।
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
            break   # model noise মুখস্থ করা শুরু করার আগেই থামুন</code></pre></p>

    <p><strong>Q38. Bagging বনাম boosting উদাহরণসহ compare করুন।</strong><br/>
    Bagging bootstrap sample-এ parallel-এ model train করে এবং variance কমাতে average নেয় (Random Forest)। Boosting sequentially model train করে, প্রতিটা আগেরটার error সংশোধন করে, bias কমায় (AdaBoost, XGBoost)।<br/>
    উদাহরণ: noisy data-তে Random Forest outlier-এর ক্ষেত্রে robust কারণ averaging noise cancel করে; একই data-তে XGBoost outlier-এ overfit করতে পারে কারণ পরের round-গুলো তাদের বড় residual-এর পেছনে ছোটে।</p>

    <p><strong>Q39. Random forest কেন variance কমায় কিন্তু bias কমায় না?</strong><br/>
    অনেকগুলো high-variance, low-bias tree-র average নিলে তাদের independent noise cancel হয়, কিন্তু প্রতিটা tree algorithm থেকে একই structural limitation ভাগ করে নেয় — averaging systematic underfitting ঠিক করতে পারে না।<br/>
    উদাহরণ: প্রতিটা tree depth=2-তে capped থাকলে (systematically একটা complex boundary-র জন্য খুব shallow), 500টা average নিলেও সেই boundary represent করা যায় না — bias high-ই থাকে।</p>

    <p><strong>Q40. Probability calibration explain করুন।</strong><br/>
    একটা calibrated model-এর predicted probability সত্যিকারের empirical frequency-র সাথে মেলে — 0.7 predict করা সব case-এর মধ্যে প্রায় 70%-ই আসলে positive হওয়া উচিত। Poorly calibrated model (raw SVM score, overconfident deep net)-এর জন্য Platt scaling বা isotonic regression দরকার।<br/>
    উদাহরণ: "70% বৃষ্টির সম্ভাবনা" predict করা একটা weather model calibrated যদি এমন দিনগুলোর ~70%-এই সত্যিই বৃষ্টি হয় — 95% বা 40% নয়।</p>

    <h3>Optimization &amp; Training (Q41–Q55)</h3>

    <p><strong>Q41. SGD, Momentum, RMSProp, Adam compare করুন।</strong><br/>
    SGD প্রতি step-এ raw gradient দিয়ে update করে (simple, noisy)। Momentum আগের gradient-গুলোর moving average জমিয়ে update smooth করে এবং ravine-এর মধ্য দিয়ে accelerate করে। RMSProp squared gradient-এর moving average দিয়ে per-parameter learning rate adapt করে। Adam momentum + RMSProp-এর adaptive scaling একসাথে combine করে।
    <pre><code>import torch.optim as optim
sgd      = optim.SGD(model.parameters(), lr=0.01)
momentum = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
rmsprop  = optim.RMSprop(model.parameters(), lr=0.001)
adam     = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))</code></pre>
    Narrow ravine-এ plain SGD দেয়াল বরাবর oscillate করে; momentum সেই oscillation কমিয়ে ravine-এর floor বরাবর accelerate করে।</p>

    <p><strong>Q42. Adam কখনো কখনো SGD-এর চেয়ে কেন খারাপ generalize করে?</strong><br/>
    Adam-এর adaptive per-parameter learning rate sharp minima-তে converge করতে পারে যা training data precisely fit করে কিন্তু খারাপ generalize করে, যেখানে SGD-এর noisier update flatter minima খোঁজে।<br/>
    উদাহরণ: অনেক vision model (ResNet) এখনও Adam-এর বদলে SGD+momentum দিয়ে train করা হয় specifically কারণ final test accuracy measurably ভালো, যদিও early training-এ Adam দ্রুত converge করে।</p>

    <p><strong>Q43. Optimization-এ saddle point problem কী?</strong><br/>
    High-dimensional non-convex surface-এ saddle point (gradient zero, কিছু direction-এ minimum, কিছুতে maximum) local minima-র চেয়ে অনেক বেশি — gradient descent তার কাছে গিয়ে অনেক ধীর হয়ে যায় কারণ gradient magnitude ছোট হয়ে আসে।<br/>
    উদাহরণ: saddle point-এর কাছে loss curve এক axis বরাবর flat দেখায় (convergence মনে হয়) কিন্তু আরেক axis বরাবর আরও গভীরে গেলে এখনও descend করছে।</p>

    <p><strong>Q44. Deep network-এ local minima কীভাবে escape করবেন?</strong><br/>
    Momentum/Adam ব্যবহার করুন (flat region দিয়ে velocity বহন করে), mini-batch SGD-এর inherent noise-এর উপর নির্ভর করুন, learning-rate warm restart ব্যবহার করুন, অথবা model width বাড়ান (over-parameterization empirically খারাপ local minima এড়ায়)।
    <pre><code>scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2)   # LR-কে periodically আবার উঁচুতে নিয়ে যায়
for epoch in range(epochs):
    train_one_epoch(...)
    scheduler.step()</code></pre></p>

    <p><strong>Q45. Weight initialization কেন গুরুত্বপূর্ণ?</strong><br/>
    খারাপ initialization প্রথম forward/backward pass থেকেই activation/gradient vanish বা explode করাতে পারে; ভালো initialization (Xavier/He) layer জুড়ে activation variance প্রায় constant রাখে।
    <pre><code>import torch.nn as nn
linear = nn.Linear(256, 256)
nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')  # ReLU network-এর জন্য He init
nn.init.xavier_normal_(linear.weight)                        # tanh/sigmoid network-এর জন্য</code></pre></p>

    <p><strong>Q46. সব weight zero দিয়ে initialize করলে কী হয়?</strong><br/>
    একটা layer-এর প্রতিটা neuron identical output produce করে এবং identical gradient পায় — symmetry কখনো break হয় না, তাই layer-এর সব neuron চিরকাল identical থেকে যায়, layer-টা এক neuron-এর capacity-তে collapse হয়ে যায়।
    <pre><code>for p in model.parameters():
    nn.init.zeros_(p)
# layer-এর প্রতিটা neuron এখন identically update হয় — 512টা neuron 1টার মতো আচরণ করে</code></pre></p>

    <p><strong>Q47. Exploding gradient এবং mitigation strategy explain করুন।</strong><br/>
    Layer derivative/weight 1-এর বেশি হলে backprop-এর মধ্য দিয়ে gradient exponentially বাড়ে, huge unstable update (বা NaN) তৈরি করে। Gradient clipping, ভালো initialization, BatchNorm, বা ছোট learning rate দিয়ে mitigate করুন।
    <pre><code>loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
optimizer.step()</code></pre>
    100 timestep জুড়ে unroll করা একটা RNN-এ weight-matrix-এর spectral norm 1.5 হলে gradient 1.5¹⁰⁰ গুণ amplify হয় — norm 5-তে clip করলে training stable থাকে।</p>

    <p><strong>Q48. Batch normalization কেন training দ্রুত করে?</strong><br/>
    এটা প্রতিটা mini-batch-এ layer-এর activation-কে zero mean/unit variance-এ normalize করে, loss landscape smooth করে এবং higher learning rate সম্ভব করে।
    <pre><code>import torch.nn as nn
layer = nn.Sequential(
    nn.Linear(256, 256),
    nn.BatchNorm1d(256),
    nn.ReLU()
)</code></pre>
    BatchNorm ছাড়া lr=0.001 দরকার হওয়া একটা CNN BatchNorm-সহ প্রায়ই lr=0.01+ এ stable train করতে পারে, training time যথেষ্ট কমিয়ে দেয়।</p>

    <p><strong>Q49. Batch normalization কখন performance খারাপ করতে পারে?</strong><br/>
    খুব ছোট batch size-এ (batch statistics noisy/unrepresentative হয়ে যায়), RNN timestep জুড়ে (recurrent computation ভেঙে দেয়), অথবা train/inference batch statistics যথেষ্ট আলাদা হলে।<br/>
    উদাহরণ: GPU memory limit-এর কারণে batch size 2 দিয়ে train করা object detection model-এ BatchNorm statistics unstable দেখায় — এর বদলে প্রায়ই nn.GroupNorm ব্যবহার করা হয়।</p>

    <p><strong>Q50. Gradient clipping এবং এর use case explain করুন।</strong><br/>
    Weight update-এর আগে gradient-এর norm (বা individual value) cap করে, একটা বড় gradient spike training অস্থির করে দেওয়া থেকে আটকায় — RNN/LSTM এবং Transformer training-এ essential।
    <pre><code># norm দিয়ে clip (পুরো gradient vector rescale হয়)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# value দিয়ে clip (প্রতিটা gradient element আলাদাভাবে clamp হয়)
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)</code></pre></p>

    <p><strong>Q51. Mini-batch training কেন full-batch-এর চেয়ে ভালো performance দেয়?</strong><br/>
    Mini-batch noisy-কিন্তু-frequent gradient update দেয় (এক epoch-এ অনেকগুলো update, একটার বদলে), সেই noise sharp minima/saddle point এড়াতে সাহায্য করে, এবং বড় dataset-এর জন্য GPU memory-তে fit করে।
    <pre><code>from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=256, shuffle=True)  # batch_size=len(dataset) নয়</code></pre></p>

    <p><strong>Q52. Batch size generalization-এ কী effect ফেলে?</strong><br/>
    ছোট batch → noisy gradient → flatter minima → ভালো generalization। বড় batch → smooth gradient → sharper minima → কখনো কখনো খারাপ test performance (the "generalization gap")।<br/>
    উদাহরণ: একই ResNet-কে batch size 32 বনাম 8192 দিয়ে (একই epoch-এ) train করলে — learning rate linearly scale ও warmup না করলে batch=8192 প্রায়ই লক্ষণীয় test-accuracy drop দেখায়।</p>

    <p><strong>Q53. Learning rate scheduling কীভাবে সাহায্য করে?</strong><br/>
    শুরুতে বড় LR progress দ্রুত করে; পরে decay করলে overshoot ছাড়াই minimum-এ precise convergence সম্ভব হয়।
    <pre><code>scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# lr = 0.1 epoch 1-30-এ, 0.01 31-60-এ, 0.001 61-90-এ</code></pre></p>

    <p><strong>Q54. Training-এ warm restart explain করুন।</strong><br/>
    Training-এর মাঝপথে periodically LR-কে আবার উঁচু value-তে reset করা (যেমন cosine annealing schedule) — optimizer-এর exploration "restart" করে — sharp minima escape করতে পারে এবং implicit ensemble-এর মতো কাজ করে (snapshot ensembling)।
    <pre><code>scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
# প্রতি 10 epoch-এ LR max-এ reset হয়; প্রতিটা restart-এর কাছের snapshot ensemble করা যায়</code></pre></p>

    <p><strong>Q55. Dropout কেন কাজ করে?</strong><br/>
    প্রতিটা training step-এ কিছু neuron randomly zero করলে co-adaptation আটকায় এবং exponentially অনেকগুলো thinned sub-network-এর একটা ensemble train করার মতো approximate করে, test time-এ যেগুলো average করা হয়।
    <pre><code>import torch.nn as nn
layer = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Dropout(p=0.5))
model.train()   # dropout active — প্রতি step-এ random ~50% neuron zero হয়
model.eval()    # dropout বন্ধ — সব neuron active, output auto-scaled</code></pre></p>

    <h3>Neural Networks &amp; Deep Learning (Q56–Q75)</h3>

    <p><strong>Q56. Deep network কেন shallow network-এর চেয়ে বেশি powerful?</strong><br/>
    Depth feature-এর hierarchical composition সম্ভব করে (edge → texture → part → object), সমানভাবে expressive shallow (wide) network-এর চেয়ে exponentially কম neuron দিয়ে complex function represent করতে পারে।<br/>
    উদাহরণ: একটা CNN-এর প্রথম layer edge detect করে, মাঝের layer texture/shape detect করে, শেষের layer পুরো object detect করে — প্রতিটা layer আগের layer-এর feature reuse করে।</p>

    <p><strong>Q57. Universal approximation theorem-এর limitation explain করুন।</strong><br/>
    UAT guarantee দেয় যে যথেষ্ট wide একটা single hidden layer যেকোনো continuous function approximate করতে পারে, কিন্তু কতগুলো neuron দরকার (exponential হতে পারে) বা random init থেকে gradient descent দিয়ে সেই network আদৌ learnable কিনা সে সম্পর্কে কিছু বলে না।<br/>
    উদাহরণ: একটা মোটামুটি complex function এক hidden layer দিয়ে approximate করতে theoretically millions of neuron লাগতে পারে, যেখানে অনেক কম total parameter-এর একটা deep network practice-এ একই accuracy পায়।</p>

    <p><strong>Q58. ReLU কেন sigmoid/tanh-এর চেয়ে ভালো?</strong><br/>
    Positive input-এ ReLU-এর gradient exactly 1 (active region-এ vanishing gradient নেই), এটা computationally সস্তা, এবং empirically sparse activation তৈরি করে।
    <pre><code>import torch
x = torch.linspace(-5, 5, 5)
print(torch.sigmoid(x).detach())      # tail-এর দিকে gradient 0-এর কাছে চলে যায়
print(torch.relu(x).detach())          # x > 0-এ gradient exactly 1, বাকি জায়গায় flat 0</code></pre></p>

    <p><strong>Q59. ReLU কখন fail করে?</strong><br/>
    "Dying ReLU" — কোনো neuron-এর weight এমনভাবে update হয় যে তার input সবসময় negative হয়ে যায়, তার output ও gradient permanently zero হয়ে যায়, আর কখনো recover করে না (negative input-এ gradient 0)।<br/>
    উদাহরণ: high learning rate training-এর শুরুতে একটা neuron-এর bias খুব negative করে দেয়; সেই neuron তারপর থেকে চিরকাল 0 output দেয়, model capacity নষ্ট হয়।</p>

    <p><strong>Q60. Leaky ReLU, ELU, GELU compare করুন।</strong><br/>
    Leaky ReLU x&lt;0-এ ছোট negative slope (যেমন 0.01x) রাখে dying neuron আটকাতে। ELU x&lt;0-এ smoothly একটা negative constant-এ saturate হয়, activation-কে zero mean-এর কাছাকাছি রাখে। GELU input-কে Gaussian-এর percentile দিয়ে weight করে, smooth probabilistic gating দেয় — BERT/GPT-তে ব্যবহৃত।
    <pre><code>import torch.nn as nn
leaky = nn.LeakyReLU(negative_slope=0.01)
elu   = nn.ELU(alpha=1.0)
gelu  = nn.GELU()   # modern Transformer block-এ প্রায় universal choice</code></pre></p>

    <p><strong>Q61. Deeper network কেন train করা কঠিন হয়ে যায়?</strong><br/>
    Depth-এর সাথে vanishing/exploding gradient জমে, এবং skip connection ছাড়া network-কে useful information pass করাতে ক্রমশ জটিল identity-এর মতো mapping শিখতে হয় — gradient descent সরাসরি এটা খুঁজে পেতে struggle করে।<br/>
    উদাহরণ: original ResNet paper-এ একটা plain (non-residual) 56-layer CNN-এর training error একটা plain 20-layer CNN-এর চেয়ে বেশি ছিল — shortcut ছাড়া শুধু depth optimization-কেই সরাসরি ক্ষতি করেছিল, শুধু generalization নয়।</p>

    <p><strong>Q62. Residual connection explain করুন এবং এটা কেন সাহায্য করে।</strong><br/>
    y = F(x) + x network-কে পুরো mapping-এর বদলে residual F(x) ≈ y − x (প্রায়ই zero-এর কাছাকাছি) শেখার সুযোগ দেয়, এবং আগের layer-গুলোতে সরাসরি identity gradient path দেয়, vanishing gradient bypass করে।
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

    <p><strong>Q63. RNN-এ vanishing gradient কী?</strong><br/>
    Backpropagation through time বারবার একই recurrent weight matrix multiply করে timestep জুড়ে; এর eigenvalue 1-এর কম হলে, sequence length-এর সাথে gradient exponentially ছোট হয়ে যায়, তাই network long-range dependency শিখতে পারে না।<br/>
    উদাহরণ: timestep 1-এর subject-কে timestep 50-এর verb-এর সাথে connect করা একটা RNN 50 step backprop-এর পর সেই dependency-র জন্য কার্যত zero gradient signal পায়।</p>

    <p><strong>Q64. LSTM কেন long-term dependency সমস্যা সমাধান করে?</strong><br/>
    Cell state-এর একটা প্রায়-linear, additive update path আছে (forget gate দ্বারা নিয়ন্ত্রিত), repeated matrix multiplication-এর বদলে, তাই forget gate 1-এর কাছাকাছি থাকলে অনেক timestep জুড়ে gradient প্রায় অপরিবর্তিত থেকে flow করতে পারে।
    <pre><code>import torch.nn as nn
lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True)
output, (h_n, c_n) = lstm(x)   # c_n forget gate-এর মাধ্যমে long-range signal বহন করে</code></pre></p>

    <p><strong>Q65. বাস্তব ক্ষেত্রে LSTM বনাম GRU compare করুন।</strong><br/>
    LSTM-এ 3টা gate (input/forget/output) এবং আলাদা cell state আছে — বেশি expressive, বেশি parameter। GRU cell ও hidden state-কে merge করে 2টা gate দিয়ে (reset/update) — কম parameter, দ্রুত train হয়, ছোট dataset-এ প্রায়ই LSTM-এর সমান।
    <pre><code>import torch.nn as nn
lstm = nn.LSTM(128, 256, batch_first=True)
gru  = nn.GRU(128, 256, batch_first=True)
print(sum(p.numel() for p in lstm.parameters()))  # প্রায় 4x hidden_size*(input+hidden)
print(sum(p.numel() for p in gru.parameters()))   # প্রায় 3x — লক্ষণীয়ভাবে কম</code></pre></p>

    <p><strong>Q66. CNN কেন image-এর জন্য ভালো কাজ করে?</strong><br/>
    Convolution image-specific দুটো prior ব্যবহার করে: local spatial correlation এবং translation invariance, local receptive field ও weight sharing-এর মাধ্যমে achieve করা।<br/>
    উদাহরণ: একটা single 3×3 edge-detecting filter image-এর প্রতিটা position-এ reuse হয়, প্রতিটা pixel location-এর জন্য আলাদা filter শেখার বদলে — parameter millions থেকে কয়েক ডজনে নেমে আসে।</p>

    <p><strong>Q67. Convolution operation intuitively explain করুন।</strong><br/>
    একটা ছোট filter (kernel) image-এর উপর দিয়ে slide করে, প্রতিটা position-এ একটা weighted sum calculate করে — এমন একটা feature map তৈরি করে যা highlight করে filter যে pattern represent করে (যেমন vertical edge) সেটা কোথায় আছে।
    <pre><code>import torch.nn as nn
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
feature_map = conv(image_batch)   # 16টা learned filter image-এর উপর দিয়ে slide করে</code></pre>
    [[1,0,-1],[1,0,-1],[1,0,-1]] এই 3×3 vertical-edge kernel যেখানেই বাম থেকে ডানে intensity উজ্জ্বল থেকে অন্ধকারে jump করে সেখানে strongly respond করে।</p>

    <p><strong>Q68. Pooling layer-এর ভূমিকা কী?</strong><br/>
    Feature map downsample করে (max-pooling প্রতিটা window-এ max নেয়), spatial resolution/computation কমায় এবং translation invariance যোগ করে।
    <pre><code>import torch.nn as nn
pool = nn.MaxPool2d(kernel_size=2, stride=2)
out = pool(feature_map)   # 224x224 → 112x112, "এই feature এখানে ছিল কিনা" রাখে</code></pre></p>

    <p><strong>Q69. Pooling কখন এড়ানো উচিত?</strong><br/>
    যখন precise spatial/location information matter করে, যেমন semantic segmentation বা pose estimation — pooling-এর information loss pixel-level output quality-কে সরাসরি ক্ষতি করে; এর বদলে strided বা dilated convolution ব্যবহার হয়।<br/>
    উদাহরণ: medical image segmentation-এর U-Net specifically skip connection ব্যবহার করে যাতে encoder-এর pooling যে fine spatial detail নষ্ট করেছে তা recover করা যায়।</p>

    <p><strong>Q70. Padding এবং stride-এর impact explain করুন।</strong><br/>
    Padding border pixel যোগ করে যাতে edge pixel সমানভাবে treat হয় এবং output size control করা যায় ("same" padding output size = input size রাখে)। Stride প্রতিটা step-এ filter কতদূর move করবে তা নিয়ন্ত্রণ করে — stride 2 output resolution প্রায় অর্ধেক করে দেয়।
    <pre><code>import torch.nn as nn
conv_same    = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1)  # 5x5 -> 5x5
conv_strided = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2)  # 5x5 -> 3x3</code></pre></p>

    <p><strong>Q71. অনেক task-এ transformer কেন RNN-কে replace করছে?</strong><br/>
    Transformer সব sequence position parallel-এ process করে (কোনো sequential dependency নেই), অনেক দ্রুত training সম্ভব করে, এবং self-attention distance যাই হোক না কেন যেকোনো দুটো position সরাসরি connect করে, RNN-এর vanishing-gradient-limited long-range dependency এড়িয়ে।<br/>
    উদাহরণ: length 512-এর একটা sequence-এ train করা — RNN-কে অবশ্যই 512টা sequential step process করতে হবে; Transformer সব attention score parallel matrix operation-এ calculate করে, GPU-তে wall-clock training time অনেক কমিয়ে দেয়।</p>

    <p><strong>Q72. Attention mechanism conceptually কী?</strong><br/>
    প্রতিটা query-এর জন্য, attention সব value vector-এর একটা weighted average calculate করে, যেখানে weight (softmax(QKᵀ/√d) থেকে) দেখায় প্রতিটা key সেই query-এর জন্য কতটা relevant — model-কে dynamically input-এর সবচেয়ে relevant অংশে focus করতে দেয়।
    <pre><code>import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / d_k ** 0.5
    weights = F.softmax(scores, dim=-1)
    return weights @ V

# অথবা সহজভাবে: nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)</code></pre></p>

    <p><strong>Q73. Self-attention কেন powerful?</strong><br/>
    এটা একটা single layer-এই (O(1) path length) sequence-এর প্রতিটা token pair-এর মধ্যে সরাসরি pairwise interaction model করে, sequence-এ distance যাই হোক না কেন, যেখানে RNN-এ দূরের token-কে interact করতে অনেক sequential step লাগে।<br/>
    উদাহরণ: 1000-word document-এ self-attention word 5 এবং word 995-কে এক layer-এই সরাসরি interact করতে দেয়, যেখানে RNN-কে 990টা intermediate hidden state দিয়ে propagate করতে হতো।</p>

    <p><strong>Q74. Positional encoding-এর প্রয়োজনীয়তা explain করুন।</strong><br/>
    Self-attention permutation-invariant — position information inject না করলে token order যাই হোক না কেন একই output দেয়, তাই embedding-এ positional encoding (sinusoidal বা learned) যোগ করা হয়।
    <pre><code>import torch

def sinusoidal_positional_encoding(seq_len, d_model):
    pos = torch.arange(seq_len).unsqueeze(1)
    i = torch.arange(d_model).unsqueeze(0)
    angle = pos / (10000 ** (2 * (i // 2) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(angle[:, 0::2])
    pe[:, 1::2] = torch.cos(angle[:, 1::2])
    return pe</code></pre>
    Positional encoding ছাড়া "dog bites man" আর "man bites dog" একই attention computation দিত কারণ attention শুধু token embedding-এর unordered set দেখে।</p>

    <p><strong>Q75. Regularization থাকা সত্ত্বেও deep network কখন overfit করে?</strong><br/>
    যখন model ছোট/noisy dataset-এর তুলনায় massively overparameterized, regularization strength যথেষ্ট না, অথবা regularization শুধু weight-কে target করে কিন্তু architecture-specific overfitting source-কে করে না।<br/>
    উদাহরণ: মাত্র 500টা labeled example-এ fine-tune করা একটা 100M-parameter transformer dropout=0.3 এবং weight decay থাকা সত্ত্বেও লক্ষণীয়ভাবে overfit করতে পারে, কারণ parameter count example সংখ্যাকে বহুগুণে ছাড়িয়ে যায়।</p>

    <h3>Advanced Topics (Q76–Q90)</h3>

    <p><strong>Q76. Transfer learning explain করুন এবং এটা কখন fail করে।</strong><br/>
    বড় source task-এ শেখা feature/weight reuse করা, সাধারণত শুধু শেষ layer fine-tune করা (বা ছোট LR দিয়ে সব layer)। এটা fail করে যখন source ও target domain খুব ভিন্ন (negative transfer) অথবা target task-এ এমন feature দরকার যা source domain-এ কখনো ছিল না।
    <pre><code>import torchvision.models as models
backbone = models.resnet50(weights='IMAGENET1K_V2')
for param in backbone.parameters():
    param.requires_grad = False        # pretrained backbone freeze করা
backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)  # শুধু নতুন head train হবে</code></pre></p>

    <p><strong>Q77. Fine-tuning কখনো কখনো কেন performance খারাপ করে?</strong><br/>
    Catastrophic forgetting (নতুন task-এর gradient pretrained feature overwrite করে), খুব high learning rate pretrained weight প্রথম দিকেই নষ্ট করে দেয়, অথবা target example খুব কম হলে fine-tuning-এ মারাত্মক overfitting হয়।<br/>
    উদাহরণ: lr=1e-3 (অনেক বেশি) দিয়ে মাত্র 2 epoch pretrained BERT fine-tune করলে প্রায়ই language understanding পুরোপুরি ভেঙে পড়ে — recommended fine-tuning LR প্রায় 2e-5।</p>

    <p><strong>Q78. Catastrophic forgetting কী?</strong><br/>
    Sequentially task B-তে train করা model task A ভুলে যায় কারণ B-এর জন্য gradient update A-এর জন্য গুরুত্বপূর্ণ weight overwrite করে দেয়, সেই আগের knowledge রক্ষা করার কোনো mechanism ছাড়াই।<br/>
    উদাহরণ: শুধু customer-support conversation-এ fine-tune করা একটা chatbot ধীরে ধীরে initial pretraining থেকে পাওয়া general conversational ability হারিয়ে ফেলে।</p>

    <p><strong>Q79. Domain adaptation-এর challenge explain করুন।</strong><br/>
    Source ও target domain-এর input/label distribution ভিন্ন — model-কে labeled target data ছাড়াই (unsupervised domain adaptation) বা খুব কম label দিয়ে adapt করতে হয়, usual train/test-same-distribution assumption violate করে।<br/>
    উদাহরণ: movie review-এ train করা sentiment classifier restaurant review-এ apply করলে — "cheesy" movie-র জন্য negative কিন্তু vocabulary/cue যথেষ্ট আলাদা যে adaptation দরকার হয়।</p>

    <p><strong>Q80. Zero-shot learning কী?</strong><br/>
    Model এমন একটা task করে বা এমন class চেনে যার labeled example সে কখনো দেখেনি, সাধারণত related task থেকে শেখা auxiliary semantic information (class description, embedding, shared representation space) ব্যবহার করে।<br/>
    উদাহরণ: CLIP তার classification training set-এ zebra image না থাকা সত্ত্বেও একটা image-কে "a photo of a zebra" হিসেবে classify করতে পারে, কারণ এটা বিশাল web data থেকে একটা joint image-text embedding space শিখেছে।</p>

    <p><strong>Q81. Class imbalance কেন gradient update-কে affect করে?</strong><br/>
    Standard cross-entropy-তে, majority class-এর gradient contribution পুরো gradient-কে dominate করে শুধু এই কারণে যে majority example অনেক বেশি, তাই model মূলত majority-class accuracy-র জন্যই optimize হয়।
    <pre><code>import torch.nn as nn
class_weights = torch.tensor([1.0, 99.0])   # class frequency-র inverse (99:1 imbalance)
criterion = nn.CrossEntropyLoss(weight=class_weights)</code></pre></p>

    <p><strong>Q82. Focal loss এবং এর advantage explain করুন।</strong><br/>
    Focal loss cross-entropy-তে একটা (1-p)^γ modulating factor যোগ করে, easy, ভালোভাবে classified example-এর loss contribution কমিয়ে দেয় এবং hard/misclassified example-এ training focus করে — extreme class imbalance-এ (যেমন object detection-এ background বনাম foreground) কাজে লাগে।
    <pre><code>import torch
import torch.nn.functional as F

def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    return (alpha * (1 - p_t) ** gamma * ce).mean()</code></pre></p>

    <p><strong>Q83. Adversarial example কী?</strong><br/>
    একটা model-কে misclassification-এ ভুল করানোর জন্য specifically একটা ছোট, প্রায়ই imperceptible পরিমাণে deliberately perturb করা input, model-এর locally linear/fragile decision boundary exploit করে।
    <pre><code>x.requires_grad = True
loss = criterion(model(x), true_label)
loss.backward()
x_adv = x + epsilon * x.grad.sign()   # FGSM attack: gradient-এর sign বরাবর নাড়ানো</code></pre></p>

    <p><strong>Q84. Neural network ছোট perturbation-এর প্রতি কেন vulnerable?</strong><br/>
    High-dimensional linear-এর মতো decision boundary মানে gradient direction-এর সাথে align করা একটা ছোট perturbation অনেক dimension জুড়ে accumulate হয়ে output-এ একটা বড় change তৈরি করতে পারে, যদিও প্রতিটা individual input change imperceptible।<br/>
    উদাহরণ: একটা FGSM attack loss-gradient direction-এ প্রতিটা pixel-কে মাত্র ε=0.01 perturb করে — individually অদৃশ্য, কিন্তু 150,000 pixel জুড়ে যোগ করলে prediction flip করার জন্য যথেষ্ট।</p>

    <p><strong>Q85. Deep model-এ interpretability challenge explain করুন।</strong><br/>
    লক্ষ লক্ষ parameter অনেক layer জুড়ে nonlinearly interact করে, তাই "এই weight" থেকে "এই decision"-এ কোনো simple mapping নেই — linear model-এর বিপরীতে, deep model explanation এমন approximation যা unfaithful হতে পারে।<br/>
    উদাহরণ: একই image classification-এ দুটো explanation method (Grad-CAM বনাম saliency map) সম্পূর্ণ ভিন্ন region-কে "important" হিসেবে highlight করতে পারে।</p>

    <p><strong>Q86. SHAP বা LIME conceptually কী?</strong><br/>
    LIME একটা complex model-কে একটা prediction-এর চারপাশে locally একটা interpretable linear model দিয়ে approximate করে। SHAP game theory-র Shapley value-এর ভিত্তিতে প্রতিটা feature-কে একটা contribution দেয়, guarantee করে contribution-গুলো prediction-এ ঠিক যোগ হয়, LIME-এ যা নেই এমন consistency guarantee সহ।<br/>
    উদাহরণ: loan denial-এর জন্য, SHAP দেখাতে পারে: base rate 60% approval, −25% low credit score, −10% short employment history, +5% high income — সব যোগ করলে actual predicted probability 30%-এ পৌঁছায়।</p>

    <p><strong>Q87. Embedding কেন semantic meaning capture করে?</strong><br/>
    Embedding এমনভাবে train হয় যে similar context-এ আসা word/item শেষ পর্যন্ত similar vector পায় (distributional hypothesis) — embedding space-এ proximity তখন semantic similarity প্রতিফলিত করে।
    <pre><code>import torch.nn as nn
import torch.nn.functional as F

embedding = nn.Embedding(num_embeddings=50000, embedding_dim=128)
king, man, woman = embedding(torch.tensor([101, 205, 310]))
similarity = F.cosine_similarity((king - man + woman).unsqueeze(0), king.unsqueeze(0))</code></pre></p>

    <p><strong>Q88. Embedding size খুব বড় হলে কী হয়?</strong><br/>
    Overfitting risk বাড়ে (training-specific pattern মুখস্থ করার জন্য বেশি parameter, বিশেষত rare item-এর জন্য), training ধীর হয়, এবং memory বাড়ে — একটা নির্দিষ্ট size-এর পর downstream accuracy-তে diminishing বা এমনকি negative return।<br/>
    উদাহরণ: মাত্র 5,000টা unique item থাকা একটা recommender-এ 2048-dim embedding সম্ভবত overfit করে এবং একটা ভালোভাবে tune করা 64 বা 128-dim embedding-এর চেয়ে memory নষ্ট করে।</p>

    <p><strong>Q89. Multi-task learning-এর benefit ও risk explain করুন।</strong><br/>
    Related task জুড়ে representation share করলে generalization ভালো হতে পারে (implicit regularization) — কিন্তু task conflict করলে negative transfer-এর risk থাকে, এবং careful loss-weighting দরকার কারণ একটা task-এর gradient বাকিগুলোকে dominate করে ক্ষতি করতে পারে।
    <pre><code>total_loss = w1 * detection_loss + w2 * depth_loss + w3 * segmentation_loss
# w1 যদি w2, w3-এর তুলনায় খুব বড় হয়, সেই task-গুলো useful gradient থেকে বঞ্চিত হয়</code></pre></p>

    <p><strong>Q90. কিছু model কেন generalize না করে memorize করে?</strong><br/>
    Model capacity আসল pattern represent করতে যা দরকার তার চেয়ে বহুগুণ বেশি হলে এবং training data সীমিত/noisy হলে, optimizer training example literally store করে training loss প্রায় zero-তে নামাতে পারে, generalizable rule শেখার বদলে।<br/>
    উদাহরণ: সম্পূর্ণ randomize করা label দিয়ে CIFAR-10-এ train করা একটা বড় network তবুও ~100% training accuracy পেতে পারে — arbitrary label-image pair মুখস্থ করে (Zhang et al., "Understanding Deep Learning Requires Rethinking Generalization")।</p>

    <h3>Real-World &amp; System Design (Q91–Q100)</h3>

    <p><strong>Q91. একটা scalable training pipeline কীভাবে design করবেন?</strong><br/>
    একটা scalable store দিয়ে data ingest করুন → versioned feature store দিয়ে feature engineering → checkpointing সহ distributed training (data/model parallelism) → experiment tracking (MLflow/W&amp;B) → production-এ promote করার আগে automated evaluation gate।<br/>
    উদাহরণ: feature computation-এর জন্য Spark, data lake হিসেবে S3, training-এর জন্য 8 GPU জুড়ে PyTorch DDP, এবং MLflow-তে প্রতিটা run-এর hyperparameter/metric log করা যাতে যেকোনো পুরনো run reproducible থাকে।</p>

    <p><strong>Q92. Production system-এ missing data কীভাবে handle করবেন?</strong><br/>
    Training-এর সময় fit করা exact imputation logic reuse করুন (live data-তে কখনো নতুন করে stats calculate করবেন না), missing-value indicator feature যোগ করুন, sensible default set করুন, এবং missing-rate monitor করুন — একটা spike সাধারণত upstream pipeline bug বোঝায়, organic missingness নয়।<br/>
    উদাহরণ: training data-তে মাত্র 2% missing থাকা "age" যদি live request-এর 40%-এ missing হয়, সেটা বোঝায় production feature pipeline ভেঙে গেছে, user-রা age report করা বন্ধ করেনি।</p>

    <p><strong>Q93. Feature drift বনাম data drift explain করুন।</strong><br/>
    Data drift (covariate shift) হলো input feature distribution P(x)-এর যেকোনো change। Feature drift প্রায়ই individual feature distribution shift করাকে বোঝায়, কখনো কখনো feature কীভাবে define/compute হয় তার change-এর কারণে। দুটোই P(y|x) না বদলিয়েই ঘটতে পারে — সেটা concept drift।<br/>
    উদাহরণ: upstream unit change-এর কারণে "user age" হঠাৎ বছরের বদলে মাসে value দেখালে, সেটা feature drift — একটা definitional break, real demographic shift নয়।</p>

    <p><strong>Q94. Deployment-এর পরে model performance কীভাবে monitor করবেন?</strong><br/>
    Live prediction distribution, input feature drift (PSI/KS test), পাওয়া গেলে delayed ground-truth label, model-এর decision-এর সাথে জড়িত business KPI, এবং এসবের উপর alerting threshold track করুন।<br/>
    উদাহরণ: একটা fraud model dashboard daily flagged-transaction rate, feature PSI score, এবং (কয়েক সপ্তাহ পরে true label এলে) actual precision/recall track করে — কোনো external কারণ ছাড়াই flagged rate হঠাৎ 5× বাড়লে alert করে।</p>

    <p><strong>Q95. Training-এ reproducibility নিশ্চিত করতে কী কী step নেবেন?</strong><br/>
    Random seed fix করুন (data shuffling, weight init, dropout), exact library/dependency version pin করুন, training data snapshot version করুন, সব hyperparameter log করুন, এবং যেখানে সম্ভব deterministic operation ব্যবহার করুন।
    <pre><code>import torch, random, numpy as np
torch.manual_seed(42); random.seed(42); np.random.seed(42)
torch.use_deterministic_algorithms(True)</code></pre></p>

    <p><strong>Q96. Model inference-এ latency কীভাবে কমাবেন?</strong><br/>
    Quantization (fp32→int8), ছোট student model-এ knowledge distillation, pruning, request batching, optimized runtime (ONNX Runtime, TensorRT), এবং frequent prediction cache করা।
    <pre><code>import torch
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)   # সাধারণত ~2-4x দ্রুত inference, সামান্য accuracy drop</code></pre></p>

    <p><strong>Q97. Accuracy ও speed-এর মধ্যে trade-off explain করুন।</strong><br/>
    বেশি accurate model সাধারণত বড়/গভীর/ensemble হয়, যা সরাসরি inference latency ও serving cost বাড়ায়; সঠিক operating point SLA এবং marginal accuracy আসলে business metric কতটা নাড়ায় তার উপর নির্ভর করে।<br/>
    উদাহরণ: একটা 2-model ensemble (AUC 0.91, 200ms) থেকে single distilled model (AUC 0.89, 20ms)-এ যাওয়া strict 30ms budget থাকা real-time bidding system-এর জন্য সঠিক trade, যদিও কাগজে-কলমে এটা "কম accurate"।</p>

    <p><strong>Q98. হঠাৎ degrade হওয়া model কীভাবে debug করবেন?</strong><br/>
    প্রথমে upstream data pipeline breakage check করুন (সবচেয়ে common আসল কারণ), বর্তমান বনাম historical feature distribution compare করুন, কোনো silent schema/library version change হয়নি তা verify করুন, concept drift check করুন, এবং serving code training code-এর সাথে মেলে কিনা নিশ্চিত করুন (training-serving skew)।<br/>
    উদাহরণ: একটা recommendation model-এর CTR রাতারাতি crash করে — investigation-এ দেখা যায় schema migration-এর পর upstream ETL job "click_count" feature-এর জন্য 0-এর বদলে null পাঠানো শুরু করেছে।</p>

    <p><strong>Q99. Deep model deploy করার common pitfall কী কী?</strong><br/>
    Training-serving skew (আলাদা preprocessing code path), serving environment-এর latency/memory constraint ignore করা, model versioning/rollback-এর কোনো plan না থাকা, অপর্যাপ্ত monitoring, এবং launch-এর আগে edge case test না করা।<br/>
    উদাহরণ: PIL-এর bicubic interpolation দিয়ে resize করা image দিয়ে train করা একটা model OpenCV-এর default (ভিন্ন) resizing দিয়ে serve করা হয় — একটা subtle preprocessing mismatch যা production-এ প্রতিটা prediction-কে silently খারাপ করে দেয়।</p>

    <p><strong>Q100. একটা failing ML system কীভাবে সম্পূর্ণভাবে redesign করবেন?</strong><br/>
    (১) শুধু offline ML metric নয়, আসল business metric-এ re-anchor করুন। (২) leakage/drift/quality issue-এর জন্য data pipeline audit করুন। (৩) model architecture/objective আসল সমস্যার সাথে মেলে কিনা re-evaluate করুন। (৪) production condition প্রতিফলিত করে evaluation rebuild করুন। (৫) continuous monitoring + feedback loop যোগ করুন যাতে পরের বার failure তাড়াতাড়ি ধরা পড়ে।<br/>
    উদাহরণ: একটা "successful" churn model (AUC 0.85) যা কখনো actual churn কমায়নি, তাকে আসল KPI (retained revenue)-এর চারপাশে redesign করা হয়, properly time-split data-তে re-train করা হয়, এবং real retention offer trigger করার permission দেওয়ার আগে shadow-mode test দিয়ে ship করা হয়।</p>
  `
};
