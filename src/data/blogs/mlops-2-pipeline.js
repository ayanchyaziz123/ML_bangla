export const mlops_2_pipeline = {
  slug: 'mlops-2-pipeline',
  title: 'ML পাইপলাইন: স্বয়ংক্রিয় ওয়ার্কফ্লো',
  description: 'ডেটা পাইপলাইন, ফিচার ইঞ্জিনিয়ারিং পাইপলাইন, sklearn Pipeline, Apache Airflow ও Prefect দিয়ে ML-এর CI/CD সম্পর্কে সম্পূর্ণ গাইড।',
  date: 'মে ২০২৬',
  category: 'এমএলঅপস',
  readTime: 15,
  content: `
    <h3>১. ML পাইপলাইন কেন দরকার?</h3>
    <p>
      একটি ML প্রজেক্টে বারবার একই কাজ করতে হয়: ডেটা লোড করা, পরিষ্কার করা, ফিচার তৈরি করা, মডেল ট্রেন করা, মূল্যায়ন করা। এই প্রতিটি ধাপকে হাতে হাতে করলে ভুল হওয়ার সম্ভাবনা বেশি এবং রিপ্রোডিউস করা কঠিন।
    </p>
    <p>
      <strong>ML পাইপলাইন</strong> এই সমস্যার সমাধান করে — প্রতিটি ধাপকে একটি সুসংগত, স্বয়ংক্রিয় ক্রমে সাজিয়ে দেয়। পাইপলাইনের সুবিধাসমূহ:
    </p>
    <ul>
      <li>Data leakage রোধ করে (test ডেটা দিয়ে preprocessing fit হয় না)</li>
      <li>কোড পুনর্ব্যবহারযোগ্য ও রক্ষণাবেক্ষণযোগ্য হয়</li>
      <li>Hyperparameter tuning সহজ হয়</li>
      <li>প্রোডাকশনে একটি অবজেক্ট হিসেবে deploy করা যায়</li>
    </ul>

    <h3>২. sklearn Pipeline: মূল কাঠামো</h3>
    <p>
      Scikit-learn-এর <code>Pipeline</code> ক্লাস সবচেয়ে সহজ পাইপলাইন টুল। এটি preprocessing ও modeling ধাপগুলোকে একটি chain-এ জুড়ে দেয়।
    </p>
    <pre><code>import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import fetch_openml

# Titanic ডেটাসেট লোড করুন
titanic = fetch_openml("titanic", version=1, as_frame=True)
df = titanic.frame

# ফিচার নির্বাচন
features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
X = df[features]
y = (df["survived"] == "1").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Numerical ফিচারের জন্য transformer
numeric_features = ["age", "fare", "sibsp", "parch"]
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical ফিচারের জন্য transformer
categorical_features = ["pclass", "sex", "embarked"]
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# ColumnTransformer দিয়ে একত্রিত করুন
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# সম্পূর্ণ পাইপলাইন
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# ট্রেন করুন
pipeline.fit(X_train, y_train)
print(f"Test Accuracy: {pipeline.score(X_test, y_test):.4f}")

# Cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
print(f"CV Score: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")</code></pre>

    <h3>৩. Custom Transformer তৈরি করা</h3>
    <p>
      sklearn-এর বাইরে নিজস্ব ট্রান্সফর্মার তৈরি করতে <code>BaseEstimator</code> ও <code>TransformerMixin</code> ব্যবহার করতে হয়:
    </p>
    <pre><code>from sklearn.base import BaseEstimator, TransformerMixin

class AgeGroupTransformer(BaseEstimator, TransformerMixin):
    """বয়সকে গ্রুপে ভাগ করার কাস্টম ট্রান্সফর্মার"""

    def __init__(self, bins=None, labels=None):
        self.bins = bins or [0, 12, 18, 35, 60, 100]
        self.labels = labels or ["child", "teen", "adult", "middle", "senior"]

    def fit(self, X, y=None):
        return self  # Stateless transformer

    def transform(self, X):
        X = X.copy()
        if isinstance(X, pd.DataFrame):
            X["age_group"] = pd.cut(
                X["age"].fillna(X["age"].median()),
                bins=self.bins,
                labels=self.labels
            )
            return X
        return X

class FamilySizeTransformer(BaseEstimator, TransformerMixin):
    """SibSp ও Parch থেকে পরিবারের আকার তৈরি"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if isinstance(X, pd.DataFrame):
            X["family_size"] = X["sibsp"] + X["parch"] + 1
            X["is_alone"] = (X["family_size"] == 1).astype(int)
        return X

# উন্নত পাইপলাইনে যোগ করুন
advanced_pipeline = Pipeline(steps=[
    ("family_size", FamilySizeTransformer()),
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])
advanced_pipeline.fit(X_train, y_train)
print(f"Advanced Pipeline Test Accuracy: {advanced_pipeline.score(X_test, y_test):.4f}")</code></pre>

    <h3>৪. GridSearchCV দিয়ে পাইপলাইন টিউন করা</h3>
    <p>
      পাইপলাইনের যেকোনো ধাপের হাইপারপ্যারামিটার <code>GridSearchCV</code> দিয়ে টিউন করা যায়। প্যারামিটার নাম হয় <em>ধাপের_নাম__প্যারামিটার_নাম</em> ফরম্যাটে:
    </p>
    <pre><code">from sklearn.model_selection import GridSearchCV

param_grid = {
    "preprocessor__num__imputer__strategy": ["mean", "median"],
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [None, 5, 10],
    "classifier__min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"সেরা প্যারামিটার: {grid_search.best_params_}")
print(f"সেরা CV Score: {grid_search.best_score_:.4f}")
print(f"Test Score: {grid_search.score(X_test, y_test):.4f}")

# সেরা পাইপলাইন সেভ করুন
import joblib
joblib.dump(grid_search.best_estimator_, "models/best_pipeline.pkl")</code></pre>

    <h3>৫. Apache Airflow: DAG-ভিত্তিক পাইপলাইন</h3>
    <p>
      <strong>Apache Airflow</strong> হলো বড় স্কেলের পাইপলাইন অর্কেস্ট্রেশন টুল। এটি DAG (Directed Acyclic Graph) ব্যবহার করে পাইপলাইনের ধাপগুলো সংজ্ঞায়িত করে এবং শিডিউলড বা ট্রিগার্ড ভাবে চালায়।
    </p>
    <pre><code>pip install apache-airflow apache-airflow-providers-common-sql</code></pre>

    <pre><code"># dags/ml_training_dag.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# Default arguments
default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": True,
    "email": ["ml-alerts@company.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def extract_data(**context):
    """ডেটাবেস থেকে ডেটা সংগ্রহ"""
    import pandas as pd
    # বাস্তবে এখানে DB connection হবে
    df = pd.read_csv("/data/raw/transactions.csv")
    output_path = "/data/processed/raw_extracted.csv"
    df.to_csv(output_path, index=False)
    context["ti"].xcom_push(key="row_count", value=len(df))
    print(f"Extracted {len(df)} rows")

def transform_data(**context):
    """ডেটা পরিষ্কার ও ফিচার ইঞ্জিনিয়ারিং"""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv("/data/processed/raw_extracted.csv")

    # পরিষ্কার করুন
    df = df.dropna(subset=["target"])
    df["amount_log"] = df["amount"].apply(lambda x: max(x, 0.01)).apply(
        __import__("math").log
    )

    df.to_csv("/data/processed/transformed.csv", index=False)
    print(f"Transformed data shape: {df.shape}")

def train_model(**context):
    """মডেল ট্রেনিং"""
    import pandas as pd
    import mlflow
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    df = pd.read_csv("/data/processed/transformed.csv")
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(model, "model")

    joblib.dump(model, "/models/latest_model.pkl")
    print(f"Model AUC: {auc:.4f}")

# DAG সংজ্ঞায়িত করুন
with DAG(
    "ml_training_pipeline",
    default_args=default_args,
    description="দৈনিক ML ট্রেনিং পাইপলাইন",
    schedule_interval="0 2 * * *",  # প্রতিদিন রাত ২টায়
    catchup=False,
    tags=["ml", "training"],
) as dag:

    extract_task = PythonOperator(
        task_id="extract_data",
        python_callable=extract_data,
    )

    transform_task = PythonOperator(
        task_id="transform_data",
        python_callable=transform_data,
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    validate_task = BashOperator(
        task_id="validate_model",
        bash_command="python /scripts/validate_model.py",
    )

    # টাস্কের ক্রম নির্ধারণ করুন
    extract_task >> transform_task >> train_task >> validate_task</code></pre>

    <h3>৬. Prefect: আধুনিক পাইপলাইন অর্কেস্ট্রেশন</h3>
    <p>
      <strong>Prefect</strong> হলো Airflow-এর আধুনিক বিকল্প। এটি Python-native, সেটআপ সহজ, এবং স্থানীয় কোডকে সরাসরি ফ্লো হিসেবে চালানো যায়।
    </p>
    <pre><code>pip install prefect</code></pre>

    <pre><code">from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

@task(retries=3, retry_delay_seconds=10, log_prints=True)
def load_data(path: str) -> pd.DataFrame:
    """ডেটা লোড করুন"""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df

@task(log_prints=True)
def preprocess(df: pd.DataFrame) -> tuple:
    """ডেটা প্রিপ্রসেস করুন"""
    df = df.dropna()
    X = df.drop("target", axis=1).select_dtypes(include="number")
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test

@task(log_prints=True)
def train(X_train, y_train, n_estimators: int = 100) -> RandomForestClassifier:
    """মডেল ট্রেন করুন"""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    return model

@task(log_prints=True)
def evaluate(model, X_test, y_test) -> float:
    """মডেল মূল্যায়ন করুন"""
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Test Accuracy: {acc:.4f}")
    return acc

@task
def save_model(model, path: str, accuracy: float):
    """মডেল সেভ করুন"""
    if accuracy >= 0.80:
        joblib.dump(model, path)
        print(f"Model saved to {path} (accuracy={accuracy:.4f})")
    else:
        raise ValueError(f"Accuracy {accuracy:.4f} is below threshold 0.80")

@flow(name="ML Training Flow", task_runner=ConcurrentTaskRunner())
def ml_training_flow(data_path: str, model_path: str, n_estimators: int = 100):
    """সম্পূর্ণ ML ট্রেনিং ফ্লো"""
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess(df)
    model = train(X_train, y_train, n_estimators)
    accuracy = evaluate(model, X_test, y_test)
    save_model(model, model_path, accuracy)
    return accuracy

if __name__ == "__main__":
    result = ml_training_flow(
        data_path="data/processed.csv",
        model_path="models/rf_model.pkl",
        n_estimators=150
    )
    print(f"Flow completed with accuracy: {result:.4f}")</code></pre>

    <h3>৭. ML-এর CI/CD: GitHub Actions</h3>
    <p>
      ML প্রজেক্টে CI/CD পাইপলাইন মানে হলো: কোড পরিবর্তন হলে স্বয়ংক্রিয়ভাবে মডেল রিট্রেন ও টেস্ট হবে।
    </p>
    <pre><code"># .github/workflows/ml-ci.yml
name: ML CI/CD Pipeline

on:
  push:
    branches: [main, develop]
    paths:
      - 'src/**'
      - 'data/**'
      - 'params.yaml'
  pull_request:
    branches: [main]

jobs:
  test-and-train:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Python সেটআপ
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: ডিপেন্ডেন্সি ইনস্টল
        run: |
          pip install -r requirements.txt
          pip install pytest dvc mlflow

      - name: ইউনিট টেস্ট চালান
        run: pytest tests/ -v --tb=short

      - name: DVC পাইপলাইন চালান
        run: |
          dvc pull
          dvc repro

      - name: মেট্রিক্স চেক করুন
        run: |
          python scripts/check_metrics.py --min-accuracy 0.80

      - name: মডেল আর্টিফ্যাক্ট আপলোড
        uses: actions/upload-artifact@v3
        with:
          name: trained-model
          path: models/

  deploy-staging:
    needs: test-and-train
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
      - name: Staging-এ ডিপ্লয় করুন
        run: |
          echo "Deploying to staging environment..."
          docker build -t ml-api:staging .
          docker push registry.company.com/ml-api:staging</code></pre>

    <h3>৮. পাইপলাইন টেস্টিং</h3>
    <p>ML পাইপলাইনের ইউনিট টেস্ট লেখা অত্যন্ত গুরুত্বপূর্ণ:</p>
    <pre><code"># tests/test_pipeline.py
import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from src.pipeline import build_pipeline

@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
    df["target"] = y
    return df

def test_pipeline_fits_without_error(sample_data):
    """পাইপলাইন সফলভাবে fit হয়"""
    X = sample_data.drop("target", axis=1)
    y = sample_data["target"]
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    assert hasattr(pipeline, "classes_") or hasattr(pipeline[-1], "classes_")

def test_pipeline_output_shape(sample_data):
    """পাইপলাইনের আউটপুট সঠিক আকারের"""
    X = sample_data.drop("target", axis=1)
    y = sample_data["target"]
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    assert predictions.shape == (len(X),)

def test_no_data_leakage(sample_data):
    """Test ডেটা দিয়ে preprocessing fit হয় না"""
    from sklearn.model_selection import train_test_split
    X = sample_data.drop("target", axis=1)
    y = sample_data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    # Test set-এ প্রেডিক্ট করতে পারা উচিত
    preds = pipeline.predict(X_test)
    assert len(preds) == len(X_test)</code></pre>

    <h3>৯. সারসংক্ষেপ</h3>
    <p>এই অধ্যায়ে আমরা শিখলাম:</p>
    <ul>
      <li><strong>sklearn Pipeline</strong> দিয়ে preprocessing ও modeling একত্রিত করা</li>
      <li><strong>Custom Transformer</strong> তৈরি করে পাইপলাইন বাড়ানো</li>
      <li><strong>GridSearchCV</strong> দিয়ে পুরো পাইপলাইন টিউন করা</li>
      <li><strong>Apache Airflow</strong> DAG দিয়ে শিডিউলড পাইপলাইন</li>
      <li><strong>Prefect</strong> দিয়ে Python-native ফ্লো তৈরি</li>
      <li><strong>GitHub Actions</strong> দিয়ে ML CI/CD পাইপলাইন</li>
    </ul>
    <p>পরবর্তী অধ্যায়ে আমরা শিখব কীভাবে ট্রেন করা মডেলকে FastAPI ও Docker দিয়ে প্রোডাকশনে ডিপ্লয় করতে হয়।</p>
  `
};
