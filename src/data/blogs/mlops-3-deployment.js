export const mlops_3_deployment = {
  slug: 'mlops-3-deployment',
  title: 'মডেল ডিপ্লয়মেন্ট: FastAPI ও Docker',
  description: 'pickle/joblib/ONNX দিয়ে মডেল সেভ করা, FastAPI REST API বানানো, Docker কন্টেইনারাইজেশন এবং সম্পূর্ণ ডিপ্লয়মেন্ট উদাহরণ।',
  date: 'মে ২০২৬',
  category: 'এমএলঅপস',
  readTime: 16,
  content: `
    <h3>১. মডেল সেভ করার পদ্ধতি</h3>
    <p>
      ML মডেল ট্রেন করার পর সেটিকে ডিস্কে সেভ করতে হয় যাতে পরবর্তীতে লোড করে প্রেডিকশন করা যায়। তিনটি প্রধান পদ্ধতি আছে:
    </p>

    <h4>Pickle দিয়ে সেভ করা</h4>
    <pre><code">import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# মডেল ট্রেন করুন
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Pickle দিয়ে সেভ করুন
with open("models/iris_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Pickle থেকে লোড করুন
with open("models/iris_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

print(f"Loaded Model Score: {loaded_model.score(X_test, y_test):.4f}")</code></pre>

    <h4>Joblib দিয়ে সেভ করা (NumPy array-এর জন্য দ্রুততর)</h4>
    <pre><code">import joblib

# সেভ করুন
joblib.dump(model, "models/iris_model.joblib", compress=3)

# লোড করুন
loaded_model = joblib.load("models/iris_model.joblib")
print(f"Joblib Model Score: {loaded_model.score(X_test, y_test):.4f}")

# পুরো পাইপলাইন সেভ করুন
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])
pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "models/iris_pipeline.joblib")
loaded_pipeline = joblib.load("models/iris_pipeline.joblib")
print(f"Pipeline Score: {loaded_pipeline.score(X_test, y_test):.4f}")</code></pre>

    <h4>ONNX ফরম্যাট (ভাষা-নিরপেক্ষ)</h4>
    <pre><code">pip install skl2onnx onnxruntime</code></pre>
    <pre><code">from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
import numpy as np

# ONNX-এ কনভার্ট করুন
initial_type = [("float_input", FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

with open("models/iris_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# ONNX Runtime দিয়ে প্রেডিক্ট করুন
sess = rt.InferenceSession("models/iris_model.onnx")
input_name = sess.get_inputs()[0].name
pred_onnx = sess.run(None, {input_name: X_test.astype(np.float32)})
print(f"ONNX Predictions: {pred_onnx[0][:5]}")</code></pre>

    <h3>২. FastAPI দিয়ে ML REST API</h3>
    <p>
      <strong>FastAPI</strong> হলো Python-এর সবচেয়ে দ্রুত ও আধুনিক REST API ফ্রেমওয়ার্ক। এটি স্বয়ংক্রিয়ভাবে API documentation তৈরি করে এবং type validation করে।
    </p>
    <pre><code">pip install fastapi uvicorn pydantic scikit-learn joblib</code></pre>

    <pre><code"># app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import joblib
import numpy as np
import logging
from contextlib import asynccontextmanager

# লগার সেটআপ
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# মডেল গ্লোবাল ভ্যারিয়েবল
ml_model = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """অ্যাপ শুরু ও শেষে মডেল লোড/আনলোড"""
    logger.info("মডেল লোড হচ্ছে...")
    ml_model["classifier"] = joblib.load("models/iris_pipeline.joblib")
    logger.info("মডেল লোড সম্পন্ন")
    yield
    ml_model.clear()
    logger.info("মডেল আনলোড সম্পন্ন")

app = FastAPI(
    title="Iris Classification API",
    description="ML মডেল দিয়ে Iris ফুলের প্রজাতি শ্রেণীবদ্ধ করুন",
    version="1.0.0",
    lifespan=lifespan
)

# Request schema
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0, le=20, description="বৃতির দৈর্ঘ্য (cm)")
    sepal_width: float = Field(..., ge=0, le=20, description="বৃতির প্রস্থ (cm)")
    petal_length: float = Field(..., ge=0, le=20, description="পাপড়ির দৈর্ঘ্য (cm)")
    petal_width: float = Field(..., ge=0, le=20, description="পাপড়ির প্রস্থ (cm)")

class BatchRequest(BaseModel):
    samples: List[IrisFeatures]

# Response schema
class PredictionResponse(BaseModel):
    prediction: int
    species: str
    confidence: float

class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int

SPECIES_MAP = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

@app.get("/", tags=["Health"])
async def root():
    return {"status": "healthy", "message": "Iris Classification API চলছে"}

@app.get("/health", tags=["Health"])
async def health_check():
    model_loaded = "classifier" in ml_model
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: IrisFeatures):
    """একটি নমুনার প্রজাতি অনুমান করুন"""
    if "classifier" not in ml_model:
        raise HTTPException(status_code=503, detail="মডেল লোড হয়নি")

    try:
        X = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]])

        model = ml_model["classifier"]
        prediction = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        confidence = float(proba[prediction])

        logger.info(f"Prediction: {SPECIES_MAP[prediction]}, Confidence: {confidence:.4f}")

        return PredictionResponse(
            prediction=prediction,
            species=SPECIES_MAP[prediction],
            confidence=confidence
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
async def predict_batch(request: BatchRequest):
    """একাধিক নমুনার প্রেডিকশন"""
    if "classifier" not in ml_model:
        raise HTTPException(status_code=503, detail="মডেল লোড হয়নি")

    X = np.array([[
        s.sepal_length, s.sepal_width, s.petal_length, s.petal_width
    ] for s in request.samples])

    model = ml_model["classifier"]
    predictions = model.predict(X)
    probas = model.predict_proba(X)

    results = [
        PredictionResponse(
            prediction=int(pred),
            species=SPECIES_MAP[int(pred)],
            confidence=float(probas[i][int(pred)])
        )
        for i, pred in enumerate(predictions)
    ]

    return BatchResponse(predictions=results, count=len(results))</code></pre>

    <pre><code"># অ্যাপ চালান
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# API টেস্ট করুন
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'</code></pre>

    <h3>৩. Docker: কন্টেইনারাইজেশন</h3>
    <p>
      <strong>Docker</strong> ML অ্যাপ্লিকেশনকে একটি বহনযোগ্য কন্টেইনারে প্যাকেজ করে। যেকোনো মেশিনে একই environment-এ অ্যাপ চলবে।
    </p>

    <h4>প্রজেক্ট স্ট্রাকচার</h4>
    <pre><code">project/
├── app/
│   ├── __init__.py
│   └── main.py
├── models/
│   └── iris_pipeline.joblib
├── requirements.txt
├── Dockerfile
└── docker-compose.yml</code></pre>

    <h4>Dockerfile লেখা</h4>
    <pre><code"># Dockerfile
# পর্যায় ১: Base image
FROM python:3.11-slim AS base

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# পর্যায় ২: Dependencies ইনস্টল
FROM base AS dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# পর্যায় ৩: Production image
FROM dependencies AS production

# Non-root user তৈরি করুন (নিরাপত্তার জন্য)
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser

# কোড কপি করুন
COPY app/ ./app/
COPY models/ ./models/

# Ownership পরিবর্তন করুন
RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]</code></pre>

    <h4>requirements.txt</h4>
    <pre><code">fastapi==0.111.0
uvicorn[standard]==0.30.1
pydantic==2.7.4
scikit-learn==1.5.0
joblib==1.4.2
numpy==1.26.4</code></pre>

    <h4>Docker image তৈরি ও চালানো</h4>
    <pre><code"># Image বিল্ড করুন
docker build -t iris-ml-api:latest .

# Container চালান
docker run -d \
  --name iris-api \
  -p 8000:8000 \
  --memory="512m" \
  --cpus="1.0" \
  iris-ml-api:latest

# লগ দেখুন
docker logs iris-api -f

# Container-এ প্রবেশ করুন
docker exec -it iris-api bash

# বন্ধ করুন
docker stop iris-api && docker rm iris-api</code></pre>

    <h3>৪. Docker Compose দিয়ে Multi-Service Setup</h3>
    <pre><code"># docker-compose.yml
version: '3.9'

services:
  ml-api:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    image: iris-ml-api:latest
    container_name: iris-api
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/iris_pipeline.joblib
      - LOG_LEVEL=info
    volumes:
      - ./models:/app/models:ro
    healthcheck:
      test: ["CMD", "python", "-c",
             "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '1.0'

  nginx:
    image: nginx:alpine
    container_name: nginx-proxy
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf:ro
    depends_on:
      - ml-api
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

volumes:
  model_cache:</code></pre>

    <pre><code"># চালানো
docker-compose up -d

# বন্ধ করা
docker-compose down

# লগ দেখা
docker-compose logs ml-api -f

# স্কেল আপ করা
docker-compose up -d --scale ml-api=3</code></pre>

    <h3>৫. Batch vs Online Serving প্যাটার্ন</h3>
    <p>মডেল সার্ভিং দুটি প্রধান প্যাটার্নে হয়:</p>

    <h4>Online Serving (Real-time)</h4>
    <pre><code"># একটি রিকোয়েস্টে একটি প্রেডিকশন (milliseconds মধ্যে)
# উপযুক্ত: recommendation, fraud detection, chatbot

import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"sepal_length": 5.1, "sepal_width": 3.5,
          "petal_length": 1.4, "petal_width": 0.2}
)
print(response.json())</code></pre>

    <h4>Batch Serving (Offline)</h4>
    <pre><code"># অনেক ডেটা একসাথে প্রসেস (minutes/hours)
# উপযুক্ত: daily reports, email recommendations

import pandas as pd
import joblib

def batch_predict(input_path: str, output_path: str):
    """CSV ফাইল থেকে batch prediction"""
    df = pd.read_csv(input_path)
    model = joblib.load("models/iris_pipeline.joblib")

    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X = df[features].values

    df["prediction"] = model.predict(X)
    df["confidence"] = model.predict_proba(X).max(axis=1)

    df.to_csv(output_path, index=False)
    print(f"Batch prediction complete: {len(df)} samples")
    return df

# চালান
results = batch_predict("data/new_flowers.csv", "output/predictions.csv")</code></pre>

    <h3>৬. সম্পূর্ণ ডিপ্লয়মেন্ট উদাহরণ</h3>
    <p>একটি সম্পূর্ণ deployment script যা মডেল বিল্ড থেকে প্রোডাকশন পর্যন্ত সব কভার করে:</p>
    <pre><code"># scripts/deploy.sh
#!/bin/bash
set -e

# কনফিগ
APP_NAME="iris-ml-api"
VERSION=\$(git rev-parse --short HEAD)
REGISTRY="registry.company.com"

echo "Building version: \$VERSION"

# ইউনিট টেস্ট
echo "Running tests..."
python -m pytest tests/ -v --tb=short

# Docker image বিল্ড
echo "Building Docker image..."
docker build -t \$APP_NAME:\$VERSION .
docker tag \$APP_NAME:\$VERSION \$REGISTRY/\$APP_NAME:\$VERSION
docker tag \$APP_NAME:\$VERSION \$REGISTRY/\$APP_NAME:latest

# Registry-তে পুশ
echo "Pushing to registry..."
docker push \$REGISTRY/\$APP_NAME:\$VERSION
docker push \$REGISTRY/\$APP_NAME:latest

# Health check
echo "Running health check..."
docker run --rm -d --name test-container -p 8001:8000 \$APP_NAME:\$VERSION
sleep 5
curl -f http://localhost:8001/health || (docker stop test-container && exit 1)
docker stop test-container

echo "Deployment successful! Version: \$VERSION"</code></pre>

    <h3>৭. সারসংক্ষেপ</h3>
    <p>এই অধ্যায়ে আমরা শিখলাম:</p>
    <ul>
      <li><strong>Pickle, Joblib, ONNX</strong> দিয়ে মডেল সেভ করার পার্থক্য ও ব্যবহার</li>
      <li><strong>FastAPI</strong> দিয়ে type-safe REST API তৈরি করা</li>
      <li><strong>Dockerfile</strong> লিখে multi-stage build করা</li>
      <li><strong>Docker Compose</strong> দিয়ে multi-service অ্যাপ চালানো</li>
      <li><strong>Batch vs Online</strong> serving প্যাটার্নের পার্থক্য</li>
    </ul>
    <p>পরবর্তী অধ্যায়ে আমরা শিখব কীভাবে প্রোডাকশন মডেলের পারফরম্যান্স মনিটর করতে হয় এবং data drift ডিটেক্ট করতে হয়।</p>
  `
};
