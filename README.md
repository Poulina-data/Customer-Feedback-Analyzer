# Customer Feedback Analyzer

An NLP-based Machine Learning project that analyzes customer reviews and classifies them into **Positive, Negative, and Neutral** sentiments.

The project follows an end-to-end NLP pipeline, covering data preparation, text preprocessing, feature extraction, model training, evaluation, and API deployment using FastAPI.

## Project Overview

Customer reviews contain valuable information about product quality and customer satisfaction. Manually analyzing thousands of reviews is time-consuming.

This project automates sentiment classification using Natural Language Processing (NLP) and Machine Learning techniques.

## Key Features

- **Sentiment Analysis:** Classifies customer reviews into Positive, Negative, and Neutral categories.
- **Sentiment Label Generation:** Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) to generate sentiment labels.
- **Text Preprocessing:** Performs lowercasing, URL removal, stop-word removal, and lemmatization using spaCy.
- **Negation Handling:** Preserves important negation words such as "not", "no", and "never".
- **Feature Extraction:** Implements TF-IDF and Word2Vec approaches.
- **Model Training:** Trains machine learning models using extracted text features.
- **Model Evaluation:** Evaluates model performance using classification reports and confusion matrices.
- **API Deployment:** Exposes sentiment predictions through a FastAPI REST API.

## Project Architecture

```text
Customer-Feedback-Analyzer/
│
├── data/
│   ├── raw/
│   │   └── Reviews.csv
│   └── processed/
│       └── cleaned_reviews.csv
│
├── models/
│   ├── sentiment_model.pkl
│   ├── sentiment_model_w2v.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│
├── src/
│   ├── api/
│   │   └── app.py
│   │
│   ├── data/
│   │   ├── load_data.py
│   │   ├── preprocess.py
│   │   └── process_dataset.py
│   │
│   ├── features/
│   │   ├── vectorizer.py
│   │   └── word2vec_vectorizer.py
│   │
│   ├── models/
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   ├── train.py
│   │   └── train_w2v.py
│   │
│   └── utils/
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## NLP Pipeline

The project follows these stages:

**1. Data Loading**
- Load customer reviews from the CSV dataset.
- Inspect and prepare review text.

**2. Sentiment Label Generation**
- Apply VADER sentiment analysis.
- Generate Positive, Negative, and Neutral labels using compound score thresholds:
  - Positive: Score ≥ 0.05
  - Negative: Score ≤ -0.05
  - Neutral: Score between -0.05 and 0.05

**3. Text Preprocessing**
- Convert text to lowercase.
- Remove URLs and unwanted characters.
- Remove stop words while preserving negations.
- Apply lemmatization using spaCy.

**4. Feature Extraction**
- TF-IDF: Converts text into numerical feature vectors based on word importance.
- Word2Vec: Represents words as dense numerical vectors based on semantic relationships.

**5. Model Training**
- Train machine learning models using the extracted features.
- Save trained models for future predictions.

**6. Model Evaluation**
- Generate classification reports.
- Analyze precision, recall, and F1-score.
- Visualize model performance using confusion matrices.

**7. Model Deployment**
- Deploy the trained sentiment model using FastAPI.
- Accept customer reviews through a REST API.
- Return predicted sentiment in JSON format.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| NLP | NLTK, spaCy |
| Sentiment Analysis | VADER |
| Feature Extraction | TF-IDF, Word2Vec |
| Machine Learning | Scikit-learn |
| Model Serialization | Joblib |
| API Deployment | FastAPI, Uvicorn |
| Development Environment | VS Code, Jupyter Notebook |
| Version Control | Git, GitHub |

## Installation and Setup

**1. Clone the repository**

```bash
git clone https://github.com/Poulina-data/Customer-Feedback-Analyzer.git
```

**2. Navigate to the project directory**

```bash
cd Customer-Feedback-Analyzer
```

**3. Create a virtual environment**

```bash
python -m venv .venv
```

**4. Activate the virtual environment**

Windows:

```bash
.venv\Scripts\activate
```

**5. Install dependencies**

```bash
pip install -r requirements.txt
```

**6. Download the spaCy English language model**

```bash
python -m spacy download en_core_web_sm
```

## Dataset Setup

The raw and processed CSV datasets are excluded from this GitHub repository because of their large file sizes.

Place your dataset in the following location:

```text
data/raw/Reviews.csv
```

The dataset is required to execute the complete data preparation and training pipeline.

## Running the Project

**1. Process the dataset**

```bash
python main.py
```

This executes the data preparation pipeline configured in `main.py`.

**2. Train the TF-IDF model**

```bash
python -m src.models.train
```

**3. Train the Word2Vec model**

```bash
python -m src.models.train_w2v
```

**4. Evaluate the model**

```bash
python -m src.models.evaluate
```

**5. Run the FastAPI application**

```bash
uvicorn src.api.app:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

## API Usage

**Endpoint:** `POST /predict`

**Request:**

```json
{
    "text": "This product is amazing and works perfectly!"
}
```

**Response:**

```json
{
    "sentiment": "positive"
}
```

The API accepts customer review text and returns the predicted sentiment.

## Future Improvements

- Improve neutral sentiment classification.
- Introduce a mixed sentiment category.
- Experiment with advanced NLP models such as LSTM and BERT.
- Improve model performance through hyperparameter tuning.
- Build an interactive sentiment analysis dashboard.
- Deploy the API to a cloud platform.

## Author

**Poulina I.**

Data Scientist | AI & Machine Learning Enthusiast

GitHub: https://github.com/Poulina-data

---

*This project demonstrates an end-to-end Natural Language Processing and Machine Learning workflow, from raw customer reviews to deployed sentiment predictions.*
