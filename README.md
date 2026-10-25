# BERT-Based Movie Review Sentiment Analysis

A binary sentiment classification project using a pretrained **BERT (`bert-base-uncased`)** model to classify IMDb movie reviews as positive or negative. The model is fine-tuned using PyTorch and Hugging Face Transformers.

## Project Overview

Understanding sentiment requires more than identifying individual words. For example, “This is a good movie” and “This is not a good movie” express different sentiments despite sharing similar words.

Traditional word-frequency approaches may struggle to capture such contextual differences. This project explores how a pretrained Transformer model can learn contextual representations and perform sentiment classification.

**Objective:** Predict whether a movie review expresses positive or negative sentiment.

## Key Features

- IMDb movie review sentiment classification
- Pretrained BERT model fine-tuning
- BERT tokenization with input IDs and attention masks
- Binary classification using a sequence-classification head
- Separate training, validation and test sets
- Evaluation using accuracy, precision, recall and F1-score
- Confusion matrix and classification report
- Sentiment prediction on custom reviews
- Saving and reloading the trained model

## Tech Stack

- **Language:** Python
- **Deep Learning Framework:** PyTorch
- **NLP Model:** BERT (`bert-base-uncased`)
- **Model Library:** Hugging Face Transformers
- **Dataset Library:** Hugging Face Datasets
- **Evaluation:** Scikit-learn
- **Visualization:** Matplotlib
- **Environment:** Google Colab with GPU support

## Dataset

The project uses the IMDb movie review dataset, which contains 50,000 labelled movie reviews: 25,000 in the original training split and 25,000 in the original test split.

For this experiment, a reproducible subset of 10,000 reviews was selected to keep training manageable.

| Split | Number of reviews | Purpose |
|---|---:|---|
| Training | 7,200 | Fine-tune model parameters |
| Validation | 800 | Select the best model checkpoint |
| Testing | 2,000 | Final held-out evaluation |
| **Total used** | **10,000** | |

### Sampling and Labels

The original training and test splits were shuffled independently using a fixed random seed (`42`). The first 8,000 examples from the shuffled training split and the first 2,000 examples from the shuffled test split were selected.

The 8,000-example training pool was then divided into training and validation sets using a 90:10 split with stratification.

The sentiment labels are:

- `0` — Negative
- `1` — Positive

The selected training pool contained 3,988 negative reviews and 4,012 positive reviews, indicating a nearly balanced class distribution.

**Note:** The experiment uses a subset of the original 50,000-review dataset, not the full dataset.

## Methodology

### 1. Data Loading and Exploration

Loaded the IMDb dataset, selected the subsets, inspected sample reviews and labels, and visualized the training pool's class distribution.

### 2. BERT Tokenization

Reviews were tokenized using the tokenizer associated with `bert-base-uncased`.

The tokenizer produces:

- **Input IDs:** Numerical IDs representing tokens.
- **Attention masks:** Indicate which tokens should be attended to rather than treated as padding.

Long reviews are truncated to a maximum sequence length of 256 tokens. Dynamic padding is used within batches.

### 3. Model Architecture

The project uses a pretrained BERT encoder with a sequence-classification head for two sentiment classes.

The input tokens pass through BERT's bidirectional Transformer encoder, which generates contextual representations. The classification model uses the `[CLS]` representation to produce logits for negative and positive sentiment.

The pretrained model is fine-tuned on the labelled IMDb reviews rather than trained from scratch.

### 4. Model Training

The model was fine-tuned using the following configuration:

| Parameter | Value |
|---|---|
| Pretrained model | `bert-base-uncased` |
| Number of classes | 2 |
| Learning rate | `2e-5` |
| Training epochs | 2 |
| Training batch size | 8 |
| Evaluation batch size | 16 |
| Maximum sequence length | 256 |
| Weight decay | 0.01 |
| Random seed | 42 |
| Loss | Cross-entropy classification loss |

The validation set is used to select the best checkpoint based on F1-score. The held-out test set is reserved for final evaluation.

### 5. Model Evaluation

The model is evaluated using accuracy, precision, recall, F1-score, a classification report and a confusion matrix.

### 6. Custom Predictions

A prediction function accepts a new movie review, tokenizes it, runs inference through the trained model and returns the predicted sentiment along with class probabilities.

### 7. Saving and Reloading

The trained model and tokenizer are saved so they can be reloaded for later inference without repeating training.

## Results

The following are the recorded results for the 2,000-review held-out test subset.

### Overall Test Performance

| Metric | Score |
|---|---:|
| **Accuracy** | **90.50%** |
| Precision | 90.66% |
| Recall | 90.30% |
| F1-score | 90.48% |
| Test loss | 0.4917 |

The model correctly classified **1,810 out of 2,000 test reviews**.

### Class-Wise Performance

| Sentiment | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Negative | 90.34% | 90.70% | 90.52% | 1,000 |
| Positive | 90.66% | 90.30% | 90.48% | 1,000 |

### Validation Performance

| Metric | Score |
|---|---:|
| Accuracy | 91.50% |
| Precision | 90.91% |
| Recall | 92.27% |
| F1-score | 91.58% |

The validation metrics were used during model selection. The test metrics represent the final evaluation on the held-out test set.

## How to Run the Project

### Option 1: Google Colab (Recommended)

1. Download or clone this repository.
2. Open `BERT_Movie_Review_Sentiment_Analysis.ipynb` in Google Colab.
3. Select **Runtime → Change runtime type → T4 GPU**, if available.
4. Run the notebook cells sequentially.
5. Allow the required libraries, IMDb dataset and pretrained BERT model to download.
6. Review the training logs, final metrics and confusion matrix.
7. Test the model using your own movie reviews.

Internet access is required for downloading the dataset and pretrained model.

### Option 2: Run Locally

Clone the repository:

```bash
git clone https://github.com/vishakh2513110-max/bert-movie-review-sentiment-analysis.git
cd bert-movie-review-sentiment-analysis
```

Install the dependencies:

```bash
pip install torch transformers datasets accelerate scikit-learn matplotlib pandas
```

Open the notebook in Jupyter or upload it to Google Colab, then run the cells in order. A compatible GPU is recommended for faster fine-tuning.

## Project Structure

```text
bert-movie-review-sentiment-analysis/
├── BERT_Movie_Review_Sentiment_Analysis.ipynb
└── README.md
```

The trained model is saved separately by the notebook and is not required to be committed to the repository.

## Limitations

- The experiment uses 10,000 selected reviews rather than the full 50,000-review dataset.
- Reviews are truncated to 256 tokens, so information near the end of longer reviews may be lost.
- The model performs binary sentiment classification and does not predict emotions or sentiment intensity.
- Performance on other domains, such as product reviews or social media posts, has not been evaluated.
- The project currently focuses on model training and inference; a Streamlit interface and deployment can be added as future improvements.

## Future Improvements

- Develop an interactive Streamlit interface.
- Compare BERT against a TF-IDF and Logistic Regression baseline.
- Analyse misclassified reviews and common error patterns.
- Experiment with longer sequence lengths and additional training data.
- Evaluate the model on reviews from other domains.

## Key Learnings

- Understanding BERT tokenization, attention masks and contextual embeddings.
- Fine-tuning a pretrained Transformer for downstream classification.
- Building a proper training, validation and testing workflow.
- Evaluating classification models using multiple metrics.
- Saving and reloading a trained NLP model for inference.

## Author

**Vishakh**

GitHub: [@vishakh2513110-max](https://github.com/vishakh2513110-max)

---

*This project was developed for learning and demonstrating Transformer-based NLP classification. Reported metrics correspond to the selected dataset subset and experiment configuration described above.*
