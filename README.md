# Fine-Tuning Transformers for Text Classification

This repository contains a Jupyter Notebook that demonstrates how to fine-tune pretrained transformer models using the Hugging Face `transformers` library, focusing on the task of text classification. The main example revolves around sentiment analysis using the Yelp Polarity dataset.

This project was developed as an introductory lab exercise of the **Neural Networks and Deep Learning** class for understanding:
- Transfer learning in NLP
- Hands-on fine-tuning of large language models for specific downstream tasks
- Usage of Hugging Face tools in applied ML scenarios

## Overview

The notebook is structured in two main parts:

### 1. **Using Pretrained Pipelines**
Introduces the `transformers` pipeline API and demonstrates usage of pretrained models for common NLP classification tasks such as:
- Natural Language Inference (NLI) using `roberta-large-mnli`
- Acceptability classification using `distilbert-base-uncased-CoLA`

### 2. **Fine-Tuning a Pretrained Model**
Provides a step-by-step walkthrough for:
- Loading and preprocessing the [Yelp Polarity](https://huggingface.co/datasets/yelp_polarity) dataset
- Tokenizing the text inputs using a Hugging Face tokenizer
- Fine-tuning a pretrained model (`distilbert-base-uncased`) on a labeled dataset
- Evaluating the performance on test data

## Requirements

You will need:
- Python 3.8+
- `transformers`
- `datasets`
- `scikit-learn`
- `torch`
- `sentence-transformers`
- `tabulate`
- `matplotlib` (optional, for plotting)

You can install all dependencies with:

```bash
pip install transformers datasets scikit-learn torch matplotlib
```

## Run the Notebook

Launch the Jupyter notebook with:

```bash
jupyter notebook finetuning_transformers.ipynb
```

## Dataset Info

- **Yelp Polarity**: A dataset containing positive and negative reviews of restaurants.
- Each review is labeled as either **positive** or **negative**, and your model learns to classify new reviews accordingly.

## Results

The notebook evaluates model performance using standard metrics such as:
- Accuracy
- Precision, Recall, F1-score (via `classification_report`)


