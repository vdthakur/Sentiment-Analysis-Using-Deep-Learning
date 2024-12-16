# Sentiment Analysis Using Deep Learning

This project builds a sentiment classifier for text reviews using deep learning. The task includes data preprocessing, tokenization, word embeddings, and training neural networks to classify reviews as positive or negative.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Data Preprocessing](#data-preprocessing)
3. [Word Embeddings](#word-embeddings)
4. [Model Architectures](#model-architectures)
5. [Results](#results)
---

## Introduction

The goal is to classify reviews into positive or negative sentiments using:
- **Multi-Layer Perceptron (MLP)**
- **Convolutional Neural Network (CNN)**
- **Long Short-Term Memory (LSTM)**

---

## Data Preprocessing

1. **Sentiment Encoding**: Labels are `1` for positive and `-1` for negative reviews.
2. **Data Cleaning**: Remove punctuation and numbers.
3. **Train-Test Split**: Use `cv000` to `cv699` for training and `cv700` to `cv999` for testing.
4. **Dataset Statistics**:
   - Count unique words.
   - Compute average and standard deviation of review lengths.
   - Plot a histogram of review lengths.
5. **Tokenization and Padding**:
   - Tokenize words by frequency rank.
   - Truncate or pad reviews to a chosen length.

---

## Word Embeddings

- Use a Keras `Embedding` layer with:
  - Vocabulary size: 5,000 words (or full vocabulary).
  - Embedding dimensions: 32.
- Represent each review as a `32 × L` matrix, flattened into a vector.

---

## Model Architectures

### Multi-Layer Perceptron (MLP)
- Three dense layers with 50 ReLU neurons and dropout (20% and 50%).
- Single sigmoid output neuron.
- Training: Adam optimizer, binary cross-entropy loss, batch size 10, and 2 epochs.

### Convolutional Neural Network (CNN)
- Add a `Conv1D` layer with:
  - 32 feature maps, kernel size 3.
- Follow with `MaxPooling1D` (pool size 2, stride 2).
- Use the same dense layers as the MLP.
- Training: Same settings as MLP.

### Long Short-Term Memory (LSTM)
- Add an LSTM layer with 32-dimensional input vectors.
- Dense layer with 256 ReLU neurons and dropout (20%).
- Training: Binary cross-entropy loss, batch size 10, and 10-50 epochs.

---

## Results

- Compare train and test accuracies for MLP, CNN, and LSTM models.

---
