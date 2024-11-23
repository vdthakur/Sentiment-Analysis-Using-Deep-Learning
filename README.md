# Sentiment Analysis Using Deep Learning

This project involves building a classifier to analyze the sentiment of text reviews using deep learning techniques. The task includes pre-processing text data, applying tokenization and word embeddings, and training various neural network models to classify reviews as positive or negative.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Data Preprocessing](#data-preprocessing)
   - [Sentiment Encoding](#sentiment-encoding)
   - [Data Cleaning](#data-cleaning)
   - [Train-Test Split](#train-test-split)
   - [Dataset Statistics](#dataset-statistics)
   - [Tokenization and Padding](#tokenization-and-padding)
3. [Word Embeddings](#word-embeddings)
4. [Model Architectures](#model-architectures)
   - [Multi-Layer Perceptron](#multi-layer-perceptron)
   - [Convolutional Neural Network](#convolutional-neural-network)
   - [Long Short-Term Memory Network](#long-short-term-memory-network)
5. [Results](#results)
6. [How to Run](#how-to-run)
7. [Dependencies](#dependencies)

---

## Introduction

This project aims to classify text reviews into positive or negative sentiments using various deep learning models, including:
- Multi-Layer Perceptron (MLP)
- Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)

The text data is divided into two folders for positive and negative reviews. The models are evaluated on train and test splits with performance metrics reported.

---

## Data Preprocessing

### Sentiment Encoding
- Positive reviews are labeled as `y = 1`.
- Negative reviews are labeled as `y = -1`.

### Data Cleaning
- Remove punctuation and numerical characters from the text data.

### Train-Test Split
- Use reviews with filenames `cv000` to `cv699` for training.
- Use reviews with filenames `cv700` to `cv999` for testing.

### Dataset Statistics
1. **Unique Words**: Count the total number of unique words in the dataset (train + test).
2. **Review Length**:
   - Calculate the average and standard deviation of review lengths.
   - Plot a histogram of review lengths.

### Tokenization and Padding
- Tokenize text using word frequency ranking (most frequent word = `1`, second most frequent = `2`, etc.).
- Select a review length `L` such that 70% (or optionally 90%) of reviews have lengths below `L`.
- Truncate reviews longer than `L` and zero-pad reviews shorter than `L`.

---

## Word Embeddings
- Use a Keras `Embedding` layer to generate word embeddings.
- Limit vocabulary size to the top 5,000 words (optionally, use the full vocabulary).
- Set the embedding vector dimension to `32`.
- Represent each document as a `32 × L` matrix, then flatten it into a vector.

---

## Model Architectures

### Multi-Layer Perceptron
1. Architecture:
   - Three dense hidden layers, each with 50 ReLU neurons.
   - Dropout: 20% for the first layer, 50% for the others.
   - Output layer: 1 sigmoid neuron.
2. Training:
   - Optimizer: Adam
   - Loss function: Binary Cross-Entropy
   - Batch size: 10
   - Epochs: 2
3. Output:
   - Report training and testing accuracies.

### Convolutional Neural Network
1. Architecture:
   - Add a `Conv1D` layer after the embedding layer with:
     - 32 feature maps
     - Kernel size: 3
   - Add a `MaxPooling1D` layer with:
     - Pool size: 2
     - Stride: 2
   - Use the same dense layers as in the MLP model.
2. Training:
   - Optimizer: Adam
   - Loss function: Binary Cross-Entropy
   - Batch size: 10
   - Epochs: 2
3. Output:
   - Report training and testing accuracies.

### Long Short-Term Memory Network
1. Architecture:
   - Add an LSTM layer after the embedding layer with 32-dimensional input vectors.
   - Follow with a dense layer of 256 ReLU neurons.
   - Dropout: 20% for both LSTM and dense layers.
2. Training:
   - Optimizer: Adam
   - Loss function: Binary Cross-Entropy
   - Batch size: 10
   - Epochs: 10-50
3. Output:
   - Report training and testing accuracies.

---

## Results
- Report and compare the train and test accuracies for the MLP, CNN, and LSTM models.
