### Detection of Fake Reviews on Online Review Platforms using Deep Learning Architectures

[Source files (repository)](https://github.com/ashishsalunkhe/DeepSpamReview-Detection-of-Fake-Reviews-on-Online-Review-Platforms-using-DeepLearning-Architectures)

---

Dataset: `deceptive-opinion.csv`

## :monkey_face: Model 1: BiLSTM + GLoVe(50D)

**Preparation**:
1. Truthful = 1, Deceptive = 0
2. Tokenize all words and limit a review length to 150
3. Pad all reviews to length of 150
4. Divide reviews to 80% training and 20% testing sets
5. Embed words using GLoVe with 50 dimensions

**Neural Network**:
1. Embedding
2. Input
3. Bidirectional LSTM (_50_)
4. GlobalMaxPool
5. Dense (_50, relu_)
6. Dropout (_0.1_)
7. Dense (_2, sigmoid_)


## :dog: Model 2: BiLSTM + Attention + GLoVe(100D)

**Preparation**:
1. Truthful = 1, Deceptive = 0
2. Preprocess the text
3. Tokenize all words and set the max review size to 415 (avg + std * 3)
4. Pad all reviews to fit a 415 shape
5. Embed words using GLoVe with 100 dimensions

**Neural Network**:
1. Input
2. Embedding
3. Bidirectional LSTM (_60_)
4. Dropout (_0.3)
5. Attention
6. _Repeat 2-5_
7. Concatenate
8. Dense (_50, relu_)
9. Dropout (_0.2_)
10. BatchNormalization
11. Dense (_1, sigmoid_)

## :cat: Model 3: CNN + LSTM + Doc2Vec + TF-IDF

**Preparation**:
1. Positive = 1, Negative = 0
2. Truthful = 1, Deceptive = 0
3. Distinguish between the aforementioned combinations as:
    1. Positive and Truthful = `TRUE_POSITIVE`
    2. Positive and Deceptive = `FALSE_POSITIVE`
    3. Deceptive and Truthful = `TRUE_NEGATIVE`
    4. Deceptive and Negative = `FALSE_NEGATIVE`
4. Removing any words that don't occur at least 5 times
5. Preprocessing and stemming (reducing all words to their roots _playing -> play_) all reviews
6. Training a Doc2Vec model with the remaining words
7. Training and fitting TF-IDF vectors
8. Extracting training and testing datasets
    1. Split to 90% training and 10% testing
    2. Average the word vectors from both TF-IDF and Doc2Vec to be used during model fitting

**Neural Network**:
1. Sequential mode setup
2. Convolution (_filters=128, kernel_size=9, relu_)
3. Dropout (_0.25_)
4. MaxPooling (_2_)
5. Dropout (_0.25_)
6. Convolution (_filters=128, kernel_size=7, relu_)
7. Dropout (_0.25_)
8. MaxPooling (_2_)
9. Dropout (_0.25_)
10. Convolution (_filters=128, kernel_size=5, relu_)
11. Dropout (_0.25_)
12. Bidirectional LSTM (_50, recurrent_dropout=0.2_)
13. Dense (_4, softmax_) `Preperation (3)`
