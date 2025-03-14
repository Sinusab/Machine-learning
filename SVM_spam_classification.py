# Import necessary libraries
import re
import numpy as np
from nltk.stem import PorterStemmer
from scipy.io import loadmat
import pandas as pd
from sklearn.svm import SVC

# ----------------------------------------------
# Part 1: Vocabulary Loading
# ----------------------------------------------
# This section loads the vocabulary list and creates mappings.

def load_vocabulary(vocab_text):
    """Load vocabulary from text and create word-to-index and index-to-word mappings."""
    # Split the vocab text into lines
    vocab_list = vocab_text.split('\n')
    
    # Create word-to-index dictionary
    vocab = {}
    for item in vocab_list:
        if item:  # Skip empty lines
            value, key = item.split()
            vocab[key] = int(value)  # Word -> index
    
    # Create index-to-word dictionary
    idx_to_vocab = {}
    for item in vocab_list:
        if item:  # Skip empty lines
            key, value = item.split()
            idx_to_vocab[int(key)] = value  # Index -> word
    
    return vocab, idx_to_vocab, vocab_list

# ----------------------------------------------
# Part 2: Email Preprocessing
# ----------------------------------------------
# This section preprocesses email content.

def preProcessEmail(email_content):
    """Preprocess email content to normalize text."""
    # Convert to lowercase
    file_content = email_content.lower()
    
    # Replace numbers with 'number'
    file_content = re.sub("[\d]+", "number", file_content)
    
    # Replace URLs with 'httpaddr'
    file_content = re.sub("https?://[^\s]+", "httpaddr", file_content)
    
    # Replace email addresses with 'emailaddr'
    file_content = re.sub("[^\s]+@[^\s]+", "emailaddr", file_content)
    
    # Remove special characters
    file_content = re.sub("[^A-Za-z0-9\s]+", "", file_content)
    
    # Handle newlines and extra spaces
    file_content = re.sub("\n{2,2}", "", file_content)
    file_content = re.sub("\s\n", " ", file_content)
    file_content = re.sub("\n", " ", file_content)
    file_content = re.sub("^\s+", "", file_content)
    
    return file_content

# ----------------------------------------------
# Part 3: Word Stemming
# ----------------------------------------------
# This section applies stemming to the preprocessed text.

def word_stemming(file_content):
    """Stem words in the preprocessed email content."""
    edited_content = preProcessEmail(file_content)
    stemmer = PorterStemmer()
    edited_content = [stemmer.stem(token) for token in edited_content.split(' ')]
    edited_content = " ".join(edited_content)
    return edited_content

# ----------------------------------------------
# Part 4: Feature Extraction
# ----------------------------------------------
# This section converts text to feature vectors.

def check_in_vocab(clean_text, vocab):
    """Find indices of words present in the vocabulary."""
    word_index = []
    for char in clean_text.split(" "):
        if len(char) > 0 and char in vocab:
            word_index.append(vocab[char])
    return word_index

def extract_features(word_index, n_features=1899):
    """Create a binary feature vector from word indices with fixed size."""
    features = np.zeros((n_features, 1))  # Match spam dataset feature size (1899)
    for i in word_index:
        if i <= n_features:  # Ensure index is within bounds
            features[i - 1] = 1  # Adjust for 1-based indexing to 0-based
    return features

# ----------------------------------------------
# Part 5: Spam Classification with SVM
# ----------------------------------------------
# This section trains and evaluates an SVM classifier.

def train_spam_classifier():
    """Train and evaluate a linear SVM for spam classification."""
    # Load training and test data
    spam_data = loadmat('spamTrain.mat')
    X_train = spam_data["X"]
    Y_train = spam_data["y"]

    spam_data_test = loadmat('spamTest.mat')
    X_test = spam_data_test["Xtest"]
    Y_test = spam_data_test["ytest"]
    
    print(f"Training data shape: {X_train.shape}")
    
    # Train SVM with fixed C
    C = 0.1
    spam_svm = SVC(C=C, kernel="linear")
    spam_svm.fit(X_train, Y_train[:, 0])
    
    # Calculate accuracies
    train_accuracy = spam_svm.score(X_train, Y_train[:, 0])
    test_accuracy = spam_svm.score(X_test, Y_test[:, 0])
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    return spam_svm, X_train, Y_train, X_test, Y_test

# ----------------------------------------------
# Part 6: Feature Weight Analysis
# ----------------------------------------------
# This section analyzes the most influential features.

def analyze_feature_weights(spam_svm, idx_to_vocab):
    """Identify and display top words by SVM weights."""
    weights = spam_svm.coef_[0]
    print(f"Weights shape: {weights.shape}")
    
    # Stack indices and weights
    data = np.hstack((np.arange(1, 1900).reshape(1899, 1), weights.reshape(1899, 1)))
    dataframe = pd.DataFrame(data, columns=['Index', 'Weight'])
    
    # Sort by weight
    dataframe.sort_values('Weight', ascending=False, inplace=True)
    
    # Print top 10 words
    print("\nTop 10 influential words:")
    indices = dataframe['Index'].values[:10].astype(int)
    for i in range(10):
        idx = indices[i]
        word = idx_to_vocab[idx]
        weight = dataframe['Weight'].values[i]
        print(f"{i+1}. {word}: {weight:.4f}")

# ----------------------------------------------
# Main Execution
# ----------------------------------------------
if __name__ == "__main__":
    # Load actual vocabulary (replace with your vocab.txt or equivalent)
    # Example: Assuming vocab.txt is in the same directory
    with open('vocab.txt', 'r') as f:
        vocab_text = f.read()
    # For testing, you can still use the small example if vocab.txt isn’t available
    # vocab_text = "1 anyone\n2 knows\n3 how\n4 much"

    file_content = "Anyone knows how much it costs?\nhttp://example.com\nemail@domain.com\n12345"

    # Load vocabulary
    print("# Loading Vocabulary")
    vocab, idx_to_vocab, vocab_list = load_vocabulary(vocab_text)
    print(f"Vocabulary list sample: {vocab_list[:5]}")

    # Process email
    print("\n# Preprocessing Email")
    pre_content = preProcessEmail(file_content)
    print(f"Preprocessed content: {pre_content}")
    clean_text = word_stemming(pre_content)
    print(f"Stemmed text: {clean_text}")

    # Extract features
    print("\n# Extracting Features")
    word_index = check_in_vocab(clean_text, vocab)
    features = extract_features(word_index)
    print(f"Word indices: {word_index}")
    print(f"Features shape: {features.shape}")
    print(f"Features (first 10): {features[:10].flatten()}")

    # Train SVM
    print("\n# Training Spam Classifier")
    spam_svm, X_train, Y_train, X_test, Y_test = train_spam_classifier()

    # Analyze weights
    print("\n# Analyzing Feature Weights")
    analyze_feature_weights(spam_svm, idx_to_vocab)