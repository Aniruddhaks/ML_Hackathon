# Hangman HMM Implementation (Part A)

## Overview
This implementation uses Hidden Markov Models (HMM) to predict letters in a Hangman game. The model learns patterns from a corpus of words to make intelligent guesses about unknown letters.

## Key Components

### 1. Data Preprocessing
- Load words from corpus.txt
- Convert to lowercase
- Filter non-alphabetic words
```python
def preprocess_words(words):
    processed_words = []
    for word in words:
        word = word.lower()
        if word.isalpha():
            processed_words.append(word)
    return processed_words
```

### 2. HMM Implementation
Using `hmmlearn` library's `MultinomialHMM`

Key Concepts:
- Hidden States: 26 states (one for each letter)
- Observations: Letter sequences in words
- Parameters:
  - Initial probabilities (π)
  - Transition probabilities (A)
  - Emission probabilities (B)

Main HMM Class:
```python
class HangmanHMM:
    def __init__(self):
        self.models = {}  # Separate model for each word length
        self.alphabet = string.ascii_lowercase
        self.letter_to_idx = {letter: idx for idx, letter in enumerate(self.alphabet)}
```

### 3. Training Process
1. Group words by length
2. For each length:
   - Convert words to sequences of indices
   - Initialize HMM with 26 states
   - Train using Baum-Welch algorithm
```python
model = hmm.MultinomialHMM(n_components=26, n_iter=100)
model.fit(X, lengths=lengths)
```

### 4. Prediction
The `HangmanPredictor` class handles letter predictions:
- Input: Masked word (e.g., "h_ll_") and guessed letters
- Output: Next best letter to guess and its probability

Key Methods:
```python
def get_letter_probabilities(word_length, known_positions)
def predict_next_letter(masked_word, guessed_letters)
def predict_letter_probs(masked_word, guessed_letters)
```

### 5. Model Validation
- Test on separate test.txt file
- Metrics:
  - Prediction accuracy by word length
  - Probability distribution of correct/incorrect predictions

## Tools Used
- numpy: Numerical computations
- hmmlearn: HMM implementation
- matplotlib/seaborn: Visualization
- pickle: Model saving

## Flow of Operation
1. Load and preprocess corpus
2. Train HMM models for each word length
3. Make predictions using:
   - Emission probabilities for unknown positions
   - Known letters in the word
   - Previously guessed letters
4. Save model for RL part

## Key Features
- Separate models for different word lengths
- Handles partially revealed words
- Provides probability distribution for all possible letters
- Excludes already guessed letters
- Normalizes probabilities for better predictions

## Model Performance
- Better accuracy on common word lengths
- Higher prediction probabilities correlate with correct guesses
- Takes into account letter patterns in English words

The HMM serves as the "brain" of the Hangman solver, providing intelligent letter predictions based on learned patterns from the training corpus.