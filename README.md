# Hangman Solver: ML Implementation

## Project Overview

This project implements an intelligent Hangman letter prediction system using two complementary approaches:
- **Part A:** Hidden Markov Model (HMM) for letter sequence prediction
- **Part B:** Reinforcement Learning (RL) agent with frequency-based prediction and hyperparameter optimization

The system learns patterns from a corpus of English words and predicts the next best letter to guess in Hangman, considering letter frequency, positional patterns, bigram relationships, and previously guessed letters.

---

## Part A: Hidden Markov Model Implementation

### Notebook
`part-a.ipynb`

### Overview
Part A implements a probabilistic HMM-based approach that trains separate models for each word length to capture length-specific letter patterns and transitions.

### Architecture

#### 1. Data Preprocessing
- Loads training corpus and test data from `Data/corpus.txt` and `Data/test.txt`
- Converts all words to lowercase
- Filters non-alphabetic characters
- Organizes words by length for stratified model training

#### 2. HMM Model Class (HangmanHMM)

**Key Components:**
- Separate MultinomialHMM for each word length (3-10 states per model)
- Letter-to-index mapping for observation sequences
- Baum-Welch training algorithm with 100 iterations
- Laplace smoothing to prevent zero probabilities

**Training Process:**
- Groups corpus words by length
- For each length: trains individual HMM with letter frequency initialization
- Number of hidden states: `min(max(3, word_length // 2), 10)`
- Emission probabilities initialized from letter frequency distributions

#### 3. Prediction Engine (HangmanPredictor)

**Algorithm:**
1. Extract emission probabilities from trained HMM for word length
2. Average probabilities across unknown positions (marked as '_')
3. Exclude already-guessed letters
4. Normalize and return highest probability letter

### Validation Metrics
- Overall success rate across all test words
- Success rate stratified by word length
- Prediction confidence analysis (correct vs incorrect)
- Model coverage statistics

### Output Files
- `hmm_hangman_model.pkl`: Serialized trained models
- `hmm_validation_results.json`: Performance metrics and statistics

---

## Part B: Reinforcement Learning Agent

### Notebook
`part-b.ipynb`

### Overview
Part B implements a frequency-based RL agent with three progressive optimization levels:
1. Baseline (no optimization)
2. Basic grid search optimization
3. Advanced optimization with bigram analysis

### Agent Architectures

#### 1. Baseline Agent (FrequencyAgent)

**Features:**
- Positional frequency analysis: tracks letter occurrence at each position for each word length
- Global frequency analysis: overall letter frequency across corpus
- Weighted combination: 60% positional + 40% global

**Prediction Process:**
1. Calculate positional frequencies for unknown positions
2. Combine with global frequencies using fixed weights
3. Exclude guessed and known letters
4. Return letter with maximum probability

#### 2. Optimized Agent (OptimizedFrequencyAgent)

**Enhancements:**
- Tunable hyperparameters: `pos_weight`, `global_weight`, `smoothing_alpha`
- Laplace smoothing with adjustable alpha parameter
- Grid search over hyperparameter space (36 configurations)

**Hyperparameter Grid:**
- `pos_weight`: [0.5, 0.6, 0.7]
- `global_weight`: [0.3, 0.4, 0.5]
- `smoothing_alpha`: [0.5, 1.0, 1.5, 2.0]

**Optimization Process:**
- Tests all combinations on 16% test sample
- Evaluates success rate and final score for each configuration
- Selects best hyperparameters based on scoring function

#### 3. Advanced Agent (AdvancedFrequencyAgent)

**Key Innovations:**
- Bigram frequency analysis: letter pair patterns (e.g., 'qu', 'th', 'ing')
- Context-aware predictions: uses adjacent known letters
- Forward and backward bigram probabilities
- Expanded hyperparameter grid (320+ configurations)

**Bigram Implementation:**
- `bigram_after`: P(next_letter | current_letter)
- `bigram_before`: P(prev_letter | current_letter)
- Combines with positional and global frequencies

**Enhanced Hyperparameter Grid:**
- `pos_weight`: [0.45, 0.50, 0.55, 0.60, 0.65]
- `global_weight`: [0.20, 0.25, 0.30, 0.35]
- `bigram_weight`: [0.10, 0.15, 0.20, 0.25]
- `smoothing_alpha`: [0.5, 1.0, 1.5, 2.0]

**Weight Constraint:** pos_weight + global_weight + bigram_weight = 1.0

### Hangman Game Environment

**Class:** HangmanGame
- Tracks masked word, guessed letters, wrong guesses, repeated guesses
- Maximum wrong guesses: 6 (standard Hangman rules)
- Win condition: all letters revealed
- Loss condition: 6 wrong guesses

### Evaluation Metrics

**Performance Indicators:**
- Success rate: percentage of games won
- Total wrong guesses: incorrect letter predictions
- Total repeated guesses: duplicate letter attempts
- Final score: `(success_rate/100 * games) - (wrong * 5) - (repeated * 2)`

**Test Configuration:**
- 16% of test dataset for all evaluations (fair comparison)
- Random seed: 42 (reproducibility)
- Same test sample used across all three approaches

### Output Files
- `rl_baseline_results.json`: Baseline agent performance
- `rl_optimized_results.json`: Grid search results with best hyperparameters
- `rl_advanced_results.json`: Advanced optimization results with all configurations
- `rl_comparison.png`: Visual comparison (baseline vs optimized)
- `rl_comprehensive_comparison.png`: Three-way comparison visualization

---

## Technical Details

### Scoring Function
```
Final Score = (Success Rate / 100 * Number of Games) 
              - (Total Wrong Guesses * 5) 
              - (Total Repeated Guesses * 2)
```

This penalizes wrong and repeated guesses heavily to encourage efficient prediction.

### Why Bigrams Improve Performance
- Captures letter pair dependencies (e.g., 'u' almost always follows 'q')
- Context-aware: uses known adjacent letters to predict missing ones
- Handles common patterns: '_ing', 'th_', 'sh_', etc.
- Reduces wrong guesses by leveraging sequential letter relationships

### Smoothing Strategy
Laplace (add-alpha) smoothing prevents zero probabilities:
```python
count[letter] += smoothing_alpha
probability = count[letter] / sum(all_counts)
```

Benefits:
- Handles unseen letter combinations
- Adjustable alpha controls smoothing strength
- Better generalization to test data

---

## Dependencies

```bash
pip install numpy matplotlib seaborn hmmlearn
```

**Required Packages:**
- numpy: Numerical computations and array operations
- matplotlib: Visualization and plotting
- seaborn: Enhanced statistical visualizations (Part A)
- hmmlearn: Hidden Markov Model training and inference (Part A)
- pickle: Model serialization
- json: Results storage
- collections.Counter: Letter frequency counting
- string: Alphabet utilities

---

## Usage

### Running Part A (HMM)
```bash
# Execute all cells in part-a.ipynb
# Output: hmm_hangman_model.pkl, hmm_validation_results.json
```

### Running Part B (RL)
```bash
# Execute all cells in part-b.ipynb sequentially
# Baseline evaluation (cell 10)
# Basic optimization (cells 12-13)
# Advanced optimization (cells 18-19)
# Comparison visualization (cell 20)
```

### Loading Trained Models
```python
import pickle

# Load HMM model
with open('hmm_hangman_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    predictor = model_data['predictor']

# Make prediction
letter, prob = predictor.predict_next_letter("h_ll_", {'h', 'l'})
```

---

## File Structure

```
ML_Hackathon/
├── Data/
│   ├── corpus.txt              # Training corpus
│   └── test.txt                # Test dataset
├── part-a.ipynb                # HMM implementation
├── part-b.ipynb                # RL agent with optimization
├── hmm_hangman_model.pkl       # Trained HMM models
└── README.md                   # This file
```

---

## Key Features

**Part A (HMM):**
- Separate models per word length for optimized accuracy
- Probabilistic framework with emission and transition matrices
- Baum-Welch training algorithm
- Confidence-based prediction analysis

**Part B (RL):**
- Progressive optimization: baseline, basic, advanced
- Systematic hyperparameter grid search
- Bigram analysis for context-aware predictions
- Multi-source probability combination
- Fair comparison across approaches (same test sample)

---

## Performance Insights

### Why This Approach Works

**HMM Advantages:**
- Captures sequential letter patterns
- Models hidden states representing word structure
- Provides probability distributions over all letters

**Frequency-Based Advantages:**
- Computationally efficient
- Directly uses observed letter statistics
- Positional awareness improves accuracy

**Bigram Advantages:**
- Context-dependent predictions
- Handles letter pair dependencies
- Reduces wrong guesses on patterned words

**Combined Strengths:**
- HMM for sequential modeling
- Frequency for positional patterns
- Bigrams for local context
- Grid search for optimal weight tuning

---

## Reproducibility

All implementations use fixed random seed (42) for reproducibility:
- Model training produces identical results
- Test sample selection is deterministic
- Grid search explores configurations in fixed order
- Results can be verified across runs

---

## Troubleshooting

**Low Success Rate:**
- Verify corpus quality and size
- Check that test data is representative
- Ensure preprocessing removes invalid words

**Model Training Failures:**
- Requires minimum 2 words per length for HMM training
- Check for sufficient training data in corpus
- Verify hmmlearn installation

**Memory Issues:**
- Grid search tests many configurations (320+ for advanced)
- Consider reducing hyperparameter grid size
- Process smaller test sample if needed

---

## Results Summary

Expected performance ranges (actual results may vary with corpus):

**Part A (HMM):**
- Overall success rate: 50-65%
- Best performance: word lengths 5-7
- Confidence gap: correct predictions show higher confidence

**Part B (RL):**
- Baseline: 40-55% success rate
- Basic optimization: 45-60% success rate
- Advanced optimization: 50-65% success rate
- Improvement: 5-15% gain from baseline to advanced

---

## Author Information

**Project:** ML Hackathon - Hangman Solver
**Repository:** ML_Hackathon (Aniruddha k s,Akanksh Rai , Akarsh T ,Anirudh Anand krishnan)
**Branch:** part-A_akanksh

---

## References

**HMM Theory:**
- Rabiner, L. R. (1989). "A tutorial on hidden Markov models and selected applications"
- hmmlearn documentation: https://hmmlearn.readthedocs.io

**Frequency Analysis:**
- English letter frequency distributions
- Positional letter statistics in word corpora

**Last Updated:** November 2025
