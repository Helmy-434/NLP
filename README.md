# Text Toxicity Classification 

A robust end-to-end deep learning pipeline built in PyTorch to classify toxic content from multi-column text inputs (user queries and image descriptions). The architecture utilizes a Bidirectional LSTM (BiLSTM) with dual global pooling (Average + Max Pooling) and class-weighted optimization to handle severely imbalanced categories.

---

## Key Features

- **Multi-Field Text Processing:** Merges dual text inputs (`query` and `image descriptions`) using a `[SEP]` delimiter token to preserve field boundaries.
- **BiLSTM with Dual Pooling:** Extracts global context across time steps by concatenating Global Average Pooling and Global Max Pooling vectors ($512$ total features).
- **Class-Weighted Cross-Entropy:** Computes inverse class weights dynamically to handle severe class imbalance across toxicity categories.
- **Training Stability:** Implements L2 norm gradient clipping (`grad_clip=5.0`) and early stopping based on validation Macro-F1 to prevent exploding gradients and overfitting.
- **Automated PDF Report Generation:** Programmatically compiles loss/F1 curves, confusion matrices, and dataset distributions into a single exportable `toxic_model_report.pdf` using `matplotlib.backends.backend_pdf`.

---

## Model Architecture 


                [ Input Text Sequence (Query + [SEP] + Description) ]
                                      │
                                      ▼
                     [ NLTK Tokenization & Vocab Lookup ]
                                      │
                                      ▼
                        [ Embedding Layer (100d) ]
                                      │
                                      ▼
                [ Bidirectional LSTM Layer (Hidden Dim: 128) ]
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
     [ Global Average Pooling (256d) ]              [ Global Max Pooling (256d) ]
              └───────────────────────┬───────────────────────┘
                                      │
                                      ▼
                        [ Concatenate Pools (512d) ]
                                      │
                                      ▼
                       [ Dropout (p=0.3) & Dense (64) ]
                                      │
                                      ▼
                               [ ReLU Activation ]
                                      │
                                      ▼
                        [ Final Linear Classifier ]
                                      │
                                      ▼
                     [ Class Probability Distribution ]



 ## Results & Performance

The pipeline was evaluated across both a Vanilla RNN baseline and the enhanced BiLSTM model on a stratified 70/15/15 train/val/test split:

| Metric | Vanilla RNN | BiLSTM + Dual Pooling (Final) |
| :--- | :---: | :---: |
| **Accuracy** | ~0.931 | **~0.942** |
| **Macro F1** | ~0.928 | **~0.940** |
| **Weighted F1** | ~0.930 | **~0.941** |

### Key Observations
1. **Global Feature Pooling:** Concatenating Average and Max Pooling across all sequence steps provided a noticeable improvement over extracting only boundary hidden states (`hidden[-1]`), as toxic trigger words can appear anywhere in the sequence.
2. **Gradient Stability:** Applying L2 norm gradient clipping (`max_norm=5.0`) eliminated loss spikes during backpropagation through time (BPTT).
3. **Class Weighting:** Balancing the loss function prevented the model from collapsing into majority non-toxic predictions on imbalanced test splits.

---

      
