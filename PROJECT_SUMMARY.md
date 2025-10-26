# Machine Translation Project: Attention Is All You Need

## Project Overview

This project implements neural machine translation models based on the paper "Attention Is All You Need" (Vaswani et al., 2017). The project evaluates three approaches to machine translation:

1. **RNN-based model** with attention mechanism
2. **Transformer model** implementing the architecture from the paper
3. **Off-the-shelf APIs** including OpenAI GPT, Google Cloud Translate, and Azure Translator

## Dataset

- **WMT English-French Dataset** (WMT14)
- Standard benchmark for neural machine translation
- Publicly available through Hugging Face datasets

## Implementation Details

### 1. RNN Model (`src/models/rnn_model.py`)
- Bidirectional LSTM encoder
- Attention mechanism for decoder
- Seq2Seq architecture
- Teacher forcing during training

### 2. Transformer Model (`src/models/transformer_model.py`)
- Multi-head self-attention
- Positional encoding
- Feed-forward networks
- Layer normalization and residual connections
- Exactly as described in "Attention Is All You Need"

### 3. Data Processing (`src/data/`)
- Custom tokenizer implementation
- WMT dataset loading and preprocessing
- Data loaders for training

### 4. Training Scripts (`src/training/`)
- Separate training scripts for RNN and Transformer
- Validation loop
- Model checkpointing
- Progress tracking

### 5. Evaluation (`src/evaluation/`)
- BLEU score calculation using sacrebleu
- Model evaluation on test set
- API evaluation
- Results visualization

## Key Features

1. Complete implementation of Transformer architecture
2. RNN baseline for comparison
3. API integration for off-the-shelf models
4. BLEU score evaluation
5. Visualization of results
6. Comprehensive logging
7. Modular code structure

## Usage

See `QUICKSTART.md` for detailed instructions on:
1. Installation
2. Data preparation
3. Training models
4. Evaluation
5. Results comparison

## Expected Outcomes

After running the complete pipeline, you will have:
- Trained RNN and Transformer models
- BLEU scores for all approaches
- Comparison visualization
- Saved model checkpoints for future use

## Dependencies

Key libraries:
- PyTorch for model implementation
- Transformers for data loading
- sacrebleu for BLEU calculation
- OpenAI/Google Cloud/Azure APIs for comparison

## File Structure

```
AIAYN/
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # Model implementations (RNN, Transformer)
│   ├── training/       # Training scripts
│   ├── evaluation/     # Evaluation and BLEU calculation
│   └── utils/          # Utilities (tokenizer, metrics, logger)
├── data/               # Processed datasets
├── models/             # Saved model checkpoints
├── results/            # Evaluation results
└── logs/               # Training logs
```

## References

- Vaswani, A., et al. (2017). "Attention Is All You Need". NIPS.
- WMT 2014 English-French dataset
- BLEU metric: Papineni et al. (2002)

## Notes

- All models can be trained on CPU or GPU
- API evaluation is optional and requires API keys
- The implementation follows best practices for PyTorch
