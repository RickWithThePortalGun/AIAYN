# Machine Translation Project: Attention Is All You Need

This project implements machine translation models based on the paper "Attention Is All You Need" (Vaswani et al., 2017) and evaluates them on the WMT English-French dataset.

## Overview

This project compares different approaches to neural machine translation:
- **RNN-based model**: Traditional recurrent neural network with attention
- **Transformer model**: Implementation based on the paper "Attention Is All You Need"
- **Off-the-shelf APIs**: Evaluation of commercial translation APIs (OpenAI, Google Cloud Translate, Azure Translator)

## Dataset

- **WMT English-French Dataset**: Standard benchmark for machine translation
- Download and preprocess using the scripts provided

## Project Structure

```
AIAYN/
├── data/                  # Downloaded datasets
├── models/                # Saved model checkpoints
├── results/               # Evaluation results
├── src/
│   ├── data/             # Data loading and preprocessing
│   ├── models/           # Model implementations
│   ├── training/         # Training scripts
│   ├── evaluation/       # Evaluation scripts
│   └── utils/            # Utility functions
├── notebooks/            # Jupyter notebooks for exploration
├── configs/              # Configuration files
└── requirements.txt      # Python dependencies
```

## Installation

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package in editable mode (recommended):
```bash
pip install -e .
```

OR install dependencies directly:
```bash
pip install -r requirements.txt
```

4. Download NLTK data:
```python
python -c "import nltk; nltk.download('punkt')"
```

5. Set up API keys in `.env` file (optional, for API evaluation):
```
OPENAI_API_KEY=your_key_here
GOOGLE_CLOUD_API_KEY=your_key_here
AZURE_API_KEY=your_key_here
AZURE_ENDPOINT=your_endpoint_here
```

## Usage

### Data Preparation
```bash
python src/data/prepare_data.py --output_dir data --vocab_size 30000
```

### Training Models

Train RNN model:
```bash
python src/training/train_rnn.py --config configs/rnn_config.yaml
```

Train Transformer model:
```bash
python src/training/train_transformer.py --config configs/transformer_config.yaml
```

### Evaluation
```bash
python src/evaluation/evaluate.py --model rnn --checkpoint models/rnn_best.pt
python src/evaluation/evaluate.py --model transformer --checkpoint models/transformer_best.pt
python src/evaluation/evaluate_apis.py --sample_size 1000
```

### BLEU Score Calculation
```bash
python src/evaluation/calculate_bleu.py --results_dir results/
```

## Results

BLEU scores will be saved in `results/bleu_scores.csv` comparing:
- RNN baseline
- Transformer model
- OpenAI GPT
- Google Cloud Translate
- Azure Translator

## References

- Vaswani, A., et al. (2017). "Attention Is All You Need". NIPS.
- WMT dataset: http://www.statmt.org/wmt14/

## License

MIT License
