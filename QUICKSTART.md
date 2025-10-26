# Quick Start Guide

This guide will walk you through setting up and running the machine translation project.

## 🚀 Quick Option: Jupyter Notebook

For the easiest experience, use the comprehensive Jupyter notebook:
```bash
jupyter notebook Machine_Translation_Pipeline.ipynb
```

This notebook includes everything in one place:
- Data preparation
- Model training
- Evaluation and visualization
- Results comparison

## 📋 Alternative: Command Line

For command-line usage, follow the steps below.

## 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

## 2. Prepare Data

Download and preprocess the WMT English-French dataset:

```bash
python src/data/prepare_data.py --output_dir data --vocab_size 30000
```

This will download the dataset and create vocabulary files.

## 3. Train Models

### Train RNN Model

```bash
python src/training/train_rnn.py \
    --data_dir data \
    --model_dir models \
    --batch_size 32 \
    --num_epochs 10 \
    --learning_rate 0.001
```

### Train Transformer Model

```bash
python src/training/train_transformer.py \
    --data_dir data \
    --model_dir models \
    --batch_size 32 \
    --num_epochs 10 \
    --learning_rate 0.0001 \
    --d_model 512 \
    --num_heads 8
```

## 4. Evaluate Models

Evaluate trained models on the test set:

```bash
# Evaluate both models
python src/evaluation/evaluate.py \
    --model all \
    --checkpoint models/rnn_best.pt \
    --data_dir data \
    --results_dir results
```

## 5. Evaluate APIs (Optional)

To evaluate off-the-shelf translation APIs:

1. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_key_here
GOOGLE_CLOUD_API_KEY=your_key_here
AZURE_API_KEY=your_key_here
AZURE_ENDPOINT=your_endpoint_here
```

2. Run evaluation:
```bash
python src/evaluation/evaluate_apis.py \
    --api all \
    --sample_size 100 \
    --data_dir data \
    --results_dir results
```

## 6. Compare Results

Generate a comparison of all BLEU scores:

```bash
python src/evaluation/calculate_bleu.py \
    --results_dir results \
    --visualize
```

This will create a CSV file and a visualization comparing all models.

## Project Structure

```
AIAYN/
├── data/                  # Preprocessed datasets
├── models/                # Trained model checkpoints
├── results/               # Evaluation results
├── src/
│   ├── data/             # Data loading and preprocessing
│   ├── models/           # Model implementations
│   ├── training/         # Training scripts
│   ├── evaluation/       # Evaluation scripts
│   └── utils/            # Utility functions
└── logs/                 # Training logs
```

## Troubleshooting

- **Out of memory**: Reduce batch size or max_len in training scripts
- **Dataset download issues**: The WMT dataset download may take time
- **API errors**: Check that your API keys are correct in the `.env` file

## Expected Results

After training and evaluation, you should see:
- RNN baseline BLEU score
- Transformer model BLEU score
- API translation BLEU scores (if API keys provided)
- Visualization comparing all methods
