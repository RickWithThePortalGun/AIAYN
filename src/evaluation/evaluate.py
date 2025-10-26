import argparse
import torch
import os
import sys
from tqdm import tqdm
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.loader import get_dataloaders
from src.models.rnn_model import EncoderRNN, DecoderRNN, Seq2SeqRNN
from src.models.transformer_model import Transformer
from src.utils.metrics import calculate_bleu
from src.utils.logger import setup_logger


def evaluate_model(model, dataloader, target_tokenizer, device, model_name):
    """Evaluate a model and return predictions and references."""
    model.eval()
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Evaluating {model_name}'):
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            target_texts = batch['target_text']
            
            # Translate
            for i in range(source.shape[0]):
                src_seq = source[i]
                decoded_indices = model.translate(src_seq)
                pred_text = target_tokenizer.decode(decoded_indices)
                
                predictions.append(pred_text)
                references.append(target_texts[i])
    
    return predictions, references


def evaluate_rnn(checkpoint_path, dataloader, device):
    """Load and evaluate RNN model."""
    logger = setup_logger('evaluate_rnn')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    source_tokenizer = checkpoint['source_tokenizer']
    target_tokenizer = checkpoint['target_tokenizer']
    
    # Create model
    encoder = EncoderRNN(
        vocab_size=len(source_tokenizer),
        embed_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.1
    )
    
    decoder = DecoderRNN(
        vocab_size=len(target_tokenizer),
        embed_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.1
    )
    
    model = Seq2SeqRNN(encoder, decoder, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    logger.info(f'Loaded RNN model from {checkpoint_path}')
    
    # Evaluate
    predictions, references = evaluate_model(model, dataloader, target_tokenizer, device, 'RNN')
    
    return predictions, references


def evaluate_transformer(checkpoint_path, dataloader, device):
    """Load and evaluate Transformer model."""
    logger = setup_logger('evaluate_transformer')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    source_tokenizer = checkpoint['source_tokenizer']
    target_tokenizer = checkpoint['target_tokenizer']
    config = checkpoint['model_config']
    
    # Create model
    model = Transformer(
        src_vocab_size=len(source_tokenizer),
        tgt_vocab_size=len(target_tokenizer),
        **config
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    logger.info(f'Loaded Transformer model from {checkpoint_path}')
    
    # Evaluate
    predictions, references = evaluate_model(model, dataloader, target_tokenizer, device, 'Transformer')
    
    return predictions, references


def main(args):
    """Main evaluation function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    logger = setup_logger('evaluate')
    logger.info(f'Using device: {device}')
    
    # Load test data
    logger.info('Loading test data...')
    _, _, test_loader, _, _ = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_len=args.max_len
    )
    
    # Evaluate models
    results = {}
    
    if args.model == 'rnn' or args.model == 'all':
        logger.info('Evaluating RNN model...')
        predictions, references = evaluate_rnn(args.checkpoint, test_loader, device)
        bleu_score = calculate_bleu(references, predictions)
        results['RNN'] = bleu_score
        logger.info(f'RNN BLEU Score: {bleu_score:.2f}')
    
    if args.model == 'transformer' or args.model == 'all':
        logger.info('Evaluating Transformer model...')
        predictions, references = evaluate_transformer(args.checkpoint, test_loader, device)
        bleu_score = calculate_bleu(references, predictions)
        results['Transformer'] = bleu_score
        logger.info(f'Transformer BLEU Score: {bleu_score:.2f}')
    
    # Save results
    os.makedirs(args.results_dir, exist_ok=True)
    df = pd.DataFrame(list(results.items()), columns=['Model', 'BLEU Score'])
    results_path = os.path.join(args.results_dir, 'bleu_scores.csv')
    df.to_csv(results_path, index=False)
    logger.info(f'Results saved to {results_path}')
    
    logger.info('Evaluation complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate models')
    parser.add_argument('--model', type=str, choices=['rnn', 'transformer', 'all'], default='all',
                       help='Model to evaluate')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--results_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_len', type=int, default=128, help='Maximum sequence length')
    
    args = parser.parse_args()
    main(args)
