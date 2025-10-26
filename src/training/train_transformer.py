import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.loader import get_dataloaders
from src.models.transformer_model import Transformer
from src.utils.logger import setup_logger


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        source = batch['source'].to(device)
        target = batch['target'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(source, target)
        
        # Reshape for loss calculation
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        target = target[:, 1:].reshape(-1)
        
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            
            output = model(source, target)
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            target = target[:, 1:].reshape(-1)
            
            loss = criterion(output, target)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train_transformer(args):
    """Main training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    logger = setup_logger('train_transformer')
    logger.info(f'Using device: {device}')
    
    logger.info('Loading data...')
    try:
        train_loader, val_loader, test_loader, source_tokenizer, target_tokenizer = get_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            max_len=args.max_len
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    logger.info('Creating model...')
    model = Transformer(
        src_vocab_size=len(source_tokenizer),
        tgt_vocab_size=len(target_tokenizer),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_ff=args.d_ff,
        max_seq_length=args.max_len,
        dropout=args.dropout
    )
    model = model.to(device)
    
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token
    
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss = validate(model, val_loader, criterion, device)
        
        logger.info(f'Epoch {epoch+1}/{args.num_epochs}')
        logger.info(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(args.model_dir, exist_ok=True)
            model_path = os.path.join(args.model_dir, 'transformer_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'source_tokenizer': source_tokenizer,
                'target_tokenizer': target_tokenizer,
                'model_config': {
                    'd_model': args.d_model,
                    'num_heads': args.num_heads,
                    'num_encoder_layers': args.num_encoder_layers,
                    'num_decoder_layers': args.num_decoder_layers,
                    'd_ff': args.d_ff,
                    'max_seq_length': args.max_len,
                    'dropout': args.dropout
                }
            }, model_path)
            logger.info(f'Saved best model to {model_path}')
    
    logger.info('Training complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Transformer model')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--model_dir', type=str, default='models', help='Model directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_len', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=6, help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    
    args = parser.parse_args()
    train_transformer(args)
