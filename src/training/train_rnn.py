import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.loader import get_dataloaders
from src.models.rnn_model import EncoderRNN, DecoderRNN, Seq2SeqRNN
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
        
        output = model(source, target, teacher_forcing_ratio=0.5)
        
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        target = target[:, 1:].reshape(-1)
        
        loss = criterion(output, target)
        loss.backward()
        
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
            
            output = model(source, target, teacher_forcing_ratio=0.0)
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            target = target[:, 1:].reshape(-1)
            
            loss = criterion(output, target)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train_rnn(args):
    """Main training function."""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    logger = setup_logger('train_rnn')
    logger.info(f'Using device: {device}')
    
    # Load data
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
    encoder = EncoderRNN(
        vocab_size=len(source_tokenizer),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    decoder = DecoderRNN(
        vocab_size=len(target_tokenizer),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    model = Seq2SeqRNN(encoder, decoder, device)
    model = model.to(device)
    
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0) 
    
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss = validate(model, val_loader, criterion, device)
        
        logger.info(f'Epoch {epoch+1}/{args.num_epochs}')
        logger.info(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(args.model_dir, exist_ok=True)
            model_path = os.path.join(args.model_dir, 'rnn_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'source_tokenizer': source_tokenizer,
                'target_tokenizer': target_tokenizer
            }, model_path)
            logger.info(f'Saved best model to {model_path}')
    
    logger.info('Training complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RNN model')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--model_dir', type=str, default='models', help='Model directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_len', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    
    args = parser.parse_args()
    train_rnn(args)
