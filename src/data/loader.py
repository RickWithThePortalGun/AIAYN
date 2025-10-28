import pickle
import os
import sys
from torch.utils.data import DataLoader

# Add parent directory to path to enable imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.prepare_data import TranslationDataset


def load_tokenizers(data_dir='data'):
    """Load tokenizers from disk."""
    try:
        with open(os.path.join(data_dir, 'source_tokenizer.pkl'), 'rb') as f:
            source_tokenizer = pickle.load(f)
        
        with open(os.path.join(data_dir, 'target_tokenizer.pkl'), 'rb') as f:
            target_tokenizer = pickle.load(f)
        
        return source_tokenizer, target_tokenizer
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Tokenizer files not found in {data_dir}. Please run data preparation first.") from e
    except Exception as e:
        raise RuntimeError(f"Error loading tokenizers: {e}") from e


def load_data_split(data_dir, split_name):
    """Load a data split from disk."""
    try:
        with open(os.path.join(data_dir, f'{split_name}_data.pkl'), 'rb') as f:
            data = pickle.load(f)
        
        return data['sources'], data['targets']
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Data split '{split_name}' not found in {data_dir}. Please run data preparation first.") from e
    except Exception as e:
        raise RuntimeError(f"Error loading data split '{split_name}': {e}") from e


def create_dataloader(data_dir, split_name, source_tokenizer, target_tokenizer, 
                     batch_size=32, max_len=128, shuffle=False):
    """Create a DataLoader for a split."""
    sources, targets = load_data_split(data_dir, split_name)
    
    dataset = TranslationDataset(
        sources, targets, source_tokenizer, target_tokenizer, max_len=max_len
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True
    )
    
    return dataloader


def get_dataloaders(data_dir='data', batch_size=32, max_len=128):
    """Get all dataloaders for train, val, and test."""
    # Load tokenizers
    source_tokenizer, target_tokenizer = load_tokenizers(data_dir)
    
    # Create dataloaders
    train_loader = create_dataloader(
        data_dir, 'train', source_tokenizer, target_tokenizer,
        batch_size=batch_size, max_len=max_len, shuffle=True
    )
    
    val_loader = create_dataloader(
        data_dir, 'val', source_tokenizer, target_tokenizer,
        batch_size=batch_size, max_len=max_len, shuffle=False
    )
    
    test_loader = create_dataloader(
        data_dir, 'test', source_tokenizer, target_tokenizer,
        batch_size=batch_size, max_len=max_len, shuffle=False
    )
    
    return train_loader, val_loader, test_loader, source_tokenizer, target_tokenizer
