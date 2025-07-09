"""
Data Preparation Script for DeepSeek-V3

This script downloads and tokenizes the TinyStories dataset.
Run this BEFORE training the model.
"""

import os
import numpy as np
import tiktoken
from tqdm.auto import tqdm
from datasets import load_dataset

def prepare_tinystories_dataset():
    """Download and tokenize TinyStories dataset."""
    
    print("=" * 60)
    print("PREPARING TINYSTORIES DATASET")
    print("=" * 60)
    
    # Check if already prepared
    if os.path.exists("train.bin") and os.path.exists("validation.bin"):
        print("✓ Dataset files already exist!")
        
        # Show file info
        train_size = os.path.getsize("train.bin") / (1024 * 1024)
        val_size = os.path.getsize("validation.bin") / (1024 * 1024)
        
        train_data = np.memmap('train.bin', dtype=np.uint16, mode='r')
        val_data = np.memmap('validation.bin', dtype=np.uint16, mode='r')
        
        print(f"├── train.bin: {len(train_data):,} tokens ({train_size:.1f} MB)")
        print(f"└── validation.bin: {len(val_data):,} tokens ({val_size:.1f} MB)")
        return
    
    print("Downloading TinyStories dataset...")
    
    # Load dataset
    ds = load_dataset("roneneldan/TinyStories")
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    def process_example(example):
        """Tokenize text."""
        ids = enc.encode_ordinary(example['text'])
        return {'ids': ids, 'len': len(ids)}
    
    print("Tokenizing dataset...")
    
    # Tokenize
    tokenized = ds.map(
        process_example,
        remove_columns=['text'],
        desc="Tokenizing splits",
        num_proc=8,
    )
    
    # Create binary files
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = f'{split}.bin'
        dtype = np.uint16  # GPT-2 vocab size < 2^16
        
        print(f"Creating {filename}...")
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024
        
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'Writing {filename}'):
            batch = dset.shard(
                num_shards=total_batches, 
                index=batch_idx, 
                contiguous=True
            ).with_format('numpy')
            
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        
        arr.flush()
        
        size_mb = os.path.getsize(filename) / (1024 * 1024)
        print(f"✓ {filename}: {arr_len:,} tokens ({size_mb:.1f} MB)")
    
    print("\n Dataset preparation completed!")
    print("You can now run: python main.py train")


if __name__ == "__main__":
    prepare_tinystories_dataset()