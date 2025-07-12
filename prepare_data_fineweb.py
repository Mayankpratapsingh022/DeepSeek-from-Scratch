"""
Memory-Efficient FineWeb-Edu Dataset Preprocessing

This version processes large datasets without loading everything into memory at once.
Key improvements:
- Streaming tokenization
- Direct file writing
- Memory-efficient shuffling
- Progress tracking
- Optional quality filtering
"""

import os
import logging
import numpy as np
import tiktoken
from tqdm.auto import tqdm
from datasets import load_dataset
import random
import re
import pandas as pd
import tempfile
import mmap

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# Main Dataset Processing Options
TRAIN_ON_CUSTOM_ROWS = True      
CUSTOM_ROW_COUNT = 800000 

# Dataset Configuration
DATASET_OPTIONS = [
    ("HuggingFaceFW/fineweb-edu", "CC-MAIN-2024-51"),
]

# Processing Parameters
CONTEXT_LENGTH = 1024       
MIN_TEXT_LENGTH = 50         # Only used when USE_QUALITY_FILTERING = True
TRAIN_SPLIT = 0.95           
RANDOM_SEED = 42            
USE_QUALITY_FILTERING = False  # Set to False to process ALL rows, True to use filtering

# Memory Management
BATCH_SIZE = 8000           # Process in smaller batches
BUFFER_SIZE = 75000         # Tokens to buffer before writing
TEMP_DIR = "temp_tokens"    # Directory for temporary files

# File Output Configuration
TRAIN_FILENAME = "train.bin"
VALIDATION_FILENAME = "validation.bin"
LOG_FILENAME = "dataset_preparation.log"

# Tokenizer Configuration
TOKENIZER_NAME = "gpt2"
DTYPE = np.uint16

# ============================================================================

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILENAME)
    ]
)
logger = logging.getLogger(__name__)

class MemoryEfficientTokenizer:
    """Tokenizes and saves data without loading everything into memory."""
    
    def __init__(self, tokenizer_name, context_length, dtype):
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.context_length = context_length
        self.dtype = dtype
        self.temp_files = []
        
    def create_temp_dir(self):
        """Create temporary directory for intermediate files."""
        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR)
    
    def clean_temp_files(self):
        """Clean up temporary files."""
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        if os.path.exists(TEMP_DIR):
            try:
                os.rmdir(TEMP_DIR)
            except:
                pass
    
    def process_batch_to_temp_file(self, batch_samples, batch_id):
        """Process a batch of samples and save tokens to temporary file."""
        temp_filename = os.path.join(TEMP_DIR, f"batch_{batch_id}.bin")
        token_buffer = []
        
        for text in batch_samples:
            tokens = self.tokenizer.encode_ordinary(text)
            
            # Split into chunks of context_length
            for i in range(0, len(tokens), self.context_length):
                chunk = tokens[i:i + self.context_length]
                if len(chunk) == self.context_length:  # Only complete chunks
                    token_buffer.extend(chunk)
                
                # Write buffer to file when it gets large
                if len(token_buffer) >= BUFFER_SIZE:
                    self._append_tokens_to_file(temp_filename, token_buffer)
                    token_buffer = []
        
        # Write remaining tokens
        if token_buffer:
            self._append_tokens_to_file(temp_filename, token_buffer)
        
        self.temp_files.append(temp_filename)
        return temp_filename
    
    def _append_tokens_to_file(self, filename, tokens):
        """Append tokens to a binary file."""
        token_array = np.array(tokens, dtype=self.dtype)
        with open(filename, 'ab') as f:
            token_array.tofile(f)
    
    def merge_temp_files(self, temp_files, output_filename):
        """Merge temporary files into final output file."""
        logger.info(f"Merging {len(temp_files)} temporary files into {output_filename}")
        
        total_tokens = 0
        with open(output_filename, 'wb') as output_file:
            for temp_file in tqdm(temp_files, desc="Merging files"):
                if os.path.exists(temp_file):
                    # Read and write in chunks to avoid memory issues
                    with open(temp_file, 'rb') as f:
                        while True:
                            chunk = f.read(BUFFER_SIZE * 2)  # Read in bytes
                            if not chunk:
                                break
                            output_file.write(chunk)
                            total_tokens += len(chunk) // 2  # 2 bytes per uint16
        
        logger.info(f"Merged {total_tokens:,} tokens into {output_filename}")
        return total_tokens

class StreamingDataProcessor:
    """Process dataset in streaming fashion without loading all into memory."""
    
    def __init__(self):
        self.tokenizer_processor = MemoryEfficientTokenizer(
            TOKENIZER_NAME, CONTEXT_LENGTH, DTYPE
        )
    
    def clean_text(self, text):
        """Remove excessive whitespace and control characters."""
        text = re.sub(r'\s+', ' ', text)
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        return text.strip()
    
    def is_quality_text(self, text):
        """Apply quality filters to determine if text should be included."""
        if not USE_QUALITY_FILTERING:
            return True  # Accept all texts when filtering is disabled
        
        # Apply quality filters only when enabled
        return len(text.strip()) > MIN_TEXT_LENGTH
    
    def process_dataset_streaming(self, dataset):
        """Process dataset in streaming fashion with memory-efficient approach."""
        logger.info(f"Processing {CUSTOM_ROW_COUNT:,} rows with streaming approach")
        logger.info(f"Quality filtering: {'ENABLED' if USE_QUALITY_FILTERING else 'DISABLED'}")
        
        # Create temporary directory
        self.tokenizer_processor.create_temp_dir()
        
        # Process data in batches
        train_temp_files = []
        val_temp_files = []
        
        current_batch = []
        processed_count = 0
        accepted_count = 0
        train_count = 0
        val_count = 0
        batch_id = 0
        
        progress_bar = tqdm(total=CUSTOM_ROW_COUNT, desc="Processing samples")
        
        try:
            for example in dataset:
                if accepted_count >= CUSTOM_ROW_COUNT:
                    break
                
                text = example.get('text', '')
                if not text:
                    continue
                
                cleaned_text = self.clean_text(text)
                processed_count += 1
                
                if self.is_quality_text(cleaned_text):
                    # Determine if this sample goes to train or validation
                    if random.random() < TRAIN_SPLIT:
                        current_batch.append(('train', cleaned_text))
                        train_count += 1
                    else:
                        current_batch.append(('val', cleaned_text))
                        val_count += 1
                    
                    accepted_count += 1
                    progress_bar.update(1)
                    
                    # Process batch when it reaches target size
                    if len(current_batch) >= BATCH_SIZE:
                        self._process_current_batch(current_batch, batch_id, 
                                                  train_temp_files, val_temp_files)
                        current_batch = []
                        batch_id += 1
                        
                        # Update progress info
                        progress_bar.set_postfix({
                            'train': train_count,
                            'val': val_count,
                            'batches': batch_id,
                            'processed': processed_count,
                            'accepted': accepted_count
                        })
        
        except Exception as e:
            logger.error(f"Error during processing: {e}")
        finally:
            progress_bar.close()
        
        # Process remaining samples
        if current_batch:
            self._process_current_batch(current_batch, batch_id, 
                                      train_temp_files, val_temp_files)
        
        logger.info(f"Processed {processed_count:,} total samples")
        logger.info(f"Accepted {accepted_count:,} samples")
        logger.info(f"Training samples: {train_count:,}, Validation samples: {val_count:,}")
        
        return train_temp_files, val_temp_files
    
    def _process_current_batch(self, batch, batch_id, train_temp_files, val_temp_files):
        """Process current batch and save to appropriate temporary files."""
        train_samples = [text for split, text in batch if split == 'train']
        val_samples = [text for split, text in batch if split == 'val']
        
        if train_samples:
            train_temp_file = self.tokenizer_processor.process_batch_to_temp_file(
                train_samples, f"train_{batch_id}"
            )
            train_temp_files.append(train_temp_file)
        
        if val_samples:
            val_temp_file = self.tokenizer_processor.process_batch_to_temp_file(
                val_samples, f"val_{batch_id}"
            )
            val_temp_files.append(val_temp_file)
    
    def finalize_dataset(self, train_temp_files, val_temp_files):
        """Merge temporary files into final dataset files."""
        # Merge training files
        if train_temp_files:
            train_tokens = self.tokenizer_processor.merge_temp_files(
                train_temp_files, TRAIN_FILENAME
            )
        else:
            train_tokens = 0
        
        # Merge validation files
        if val_temp_files:
            val_tokens = self.tokenizer_processor.merge_temp_files(
                val_temp_files, VALIDATION_FILENAME
            )
        else:
            val_tokens = 0
        
        # Clean up temporary files
        self.tokenizer_processor.clean_temp_files()
        
        return train_tokens, val_tokens

def load_dataset_with_fallbacks():
    """Load dataset with fallback options."""
    for dataset_name, config_name in DATASET_OPTIONS:
        try:
            logger.info(f"Attempting to load {dataset_name}")
            
            if config_name:
                ds = load_dataset(dataset_name, name=config_name, split="train", streaming=True)
            else:
                ds = load_dataset(dataset_name, split="train", streaming=True)
            
            logger.info(f"Successfully loaded {dataset_name}")
            return ds, dataset_name
            
        except Exception as e:
            logger.warning(f"Failed to load {dataset_name}: {str(e)[:100]}")
            continue
    
    raise RuntimeError("Could not load any dataset. Check internet connection.")

def verify_files():
    """Verify the created binary files."""
    logger.info("Verifying output files")
    
    if not (os.path.exists(TRAIN_FILENAME) and os.path.exists(VALIDATION_FILENAME)):
        logger.error("Binary files not found")
        return False
    
    try:
        train_data = np.fromfile(TRAIN_FILENAME, dtype=DTYPE)
        val_data = np.fromfile(VALIDATION_FILENAME, dtype=DTYPE)
        
        logger.info(f"Training tokens: {len(train_data):,}")
        logger.info(f"Validation tokens: {len(val_data):,}")
        logger.info(f"Training sequences: {len(train_data) // CONTEXT_LENGTH:,}")
        logger.info(f"Validation sequences: {len(val_data) // CONTEXT_LENGTH:,}")
        
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False

def main():
    """Main function to prepare the dataset with memory-efficient processing."""
    logger.info("Starting memory-efficient FineWeb-edu dataset preparation")
    logger.info(f"Configuration: {CUSTOM_ROW_COUNT:,} rows, context length {CONTEXT_LENGTH}")
    logger.info(f"Batch size: {BATCH_SIZE}, Buffer size: {BUFFER_SIZE}")
    logger.info(f"Quality filtering: {'ENABLED' if USE_QUALITY_FILTERING else 'DISABLED'}")
    
    # Set random seed
    random.seed(RANDOM_SEED)
    
    # Check if files already exist
    if os.path.exists(TRAIN_FILENAME) and os.path.exists(VALIDATION_FILENAME):
        logger.info("Dataset files already exist")
        verify_files()
        return
    
    # Load dataset
    dataset, dataset_name = load_dataset_with_fallbacks()
    logger.info(f"Using dataset: {dataset_name}")
    
    # Initialize processor
    processor = StreamingDataProcessor()
    
    # Process dataset
    train_temp_files, val_temp_files = processor.process_dataset_streaming(dataset)
    
    # Finalize dataset
    train_tokens, val_tokens = processor.finalize_dataset(train_temp_files, val_temp_files)
    
    # Summary
    logger.info("Dataset preparation completed")
    logger.info(f"Training tokens: {train_tokens:,}")
    logger.info(f"Validation tokens: {val_tokens:,}")
    logger.info(f"Ready for model training")
    
    # Verify output
    verify_files()

if __name__ == "__main__":
    main()