# DeepSeek V3 from Scratch

A complete implementation of the DeepSeek V3 architecture with modern transformer innovations including Multi-Head Latent Attention (MLA), Mixture of Experts (MoE), and Multi-Token Prediction (MTP). This project demonstrates the implementation of a 100+ million parameter language model trained on the FineWeb-Edu dataset.

## Architecture Overview


![DeepSeek architecture](https://github.com/user-attachments/assets/8751e031-61e8-4ef2-9823-5e4316bd6356)

DeepSeek V3 introduces several key architectural improvements over traditional transformer models:

### Core Innovations

**Multi-Head Latent Attention (MLA)**
 ![Multi-Head-Latent-Attention](https://github.com/user-attachments/assets/564a2bf0-ab76-4a50-ae91-2f3eadef337d)
- Compresses key-value pairs into shared latent representations
- Dramatically reduces memory usage compared to traditional multi-head attention
- Maintains the expressiveness of multiple attention heads while using significantly less memory
- Critical for handling longer sequences efficiently






**Mixture of Experts (MoE)**
 ![Mixture of Experts](https://github.com/user-attachments/assets/d7a4196d-753f-4aa5-9534-067c2a84c0ae)
- Replaces dense feed-forward networks with sparse expert networks
- Uses 8 experts but only activates 2 per token
- Achieves 4x model capacity with only 25% computational overhead
- Each expert specializes in different domains (numbers, language, code, etc.)

**Multi-Token Prediction (MTP)**
 ![Multi Token prediction](https://github.com/user-attachments/assets/52051bc1-641e-44f4-af4e-63f64f133a64)
- Predicts multiple tokens simultaneously during training
- Improves training efficiency by providing more learning signals per forward pass
- Enables faster inference through speculative decoding

**Additional Components**
- **RoPE (Rotary Positional Encoding)**: Better handling of longer sequences and relative positions
- **RMS Norm**: Computationally simpler normalization without mean centering
- **SwiGLU Activation**: Gated activation function for improved information flow control


Final Model Weights 

https://huggingface.co/Mayank022/DeepSeek-V3-from-Scratch/tree/main


## Model Configuration

<img width="1920" height="1080" alt="Model Summary" src="https://github.com/user-attachments/assets/7613bf81-55da-47ff-a31b-21fd46fbae19" />

### Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model Parameters | 109,032,032 | Total trainable parameters |
| Vocabulary Size | 50,257 | Number of unique tokens |
| Block Size | 1,024 | Maximum sequence length |
| Embedding Dimension | 512 | Hidden dimension size |
| Number of Layers | 8 | Transformer blocks |
| Attention Heads | 8 | Multi-head attention |
| Batch Size | 32 | Training batch size |
| Learning Rate | 0.0003 | Initial learning rate |
| Min Learning Rate | 0.00001 | Minimum learning rate |
| Warmup Steps | 2,000 | Learning rate warmup |
| Max Iterations | 20,000 | Maximum training steps |
| Dropout | 0.1 | Dropout probability |
| Gradient Accumulation | 8 | Steps before optimizer update |

### MoE Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Number of Experts | 8 | Total expert networks |
| Experts per Token | 2 | Active experts per forward pass |
| Expert Efficiency | 25% | Computation vs full dense model |
| Capacity Multiplier | 4x | Model capacity increase |

### Attention Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| KV LoRA Rank | 128 | Key-Value compression rank |
| Q LoRA Rank | 192 | Query compression rank |
| MTP Heads | 1 | Multi-token prediction heads |

## Dataset

**Primary Dataset**: FineWeb-Edu (CC-MAIN-2024 subset)

https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1

https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu 

- Total available records: 13 million
- Used for training: 2 million records
- Training tokens: 2.5 billion
- Validation tokens: 132.8 million
- Format: Educational web content optimized for language model training

**Fallback Dataset**: TinyStories

https://huggingface.co/datasets/roneneldan/TinyStories

- Used for initial prototyping and architecture validation
- Simpler content for testing basic functionality

Paper

https://arxiv.org/pdf/2412.19437


## Project Structure

```
DeepSeek-from-Scratch/
├── models/
│   ├── attention.py          # Multi-Head Latent Attention implementation
│   ├── config.py            # Model configuration parameters
│   ├── layers.py            # RoPE, RMS Norm, SwiGLU implementations
│   ├── model.py             # Main DeepSeek transformer block
│   ├── moe.py               # Mixture of Experts implementation
│   └── mtp.py               # Multi-Token Prediction implementation
├── training/
│   ├── data_loader.py       # Dataset loading and preprocessing
│   └── trainer.py           # Training loop and optimization
├── inference/
│   ├── generator.py         # Text generation utilities
│   └── run_inference.py     # Inference script
├── notebooks/
│   ├── Mixture_of_Experts_from_Scratch.ipynb
│   ├── Multi_Head_Latent_Attention_From_Scratch.ipynb
│   └── Multi_Token_Prediction_from_Scratch.ipynb
├── prepare_data_fineweb.py  # FineWeb dataset preparation
├── prepare_data_tiny_stories.py  # TinyStories dataset preparation
└── main.py                  # Main training script
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/username/DeepSeek-from-Scratch.git
cd DeepSeek-from-Scratch
```

2. Create and activate virtual environment:
```bash
python -m venv deepseek_env
source deepseek_env/bin/activate  # Linux/Mac
# or
deepseek_env\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

### Data Preparation

Prepare the FineWeb-Edu dataset:
```bash
python prepare_data_fineweb.py
```

Or use TinyStories for testing:
```bash
python prepare_data_tiny_stories.py
```

### Start Training

```bash
python main.py train
```

Training configurations can be modified in `models/config.py`.

### Monitoring

Training progress is tracked using Weights & Biases:
- Model checkpoints are saved automatically
- Loss curves and metrics are logged in real-time
- Training time: Approximately 7 hours on A100 80GB
- Cost: $9.53 for full training run

## Inference

Update the test prompts in `inference/run_inference.py` and run:

```bash
python inference/run_inference.py
```

The model will generate text completions based on your input prompts.

## Key Implementation Details

### Multi-Head Latent Attention

Traditional multi-head attention stores separate key-value pairs for each head, leading to significant memory overhead. MLA compresses these into shared latent representations:

- Memory reduction: Proportional to number of attention heads
- Performance maintenance: Retains expressiveness of full multi-head attention
- Scalability: Enables training with longer sequences

### Mixture of Experts

Instead of processing every token through the same large feed-forward network, MoE routes tokens to specialized experts:

- Sparse activation: Only 25% of the model is active per token
- Specialization: Different experts learn different types of patterns
- Efficiency: 4x capacity increase with minimal computational overhead

### Multi-Token Prediction

Enhances training by predicting multiple future tokens simultaneously:

- Training efficiency: More learning signals per forward pass
- Inference optimization: Enables speculative decoding techniques
- Performance improvement: Better gradient flow during training

## Performance Metrics

### Training Results

- **Final Loss**: Achieved convergence after 20,000 iterations
- **Training Time**: 7 hours 1 minute on NVIDIA A100 80GB
- **Memory Usage**: Efficient memory utilization with MLA compression
- **Convergence**: Stable training with proper learning rate scheduling

### Model Efficiency

- **Parameter Efficiency**: 109M parameters with MoE sparse activation
- **Memory Efficiency**: Reduced KV cache through latent attention
- **Computational Efficiency**: 25% active parameters per forward pass

## Technical Challenges Addressed

### Dataset Selection

The choice of dataset was critical for demonstrating the architecture's benefits:

- **TinyStories**: Too simple, didn't justify advanced architecture components
- **Raw Web Data**: Too complex for resource-constrained training
- **FineWeb-Edu**: Perfect balance of complexity and educational content quality

### Architecture Decisions

Careful consideration was given to which components to include:

- **Essential Components**: MLA, MoE, MTP all included for comprehensive implementation
- **Training Constraints**: Context length limited to 1024 tokens due to compute budget
- **Resource Management**: Balanced model size with available GPU memory

## Future Enhancements

### Planned Improvements

- **Dataset Expansion**: Experiment with larger subsets of FineWeb-Edu
- **Evaluation Metrics**: Implement comprehensive benchmarking suite
- **Architecture Extensions**: Additional transformer innovations and optimizations
- **Scaling Studies**: Analysis of performance across different model sizes


