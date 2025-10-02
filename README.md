# Mini-GPT

A minimal implementation of a GPT-style language model with character-level tokenization, built from scratch using PyTorch. This project demonstrates the core concepts of transformer-based language models in a simple, educational format.

## Features

- **Character-level tokenization** - Simple and effective for small datasets
- **GPT-style transformer architecture** with causal self-attention
- **Flexible training and inference** with various options
- **Model checkpointing** - Save and load trained models
- **Text generation** with temperature control and stop tokens
- **Evaluation tools** - Test model creativity and perplexity

## Installation

1. Clone the repository:
```bash
git clone https://github.com/samiveikko/mini-gpt.git
cd mini-gpt
```

2. Create a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install torch tqdm
```

## Quick Start

### 1. Prepare Your Data

Create a `corpus.txt` file with your training text:
```bash
echo "Your training text goes here. The model will learn patterns from this text." > corpus.txt
```

### 2. Train the Model

```bash
python script.py
```

This will:
- Load your corpus
- Train the model for 3000 iterations
- Save the model as `mini_gpt_char.pt`
- Generate sample text with default prompt "machine learning"

### 3. Generate Text

Load a trained model and generate text:
```bash
python script.py --load mini_gpt_char.pt --prompt "Your prompt here"
```

## Usage Examples

### Basic Training and Generation
```bash
# Train a new model
python script.py

# Generate text with default prompt "machine learning"
python script.py --load mini_gpt_char.pt

# Generate with custom prompt
python script.py --load mini_gpt_char.pt --prompt "cat eats"
```

### Advanced Generation Options
```bash
# Control generation length and temperature
python script.py --load mini_gpt_char.pt --prompt "dog" --max-tokens 200 --temperature 0.8

# Stop generation at specific characters
python script.py --load mini_gpt_char.pt --prompt "bird" --stop-tokens ".," --max-tokens 100

# Lower temperature for more focused generation
python script.py --load mini_gpt_char.pt --prompt "cat" --temperature 0.3
```

### Interactive Testing
```bash
# Test model creativity with various prompts
python script.py --load mini_gpt_char.pt --test-creativity

# Interactive mode for testing different prompts
python script.py --load mini_gpt_char.pt --interactive

# Calculate perplexity for specific text
python script.py --load mini_gpt_char.pt --perplexity "your test text here"
```

Note: The current script.py implementation includes `--test-creativity`, `--interactive`, and `--perplexity` features, but these may not be fully implemented in the current version.

## Configuration

You can modify the model configuration in `script.py`:

```python
@dataclass
class Config:
    data_path: str = "corpus.txt"     # Your training data
    block_size: int = 64              # Context length (tokens)
    batch_size: int = 32              # Training batch size
    n_layers: int = 4                 # Number of transformer layers
    n_heads: int = 4                  # Number of attention heads
    n_embd: int = 128                 # Embedding dimension
    dropout: float = 0.1              # Dropout rate
    lr: float = 3e-4                  # Learning rate
    max_iters: int = 3000             # Training iterations
    eval_interval: int = 300          # Evaluation frequency
    eval_iters: int = 100             # Evaluation iterations
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42                    # Random seed
    ckpt_path: str = "mini_gpt_char.pt"
```

## Command Line Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--load` | Load existing model | `--load mini_gpt_char.pt` |
| `--prompt` | Text prompt for generation (default: "machine learning") | `--prompt "cat"` |
| `--max-tokens` | Maximum tokens to generate (default: 100) | `--max-tokens 200` |
| `--temperature` | Generation temperature (default: 1.0) | `--temperature 1.2` |
| `--stop-tokens` | Stop generation at these chars | `--stop-tokens ".,!"` |

## Tips for Better Results

### 1. Corpus Size and Quality
- **More data is better**: Aim for at least 10,000+ characters
- **Clean text**: Remove formatting, normalize whitespace
- **Diverse content**: Include different sentence types and topics

### 2. Model Configuration
- **Block size**: Should be smaller than your corpus length
- **Training iterations**: More iterations = better learning
- **Temperature**: Lower (0.3-0.7) for focused text, higher (1.2-2.0) for creativity

### 3. Generation Tips
- **Start simple**: Use short, common words as prompts
- **Use stop tokens**: Prevent runaway generation
- **Experiment with temperature**: Find the right balance for your use case

## Example Corpus

Here's an example of good training data structure:

```
Cats are animals that love to sleep and play with yarn.
Dogs are loyal companions that enjoy fetching balls and going for walks.
Birds can fly high in the sky and build nests in trees.
Fish live in water and swim gracefully through the ocean.
Horses are strong animals that people ride for transportation.
Cows give us milk and graze peacefully in green meadows.
Rabbits hop quickly and have long ears for listening.
Bears are large animals that hibernate during winter months.
Elephants are the biggest land animals with long trunks for eating.
Lions are known as the king of the jungle with their mighty roar.
Tigers have beautiful striped fur and are excellent hunters.
Monkeys swing from tree to tree using their long arms.
Dolphins are intelligent sea creatures that love to jump and play.
Whales are the largest animals on Earth living in the deep ocean.
Butterflies have colorful wings and transform from caterpillars.
Bees make honey by collecting nectar from flowers and plants.
Frogs can jump very high and live both on land and in water.
Turtles carry their homes on their backs and live very long lives.
Penguins cannot fly but are excellent swimmers in cold waters.
Owls can see in the dark and hunt mice during the night.
```

## Troubleshooting

### Common Issues

1. **"block_size too large" error**
   - Reduce `block_size` in config or add more data to corpus

2. **Model generates repetitive text**
   - Try higher temperature (`--temperature 1.5`)
   - Use stop tokens (`--stop-tokens "."`)
   - Train for more iterations

3. **"Character not in vocabulary" error**
   - Use only characters that exist in your training corpus
   - Check available characters with `--interactive` mode

4. **Model gets stuck in loops**
   - Use sampling instead of deterministic generation
   - Try different prompts
   - Increase temperature

## File Structure

```
mini-gpt/
├── script.py              # Main training and inference script
├── corpus.txt             # Your training data
├── mini_gpt_char.pt       # Trained model checkpoint (ignored by git)
├── .gitignore             # Ignores model files
└── README.md              # This file
```

## Model Architecture

The model uses a simplified GPT architecture:

- **Token Embeddings**: Map characters to vectors
- **Position Embeddings**: Add positional information
- **Multi-Head Self-Attention**: Causal attention mechanism
- **Feed-Forward Networks**: Process attention outputs
- **Layer Normalization**: Stabilize training
- **Output Head**: Predict next character probabilities

## Contributing

Feel free to contribute improvements:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

This implementation is inspired by Andrej Karpathy's nanoGPT and the original GPT paper. It's designed for educational purposes to understand transformer architectures and language modeling.
