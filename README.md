# InKrat
This is the official implementation for paper "InKrat: Interpretable Diagnosis Prediction Models Based on Cross-Modal Knowledge Graph Semantic Retrieval Fusion"

## Project Structure

```
.
├── load_text.py       # Load note data
├── train_multi.py     # Multi-GPU training script
├── test.py            # Model testing and evaluation
└── utils.py           # Model design and utility functions
```

## File Descriptions

### `load_text.py`
Responsible for loading and preprocessing note data. This script prepares the dataset for model training and should be executed first.

### `train_multi.py`
Main training script with multi-GPU support using PyTorch's distributed training. It handles model initialization, loss functions, and the training loop. To run with 4 GPUs:

torchrun --nproc_per_node=4 train_multi.py

### `test.py`
Used for testing and evaluating the trained model on the test set. Outputs performance metrics such as accuracy, precision, recall, etc.

### `utils.py`
Defines the model architecture and related utility functions, such as embedding layers, attention mechanisms, and forward propagation logic.

## Usage Instructions

1. Run `load_text.py` to preprocess and load the note data.
2. Train the model using `train_multi.py` with multi-GPU support.
3. Evaluate the trained model using `test.py`.
4. To modify the model structure, edit `utils.py`.

## Environment Requirements

- Python 3.8 or higher
- PyTorch 1.10 or higher

## Contact

If you have any questions or suggestions, feel free to open an issue or contact the author.
