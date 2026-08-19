# Self-Supervised Wav2Vec 2.0 ASR Fine-Tuning

This repository contains a complete pipeline for fine-tuning a self-supervised speech representation model, specifically [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53), for Automatic Speech Recognition (ASR). The project is implemented in a Jupyter Notebook using PyTorch and the Hugging Face `transformers` and `datasets` libraries.

## 🚀 Features

- **Data Loading & Preprocessing**: Loads custom speech datasets from CSV files and processes audio and transcripts.
- **Custom Tokenizer**: Builds a character-level vocabulary and a CTC tokenizer from the dataset's transcripts.
- **Audio Processing**: Handles audio resampling (to 16kHz) and feature extraction using `torchaudio` and `librosa`.
- **Fine-tuning**: Fine-tunes the cross-lingual Wav2Vec 2.0 (XLSR-53) model using Connectionist Temporal Classification (CTC) loss.
- **Evaluation**: Computes Word Error Rate (WER) during training to evaluate model performance.

## 📁 Repository Structure

- `ASRpretrained_(1)_(2).ipynb`: The main Jupyter notebook containing the entire end-to-end code for dataset preparation, vocabulary generation, preprocessing, model configuration, and training.

## 🛠️ Prerequisites

To run the notebook, you will need the following libraries:

- `torch` and `torchaudio`
- `transformers`
- `datasets`
- `librosa`
- `jiwer` (for Word Error Rate calculation)
- `pandas`, `numpy`, `IPython`

You can install the required dependencies using pip:
```bash
pip install torch torchaudio transformers datasets librosa jiwer pandas numpy
```

## 🧠 Workflow Overview

1. **Dataset Preparation**: The dataset is loaded from CSV files (`final_final.csv` and `devansh_final_1_2.csv`) that include audio file paths and corresponding transcripts.
2. **Text Normalization**: Special characters and punctuation are removed, and the text is converted to lowercase.
3. **Vocabulary Creation**: Extracts all unique characters across the training and testing datasets to create a custom vocabulary (`vocab.json`).
4. **Feature Extraction**: Uses `Wav2Vec2FeatureExtractor` to process audio arrays and `Wav2Vec2CTCTokenizer` to process transcripts.
5. **Resampling**: Ensures all audio clips are resampled to 16,000 Hz, which is the required sampling rate for Wav2Vec 2.0 models.
6. **Training**: Uses Hugging Face's `Trainer` class along with a dynamic padding data collator (`DataCollatorCTCWithPadding`) to fine-tune the `facebook/wav2vec2-large-xlsr-53` model.

## ⚙️ Model Training Configurations

The model is trained using mixed-precision (`fp16`) for memory efficiency and faster training. Some of the key hyperparameters used in the `TrainingArguments` include:

- `learning_rate`: 3e-4
- `per_device_train_batch_size`: 32
- `gradient_accumulation_steps`: 2
- `num_train_epochs`: 200
- `warmup_steps`: 500
- `evaluation_strategy`: "steps"

## 🚀 How to Run

1. Clone this repository.
2. Ensure you have the required datasets available as CSV files. Update the paths to the CSV files inside the notebook (`/content/final_final.csv`, etc.).
3. If running on Google Colab, make sure to mount your Google Drive as the notebook saves checkpoints and the processor to a Drive path.
4. Execute the cells in `ASRpretrained_(1)_(2).ipynb` sequentially.

## 📝 License

This project is open-source and available under standard open-source licenses. Please refer to the specific license terms of the underlying models and datasets used.
