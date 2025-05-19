# 🧠 SetFit Text Classification

[![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/github/license/yourusername/your-repo)](LICENSE)
![Stars](https://img.shields.io/github/stars/yourusername/your-repo)
![Forks](https://img.shields.io/github/forks/yourusername/your-repo)
![Issues](https://img.shields.io/github/issues/yourusername/your-repo)
![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/your-repo)


## 📌 Overview

This project demonstrates how to fine-tune a [SetFit](https://github.com/huggingface/setfit) model for text classification tasks. SetFit (Sentence Transformer Fine-tuning) is an efficient and prompt-free framework that enables few-shot learning by fine-tuning sentence transformers with contrastive learning, followed by training a classification head.

## 🚀 Features

* **Few-Shot Learning**: Achieve high accuracy with as few as 8 labeled examples per class.
* **Prompt-Free**: No need for handcrafted prompts or verbalizers.
* **Fast Training**: Efficient training suitable for limited computational resources.
* **Multilingual Support**: Utilize multilingual sentence transformers for various languages.

## 📂 Dataset Structure

Ensure your dataset is in CSV format with the following columns:

* `text`: The input text to classify.
* `label`: The corresponding label for the text.

Example:

```csv
text,label
"I love this product!",positive
"This is the worst experience I've had.",negative
```

## 🛠️ Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
pip install -r requirements.txt
```

## 📈 Training the Model

Use the provided script to train the SetFit model:

```bash
python train.py --train_path data/train.csv --eval_path data/validation.csv --model_name sentence-transformers/paraphrase-mpnet-base-v2 --batch_size 16 --num_epochs 3
```

Replace `data/train.csv` and `data/validation.csv` with the paths to your training and validation datasets, respectively.

## 🔍 Evaluation

After training, evaluate the model's performance:

```bash
python evaluate.py --model_path output/model --test_path data/test.csv
```

Replace `output/model` with the path to your trained model and `data/test.csv` with your test dataset.

## 📊 Results

| Metric   | Score |
| -------- | ----- |
| Accuracy | 0.92  |
| F1-Score | 0.91  |

*Note: Replace the above scores with your actual evaluation results.*

## 📚 References

* [SetFit: Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
* [Hugging Face SetFit Repository](https://github.com/huggingface/setfit)
* [Hugging Face Blog on SetFit](https://huggingface.co/blog/setfit)

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

