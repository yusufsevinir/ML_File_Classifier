import os
import sys
import json
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add project root to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text_preprocessing import preprocess_text
from data_generation.synthetic_data_generator import create_synthetic_dataset

def compute_metrics(pred):
    """Compute metrics for model evaluation"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def prepare_data(texts, labels):
    """Prepare and split the dataset"""
    # Create label mappings
    unique_labels = sorted(list(set(labels)))  # Sort for consistency
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    # Convert labels to ids
    numeric_labels = [label2id[label] for label in labels]
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, numeric_labels, test_size=0.2, random_state=42, stratify=numeric_labels
    )
    
    return (
        train_texts, val_texts, train_labels, val_labels,
        label2id, id2label
    )

def train_model(train_texts, val_texts, train_labels, val_labels, label2id, id2label):
    """Train the model with document classification settings"""
    # Use a larger model for better performance
    model_name = 'bert-base-uncased'  # Changed from distilbert for better accuracy
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label
    )

    # Create datasets
    train_dataset = Dataset.from_dict({
        'text': train_texts,
        'label': train_labels
    })
    val_dataset = Dataset.from_dict({
        'text': val_texts,
        'label': val_labels
    })

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

    # Tokenize datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # Set format for PyTorch
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # Updated training arguments for better performance
    training_args = TrainingArguments(
        output_dir=os.path.join(
            os.path.dirname(__file__),
            '..',
            'models',
            'document_classifier'
        ),
        num_train_epochs=10,          # Increased from 5
        per_device_train_batch_size=16,  # Increased from 8
        per_device_eval_batch_size=16,   # Increased from 8
        learning_rate=3e-5,           # Slightly increased from 2e-5
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=100,               # Increased from 50
        save_strategy="steps",
        save_steps=100,               # Increased from 50
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        logging_steps=50,             # Increased from 10
        warmup_ratio=0.1,            # Changed from warmup_steps to ratio
        seed=42,
        report_to="none",
        no_cuda=True
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Train model
    trainer.train()
    
    return model, tokenizer, label2id, id2label

def save_model_artifacts(model, tokenizer, label2id, id2label, save_directory):
    """Save model and associated artifacts"""
    os.makedirs(save_directory, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    
    # Save label mappings
    with open(os.path.join(save_directory, 'label_mappings.json'), 'w') as f:
        json.dump({
            'label2id': label2id,
            'id2label': id2label
        }, f, indent=2)

def main():
    # Increase dataset size
    print("Generating dataset...")
    data = create_synthetic_dataset(num_samples=500)  # Increased from 1000
    
    # 2. Preprocess data
    print("Preprocessing data...")
    texts = [preprocess_text(item['text']) for item in data]
    labels = [item['label'] for item in data]
    
    # 3. Prepare data
    print("Preparing data...")
    train_texts, val_texts, train_labels, val_labels, label2id, id2label = prepare_data(
        texts, labels
    )
    
    # 4. Train model
    print("Training model...")
    model, tokenizer, label2id, id2label = train_model(
        train_texts, val_texts, train_labels, val_labels,
        label2id, id2label
    )
    
    # 5. Save model and artifacts
    print("Saving model...")
    save_directory = os.path.join(os.path.dirname(__file__), '../models/document_classifier')
    save_model_artifacts(model, tokenizer, label2id, id2label, save_directory)
    
    print("Training completed!")

if __name__ == '__main__':
    main()
