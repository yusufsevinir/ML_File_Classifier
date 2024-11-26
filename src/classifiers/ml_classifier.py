# src/classifiers/ml_classifier.py

from .base_classifier import BaseClassifier
from src.extraction.text_extractor import extract_text
from src.utils.text_preprocessing import preprocess_text
from werkzeug.datastructures import FileStorage
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import logging
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class MLClassifier(BaseClassifier):
    def __init__(self):
        # Update path to use src/models
        model_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'models',
            'document_classifier'
        )
        logger.info(f"Loading model from: {model_dir}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        # Ensure model is in evaluation mode
        self.model.eval()

        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Mapping from label IDs to label names
        self.id2label = self.model.config.id2label

        logger.info(f"Initialized ML Classifier with device: {self.device}")
        logger.info(f"Available labels: {self.id2label}")

    def classify(self, file: FileStorage) -> dict:
        logger.debug("=== Starting ML Classification ===")
        
        # Extract and preprocess text
        text = extract_text(file)
        text_processed = preprocess_text(text)
        logger.debug(f"Preprocessed text length: {len(text_processed)}")
        logger.debug(f"Text preview: {text_processed[:200]}")

        # Tokenize the input text
        inputs = self.tokenizer(
            text_processed,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        logger.debug(f"Input token length: {len(inputs['input_ids'][0])}")
        if len(inputs['input_ids'][0]) == 512:
            logger.warning("Input was truncated to 512 tokens")

        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Get predictions with confidence scores
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=1)
            
            # Get top 3 predictions with their probabilities
            top_probs, top_indices = torch.topk(probabilities, k=min(3, len(self.id2label)))
            
            predictions = []
            for prob, idx in zip(top_probs[0], top_indices[0]):
                label = self.id2label[idx.item()]
                confidence = prob.item()
                predictions.append((label, confidence))
                logger.debug(f"Label: {label}, Confidence: {confidence:.4f}")

            # Return the top prediction with additional debug info
            return {
                "category": predictions[0][0],
                "confidence": predictions[0][1],
                "debug": {
                    "all_predictions": [
                        {"label": label, "confidence": f"{conf:.4f}"} 
                        for label, conf in predictions
                    ],
                    "text_length": len(text_processed),
                    "token_length": len(inputs['input_ids'][0])
                }
            }