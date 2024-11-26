from .ml_classifier import MLClassifier
from .rule_based_classifier import RuleBasedClassifier

def get_classifier(classifier_type='rule_based'):
    """
    Factory method to initialize classifiers.
    """
    if classifier_type == 'ml_based':
        return MLClassifier()
    else:
        return RuleBasedClassifier()