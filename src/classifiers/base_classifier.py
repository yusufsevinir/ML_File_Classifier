from abc import ABC, abstractmethod
from werkzeug.datastructures import FileStorage

class BaseClassifier(ABC):
    @abstractmethod
    def classify(self, file: FileStorage) -> str:
        pass