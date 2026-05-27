from abc import ABC, abstractmethod

class Predictor(ABC):
    
    @abstractmethod
    def predict_for_store(self, train_store, test_store):
        pass
