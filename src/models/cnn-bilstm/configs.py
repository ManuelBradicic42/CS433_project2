import os
from datetime import datetime

from mltu.configs import BaseModelConfigs


class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join("models/handwriting_recognition_torch", datetime.strftime(datetime.now(), "%Y%m%d%H%M"))
        self.dataset_path = "data/example_data"
        self.vocab = ""
        self.height = 32
        self.width = 128
        self.max_text_length = 0
        self.batch_size = 64
        self.learning_rate = 0.005
        self.train_epochs = 30
