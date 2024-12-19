import os
import tarfile
from tqdm import tqdm
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

import torch
import torch.optim as optim
from torchsummaryX import summary

from mltu.torch.model import Model
from mltu.torch.losses import CTCLoss
from mltu.torch.dataProvider import DataProvider
from mltu.torch.metrics import CERMetric, WERMetric
from mltu.torch.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Model2onnx, ReduceLROnPlateau
import onnx
from pathlib import Path

from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding, ImageShowCV2
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate, RandomSharpen
from mltu.annotations.images import CVImage

from model import Network
from configs import ModelConfigs

import wandb
from mltu.torch.callbacks import Callback
import pandas as pd

print("Make sure to run this script from the project's root folder !")

wandb.init(project="handwriting_recognition", save_code=True)

configs = ModelConfigs()

dataset, vocab, max_len = [], set(), 0

# Get list of image files
image_files = [f for f in os.listdir(configs.dataset_path) if f.endswith('.png')]

print(f"{len(image_files)} images found in dataset")

# Load dataset (images and labels)
for img_file in tqdm(image_files):
    img_path = os.path.join(configs.dataset_path, img_file)
    label_path = img_path.replace('.png', '.txt')
    
    if not os.path.exists(img_path) or not os.path.exists(label_path):
        print(f"File not found: {img_path} or {label_path}")
        continue

    with open(label_path, 'r') as file:
        label = file.read().strip()

    dataset.append([img_path, label])
    vocab.update(list(label))
    max_len = max(max_len, len(label))

# Save vocab and maximum text length to configs
configs.vocab = "".join(sorted(vocab))
configs.max_text_length = max_len
configs.save()
# Set wandb config
wandb.config.update(vars(configs))

# Create a data provider for the dataset
train_dataProvider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        # ImageShowCV2(), # uncomment to show images when iterating over the data provider
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=False),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=max_len, padding_value=len(configs.vocab))
        ],
    use_cache=True,
)

# Split the dataset into training and validation sets
# train_dataProvider, test_dataProvider = data_provider.split(split=0.9)

# Augment training data with random brightness, rotation and erode/dilate
train_dataProvider.augmentors = [
    RandomBrightness(), 
    RandomErodeDilate(),
    RandomSharpen(),
    RandomRotate(angle=10), 
    ]

# Load evaluation set
eval_df = pd.read_csv("data/test_data/val.csv").values.tolist()

# Create a data provider for the evaluation set
eval_dataProvider = DataProvider(
    dataset=eval_df,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=False),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=max_len, padding_value=len(configs.vocab))
        ],
    use_cache=True,
)

network = Network(len(configs.vocab), activation="leaky_relu", dropout=0.3)
loss = CTCLoss(blank=len(configs.vocab))
optimizer = optim.Adam(network.parameters(), lr=configs.learning_rate)

# uncomment to print network summary, torchsummaryX package is required
summary(network, torch.zeros((1, configs.height, configs.width, 3)))

# put on cuda device if available
if torch.cuda.is_available():
    print("CUDA")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    network = network.cuda()
# elif torch.backends.mps.is_available():
#     print("MPS")
#     network = network.to('mps')
else:
    print("CPU")
    network = network.cpu()

start_epoch = 0
# Load checkpoint if available
# checkpoint_path = os.path.join(configs.model_path, "model.pt")
# if os.path.exists(checkpoint_path):
#     checkpoint = torch.load(checkpoint_path)
#     network.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     start_epoch = checkpoint['epoch'] + 1
#     print(f"Resuming training from epoch {start_epoch}")
# else:
#     print("No checkpoint found, starting training from scratch.")

### Create callbacks

# Create custom WandB callback
class WandbCallback(Callback):
    def __init__(self, network, optimizer, model_path):
        super().__init__()
        self.network = network
        self.optimizer = optimizer
        self.model_path = model_path

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            wandb.log(logs, step=epoch)

        # Save checkpoint
        checkpoint_path = os.path.join(self.model_path, "model.pt")
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch
        }, checkpoint_path)
        
        # Upload checkpoint to WandB
        wandb.save(checkpoint_path)

wandb_callback = WandbCallback(network, optimizer, configs.model_path)

class CheckpointWithOnnx(ModelCheckpoint):
    def __init__(self, input_shape, onnx_save_path, metadata=None, model=None, *args, **kwargs):
        """
        Checkpoint callback avec export ONNX intégré.
        
        Args:
            input_shape: La forme du tensor d'entrée pour l'export ONNX.
            onnx_save_path: Chemin où sauvegarder le modèle ONNX.
            metadata: Dictionnaire contenant les métadonnées à ajouter au modèle ONNX.
        """
        super().__init__(*args, **kwargs)
        self.model = model
        self.input_shape = input_shape
        self.onnx_save_path = Path(onnx_save_path)
        self.metadata = metadata

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

        if self.best == logs.get(self.monitor):
            self.logger.info("Best model detected, exporting to ONNX...")

            # Log best CER and WER to WandB
            best_val_cer = logs.get("val_CER")
            best_val_wer = logs.get("val_WER")
            wandb.log({"best_val_CER": best_val_cer, "best_val_WER": best_val_wer}, step=epoch)

            # Load best model weights
            try:
                state_dict = torch.load(self.filepath)
                self.model.model.load_state_dict(state_dict)
            except Exception as e:
                self.logger.error(f"Failed to load best model for ONNX export: {e}")
                return

            # Prepare model for export
            original_device = next(self.model.model.parameters()).device
            self.model.model.to("cpu")
            self.model.model.eval()

            # Dummy input for ONNX export
            dummy_input = torch.randn(self.input_shape)

            # Export to ONNX
            try:
                torch.onnx.export(
                    self.model.model,
                    dummy_input,
                    self.onnx_save_path,
                    export_params=True,
                    opset_version=14,
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                )
                self.logger.info(f"Model exported to ONNX at {self.onnx_save_path}")

                # Add metadata to ONNX model
                if self.metadata and isinstance(self.metadata, dict):
                    onnx_model = onnx.load(self.onnx_save_path)
                    for key, value in self.metadata.items():
                        meta = onnx_model.metadata_props.add()
                        meta.key = key
                        meta.value = str(value)
                    onnx.save(onnx_model, self.onnx_save_path)

            except Exception as e:
                self.logger.error(f"Failed to export ONNX model: {e}")
                
            # place model back to original device
            self.model.model.to(original_device)

# Create ModelCheckpoint callback to save the best model
checkpoint_with_onnx = CheckpointWithOnnx(
    input_shape=(1, configs.height, configs.width, 3),
    onnx_save_path=os.path.join(configs.model_path, "best_model.onnx"),
    metadata={"vocab": configs.vocab},
    filepath=os.path.join(configs.model_path, "best_model.pt"),
    monitor="val_CER",
    save_best_only=True,
    mode="min",
    verbose=1
)
# earlyStopping = EarlyStopping(monitor="val_CER", patience=300, mode="min", verbose=1)
tb_callback = TensorBoard(configs.model_path + "/logs")
reduce_lr = ReduceLROnPlateau(monitor="val_CER", factor=0.9, patience=10, verbose=1, mode="min", min_lr=1e-6)

# create model object that will handle training and testing of the network
model = Model(network, optimizer, loss, metrics=[CERMetric(configs.vocab), WERMetric(configs.vocab)])

checkpoint_with_onnx.model = model # set model for checkpoint callback

# watch the model with wandb
wandb.watch(network, 'all')


model.fit(
    train_dataProvider, 
    eval_dataProvider, 
    epochs=configs.train_epochs, 
    callbacks=[tb_callback, wandb_callback, checkpoint_with_onnx, reduce_lr],
    initial_epoch=start_epoch
    )

# Save training and validation datasets as csv files
train_dataProvider.to_csv(os.path.join(configs.model_path, "train.csv"))
eval_dataProvider.to_csv(os.path.join(configs.model_path, "val.csv"))


# Save ONNX file to WandB
onnx_model_path = os.path.join(configs.model_path, "best_model.onnx")
wandb.save(onnx_model_path)