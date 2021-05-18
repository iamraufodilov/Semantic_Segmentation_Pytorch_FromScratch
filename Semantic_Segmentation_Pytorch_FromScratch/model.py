# load libraries
import segmentation_models_pytorch as smp
from training import Trainer

# load the model
model = smp.Unet("resnet18", encoder_weights="imagenet", classes=1, activation=None)

# train the model
model_trainer = Trainer(model)
model_trainer.start()