import torch
from pytorch_lightning import Trainer
from beat_this.dataset import BeatDataModule
from beat_this.model.pl_module import PLBeatThis

# Step 1: Set the Data Directory
# Ensure this path points to your dataset directory
import multiprocessing
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    # Initialize the DataModule
    datamodule = BeatDataModule(
        data_dir="data",
        # batch_size=8,
        num_workers=16,  # Disable multiprocessing
    )

    # Step 3: Initialize the Model
    # Customize model configuration as needed for your training task
    pl_model = PLBeatThis(
        model_type="BeatThisSmall",  # Choose between BeatThisSmall or other variants
        num_filters=100,  # Number of filters for convolutional layers
       #kernel_size=100,  # Kernel size for convolutional layers
       # num_dilations=10,  # Number of dilation layers
       # dropout_rate=0.15,  # Dropout rate for regularization
      #  lr=0.001,  # Initial learning rate
      #  weight_decay=0.02,  # Weight decay for optimizer to prevent overfitting
     #   pos_weights={"beat": 1, "downbeat": 5},  # Loss weights for beat/downbeat imbalance
       # max_epochs=50,  # Maximum number of epochs for training
    )

    # Step 4: Configure the Trainer
    # Configure the Trainer with logging and checkpointing
    trainer = Trainer(
        max_epochs=300,
        precision=16,
        log_every_n_steps=50,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        callbacks=[]
    )


    trainer.fit(pl_model, datamodule)
