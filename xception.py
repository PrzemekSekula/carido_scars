"""
This module contains the main function for training and testing the model.
Autor: Przemek Sekula
Created: 2025-10-04
Last modified: 2025-10-04
"""

# %%

import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from data_tools import ImageDataModule
from model import ImageClassifier
from tools import load_config


# %%

def main(config, data_module, model):
   
    logger = TensorBoardLogger(
        save_dir=".", # Saves logs in the current directory
        name=config["log_dir"], # Creates a subfolder with this name
        default_hp_metric=False # Optional: disables a default metric
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc', 
        dirpath='checkpoints/', 
        filename=f'{config["model_name"]}-best-checkpoint', 
        save_top_k=1, 
        mode='max'
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        verbose=True, 
        mode='min'
    )
    
    trainer = pl.Trainer(
        max_epochs=config["num_epochs"], 
        accelerator="auto", 
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=logger 
    )
    
    print("\n--- Starting Training ---")
    trainer.fit(model, datamodule=data_module)
    
    print("\n--- Starting Testing on the Best Model ---")
    # Capture the results from trainer.test(), now including the model
    test_results = trainer.test(
        model=model, 
        datamodule=data_module, 
        ckpt_path='best')


    # 2. Define the path for the results file inside the specific run's log folder
    results_path = os.path.join(trainer.logger.log_dir, "results.txt")
    
    # 3. Write the results to the file
    with open(results_path, 'w') as f:
        f.write("--- Final Test Metrics ---\n")
        # trainer.test() returns a list of dictionaries, we take the first one
        for key, value in test_results[0].items():
            f.write(f"{key}: {value}\n")
            

    print (test_results)            
    print(f"✅ Test results saved to: {results_path}")
    

# %%

if __name__ == '__main__':    
    pl.seed_everything(42)
    config = load_config('./configs/xception_leg.yaml')
    data_module = ImageDataModule(config)

    # 1. Explicitly call setup() to calculate class weights
    # This is needed because we require the weights before initializing the model
    data_module.setup('fit')

    # 2. Add the calculated weights to the config dictionary
    # It will be None if use_class_weights was False
    config['class_weights'] = data_module.class_weights
    
    # 3. Now initialize the model with the updated config
    model = ImageClassifier(config)

    #print (model)

# %%

if __name__ == '__main__':
    main(config, data_module, model)

