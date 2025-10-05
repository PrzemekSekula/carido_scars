"""
This module contains the main function for training and testing the model.
Autor: Przemek Sekula
Created: 2025-10-04
Last modified: 2025-10-04
"""

# %%

import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from data_tools import ImageDataModule
from model import ImageClassifier
from tools import parse_with_config_file, log_args_to_tensorboard


# %%

def main(args):


    logger = TensorBoardLogger(
        os.path.join(args.log_dir, args.subdir_name), 
        name=args.model_name
        )
    log_args_to_tensorboard(logger, args)

    data_module = ImageDataModule(args)

    # 1. Explicitly call setup() to calculate class weights
    # This is needed because we require the weights before initializing the model
    data_module.setup('fit')

    # 2. Add the calculated weights to the config dictionary
    # It will be None if use_class_weights was False
    args.class_weights = data_module.class_weights
    
    # 3. Now initialize the model with the updated config
    model = ImageClassifier(args)


    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc', 
        dirpath=os.path.join(args.checkpoint_dir, args.subdir_name), 
        filename=f'{args.model_name}-best-checkpoint', 
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
        max_epochs=args.num_epochs, 
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
            

    print (test_results)            
    print(f"✅ Test results saved to: {results_path}")
    

# %%

if __name__ == '__main__':    
    pl.seed_everything(42)
    parser = argparse.ArgumentParser(
        description='Trains the model. Defaults are loaded from configs.yaml, '
        'from the defaults section. to Change the config, use the --configs argument.'
    )
    parser.add_argument("--configs", nargs="+", help="List of named configs from configs.yaml to load.")
    main(parse_with_config_file(parser, defaults_name="defaults")) 
