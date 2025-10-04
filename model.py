"""
This module contains the model definition with PyTorch LightningModule.
Autor: Przemek Sekula
Created: 2025-10-04
Last modified: 2025-10-04
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm
import torchmetrics

class ImageClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        
        self.backbone = timm.create_model(self.hparams.model_name, pretrained=True, num_classes=0)
        
        # Start with all layers frozen
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        num_features = self.backbone.num_features
        self.classifier = nn.Linear(num_features, self.hparams.num_classes)
        # Get weights from hparams. It will be None if not provided.

        class_weights = self.hparams.get('class_weights', None)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)        
        
        # Use 'multiclass' task which works for binary as well
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.precision_metric = torchmetrics.Precision(task="multiclass", num_classes=self.hparams.num_classes, average='macro')
        self.recall_metric = torchmetrics.Recall(task="multiclass", num_classes=self.hparams.num_classes, average='macro')
        self.f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_classes, average='macro')
        
        
        self.is_finetuning = False

    def _unfreeze_last_layers(self):
        """Helper function to unfreeze the last N layers of the backbone."""
        layers_to_unfreeze = list(self.backbone.children())[-self.hparams.unfreeze_layers:]
        
        print(f"Unfreezing {len(layers_to_unfreeze)} layer blocks of the backbone...")
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True

    def on_train_epoch_start(self):
        """Lightning hook to trigger fine-tuning phase."""
        if self.hparams.do_finetune and self.current_epoch == self.hparams.warmup_epochs:
            self.is_finetuning = True
            self._unfreeze_last_layers()
            self.trainer.strategy.setup_optimizers(self.trainer)
            print(f"--- Switched to Fine-Tuning phase at epoch {self.current_epoch} ---")

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def _common_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch, batch_idx)
        acc = self.accuracy(preds, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch, batch_idx)
        acc = self.accuracy(preds, labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)


    def test_step(self, batch, batch_idx):
        # In the test step, update the state of all metrics
        loss, preds, labels = self._common_step(batch, batch_idx)
        self.accuracy.update(preds, labels)
        self.precision_metric.update(preds, labels)
        self.recall_metric.update(preds, labels)
        self.f1_metric.update(preds, labels)
        self.log('test_loss', loss)

    def on_test_epoch_end(self):
        # At the end of the test epoch, compute and log the final metrics
        print("\n--- Final Test Metrics ---")
        self.log("test_acc", self.accuracy.compute())
        self.log("test_precision", self.precision_metric.compute())
        self.log("test_recall (macro)", self.recall_metric.compute())
        self.log("test_f1_score (macro)", self.f1_metric.compute())
        self.log("test_balanced_accuracy", self.recall_metric.compute()) # Macro recall is balanced accuracy
        
        # Manually print the results to the console
        print(f"Accuracy: {self.accuracy.compute():.4f}")
        print(f"Balanced Accuracy: {self.recall_metric.compute():.4f}")
        print(f"Precision (Macro): {self.precision_metric.compute():.4f}")
        print(f"Recall (Macro): {self.recall_metric.compute():.4f}")
        print(f"F1-Score (Macro): {self.f1_metric.compute():.4f}")
    

    def configure_optimizers(self):
        if not self.is_finetuning:
            print(f"--- Setting up Warm-up optimizer (LR: {self.hparams.learning_rate}) ---")
            optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.hparams.learning_rate)
        else:
            print(f"--- Setting up Fine-Tuning optimizer (Head LR: {self.hparams.learning_rate}, Backbone LR: {self.hparams.finetune_lr}) ---")
            optimizer = torch.optim.Adam([
                {'params': self.classifier.parameters(), 'lr': self.hparams.learning_rate},
                {'params': [p for p in self.backbone.parameters() if p.requires_grad], 'lr': self.hparams.finetune_lr}
            ])
        return optimizer