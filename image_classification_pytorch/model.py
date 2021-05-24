import torch
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

import torchmetrics

import timm


class ICPModel(pl.LightningModule):
    def __init__(self,
                 model_type,
                 num_classes,
                 optimizer,
                 scheduler,
                 learning_rate=0.0001):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.optimizers = ['Adam()']
        self.schedulers = ['ExponentialLR()']

        # load network
        if model_type in ['tf_efficientnet_b0_ns', 'tf_efficientnet_b3_ns', 'tf_efficientnet_b4_ns',
                          'tf_efficientnet_b5_ns', 'tf_efficientnet_b6_ns',
                          'tf_efficientnet_b7_ns']:  # noisy_students
            model = timm.create_model(model_type, pretrained=True)
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, self.num_classes)
            self.model = model
        else:
            assert (
                False
            ), f"model_type '{model_type}' not implemented, use: --model_type [imagenet]"

        self.loss_func = nn.CrossEntropyLoss()
        self.f1 = torchmetrics.F1(num_classes=self.num_classes)

    def loss(self, logits, labels):
        return self.loss_func(input=logits, target=labels)

    # will be used during inference
    def forward(self, x):
        return self.model(x)

    # logic for a single training step
    def training_step(self, batch, batch_idx):

        x, y = batch
        output = self.forward(x)
        train_loss = self.loss(output, y)

        # training metrics
        output = torch.argmax(output, dim=1)
        acc = accuracy(output, y)

        self.log('train_loss', train_loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)

        return train_loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        val_loss = self.loss(output, y)

        # validation metrics
        output = torch.argmax(output, dim=1)
        acc = accuracy(output, y)

        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        return val_loss

    # # logic for a single testing step
    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        test_loss = self.loss(output, y)

        # validation metrics
        output = torch.argmax(output, dim=1)
        acc = accuracy(output, y)

        self.log('test_loss', test_loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)

        return test_loss

    # def training_epoch_end(self, outputs):
    #     self.log('train_f1_epoch', self.f1.compute())
    #     self.f1.reset()
    #
    # def validation_epoch_end(self, outputs):
    #     self.log('val_f1_epoch', self.f1.compute(), prog_bar=True)
    #     self.f1.reset()
    #
    # def test_epoch_end(self, outputs):
    #     self.log('test_f1_epoch', self.f1.compute())
    #     self.f1.reset()

    def configure_optimizers(self):

        if type(self.optimizer).__name__ == 'Adam':
            gen_opt = torch.optim.Adam(self.parameters(),
                                       lr=self.learning_rate,
                                       betas=self.optimizer.betas,
                                       eps=self.optimizer.eps,
                                       weight_decay=self.optimizer.weight_decay,
                                       amsgrad=self.optimizer.amsgrad)
        else:
            assert (
                False
            ), f"Optimizer '{self.optimizer}' not recognized. Please, choose from {self.optimizers}"

        if type(self.scheduler).__name__ == 'ExponentialLR':
            gen_sched = {'scheduler': torch.optim.lr_scheduler.ExponentialLR(gen_opt,
                                                                             gamma=self.scheduler.gamma,
                                                                             last_epoch=self.scheduler.last_epoch,
                                                                             verbose=self.scheduler.verbose),
                         'interval': self.scheduler.interval}
        else:
            assert (
                False
            ), f"Scheduler '{self.scheduler}' not recognized. Please, choose from {self.schedulers}"

        return [gen_opt], [gen_sched]
