import torch
from torch import nn

import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

import pretrainedmodels
import torchmetrics

# for efficient model transfer learning
from efficientnet_pytorch import EfficientNet
import timm



class ICPModel(pl.LightningModule):
    def __init__(self, model_type, num_classes, confmat, learning_rate=0.0001):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.confmat = confmat

        conf_matrix = np.zeros((3, 3))
        self.conf_matrix = conf_matrix

        self.f1 = torchmetrics.F1(num_classes=num_classes)

        # load network
        if model_type in ["inceptionv4", "vgg16_bn", 'senet154']:
            model = pretrainedmodels.__dict__[model_type](num_classes=1000, pretrained='imagenet')
            model.eval()
            num_features = model.last_linear.in_features
            # Заменяем Fully-Connected слой на наш линейный классификатор
            model.last_linear = nn.Linear(num_features, self.num_classes)
            self.model = model
        elif model_type in ['efficientnet-b3']:
            model = EfficientNet.from_pretrained('efficientnet-b3')
            # model = EfficientNet.from_pretrained('efficientnet-b3')
            # model.load_state_dict(
            #     torch.load('resources-for-google-landmark-recognition-2020/efficientnet-b3-5fb5a3c3.pth'))
            in_features = model._fc.in_features
            model._fc = nn.Linear(in_features, self.num_classes)
            self.model = model
        elif model_type in ['efficientnet-b4']:
            model = EfficientNet.from_name('efficientnet-b4')
            model.load_state_dict(
                torch.load('resources-for-google-landmark-recognition-2020/efficientnet-b4-6ed6700e.pth'))
            in_features = model._fc.in_features
            model._fc = nn.Linear(in_features, self.num_classes)
            self.model = model
        elif model_type in ['efficientnet-b6']:
            model = EfficientNet.from_pretrained('efficientnet-b6')
            in_features = model._fc.in_features
            model._fc = nn.Linear(in_features, self.num_classes)
            self.model = model
        elif model_type in ['dm_nfnet_f4', 'dm_nfnet_f5', 'dm_nfnet_f6']:
            model = timm.create_model(model_type, pretrained=True)
            in_features = model.head.fc.in_features
            model.head.fc = nn.Linear(in_features, self.num_classes)
            self.model = model
        elif model_type in ['vit_base_patch16_384']:
            model = timm.create_model(model_type, pretrained=True)
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, self.num_classes)
            self.model = model
        elif model_type in ['tf_efficientnet_b0_ns', 'tf_efficientnet_b3_ns', 'tf_efficientnet_b4_ns',
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
        if batch_idx == 0:
            self.f1.reset()

        x, y = batch
        output = self.forward(x)
        train_loss = self.loss(output, y)

        # training metrics
        output = torch.argmax(output, dim=1)
        acc = accuracy(output, y)
        f1 = self.f1(output, y)

        self.log('train_loss', train_loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        self.log('train_f1_step', f1)

        return train_loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        val_loss = self.loss(output, y)

        # validation metrics
        output = torch.argmax(output, dim=1)
        acc = accuracy(output, y)
        f1 = self.f1(output, y)

        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_f1_step', f1)

        return val_loss

    # # logic for a single testing step
    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        test_loss = self.loss(output, y)

        # validation metrics
        output = torch.argmax(output, dim=1)
        acc = accuracy(output, y)
        f1 = self.f1(output, y)

        # tp, fp, tn, fn = _stat_scores_update(
        #     preds,
        #     y,
        #     num_classes=self.num_classes
        # )

        self.log('test_loss', test_loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        self.log('test_f1_step', f1, prog_bar=True)

        # t = self.confmat(output, y).detach().cpu().numpy()
        # self.conf_matrix = np.add(self.conf_matrix, t)
        # if (batch_idx == 14):
        #     print()
        #     print(self.conf_matrix)

        return test_loss

    def training_epoch_end(self, outputs):
        self.log('train_f1_epoch', self.f1.compute())
        self.f1.reset()

    def validation_epoch_end(self, outputs):
        self.log('val_f1_epoch', self.f1.compute(), prog_bar=True)
        self.f1.reset()

    def test_epoch_end(self, outputs):
        self.log('test_f1_epoch', self.f1.compute())
        self.f1.reset()

    def configure_optimizers(self):
        gen_opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=2e-5)

        # gen_sched = torch.optim.lr_scheduler.StepLR(gen_opt, step_size=1, gamma=0.999) #decay LR by a factor of 0.999 every 1 epoch

        gen_sched = {'scheduler':
                         torch.optim.lr_scheduler.ExponentialLR(gen_opt, gamma=0.999, verbose=False),
                     'interval': 'step'}  # called after each training step

        # dis_sched = torch.optim.lr_scheduler.CosineAnnealingLR(gen_opt,
        #                                                        T_max=10)  # called every epoch,
        # lower the learning rate to its minimum in each epoch and then restart from the base learning rate
        return [gen_opt], [gen_sched]

    # def configure_optimizers(self):
    #     optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=2e-5)
    #     lr_scheduler = PolynomialLRDecay(optimizer, max_decay_steps=50,
    #                                      end_learning_rate=1e-6,
    #                                      power=0.9, verbose=True)
    #
    #     # optimizer = MADGRAD(self.parameters(), lr=self.lr, weight_decay=2.5e-5)
    #     # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     #     optimizer, milestones=[30, 60, 90], gamma=0.1, verbose=True)
    #
    #     # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     #     optimizer=optimizer,
    #     #     factor=0.5,
    #     #     threshold=0.01,
    #     #     threshold_mode='rel',
    #     #     cooldown=3,
    #     #     mode='max',
    #     #     min_lr=1e-6,
    #     #     verbose=True,
    #     #     )
    #
    #     lr_dict = {
    #         'scheduler': lr_scheduler,
    #         'reduce_on_plateau': False,
    #         'monitor': 'val_f1_epoch',
    #         'interval': 'epoch',
    #         'frequency': 1,
    #     }
    #
    #     return [optimizer], [lr_dict]
