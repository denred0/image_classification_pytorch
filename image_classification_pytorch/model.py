import torch
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

import torchmetrics

import timm
import pretrainedmodels

from image_classification_pytorch.dict import *


class ICPModel(pl.LightningModule):
    def __init__(self,
                 model_type,
                 num_classes,
                 optimizer,
                 scheduler,
                 classes_weights,
                 learning_rate=0.0001):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_type = model_type

        self.optimizers = optimizer
        self.schedulers = scheduler
        self.classes_weigts = classes_weights
        print('classes_weights', classes_weights)
        self.classes_weigts = torch.FloatTensor(self.classes_weigts).cuda()

        self.loss_func = nn.CrossEntropyLoss(weight=self.classes_weigts)
        self.f1 = torchmetrics.F1(num_classes=self.num_classes)

        # load network
        if self.model_type in ['densenet121',  # classifier
                               'densenet161',
                               'densenet169',
                               'densenet201',
                               'densenetblur121d',
                               'dpn68',
                               'dpn68b',
                               'dpn92',
                               'dpn98',
                               'dpn107',
                               'dpn131',
                               'efficientnet_b0',
                               'efficientnet_b1',
                               'efficientnet_b1_pruned',
                               'efficientnet_b2',
                               'efficientnet_b2a',
                               'efficientnet_b3',
                               'efficientnet_b3_pruned',
                               'efficientnet_b3a',
                               'efficientnet_em',
                               'efficientnet_es',
                               'efficientnet_lite0',
                               'fbnetc_100',
                               'hrnet_w18',
                               'hrnet_w18_small',
                               'hrnet_w18_small_v2',
                               'hrnet_w30',
                               'hrnet_w32',
                               'hrnet_w40',
                               'hrnet_w44',
                               'hrnet_w48',
                               'hrnet_w64',
                               'mixnet_l',
                               'mixnet_m',
                               'mixnet_s',
                               'mixnet_xl',
                               'mnasnet_100',
                               'mobilenetv2_100',
                               'mobilenetv2_110d',
                               'mobilenetv2_120d',
                               'mobilenetv2_140',
                               'mobilenetv3_large_100',
                               'mobilenetv3_rw',
                               'semnasnet_100',
                               'spnasnet_100',
                               'tf_efficientnet_b0',
                               'tf_efficientnet_b0_ap',
                               'tf_efficientnet_b0_ns',
                               'tf_efficientnet_b1',
                               'tf_efficientnet_b1_ap',
                               'tf_efficientnet_b1_ns',
                               'tf_efficientnet_b2',
                               'tf_efficientnet_b2_ap',
                               'tf_efficientnet_b2_ns',
                               'tf_efficientnet_b3',
                               'tf_efficientnet_b3_ap',
                               'tf_efficientnet_b3_ns',
                               'tf_efficientnet_b4',
                               'tf_efficientnet_b4_ap',
                               'tf_efficientnet_b4_ns',
                               'tf_efficientnet_b5',
                               'tf_efficientnet_b5_ap',
                               'tf_efficientnet_b5_ns',
                               'tf_efficientnet_b6',
                               'tf_efficientnet_b6_ap',
                               'tf_efficientnet_b6_ns',
                               'tf_efficientnet_b7',
                               'tf_efficientnet_b7_ap',
                               'tf_efficientnet_b7_ns',
                               'tf_efficientnet_b8',
                               'tf_efficientnet_b8_ap',
                               'tf_efficientnet_cc_b0_4e',
                               'tf_efficientnet_cc_b0_8e',
                               'tf_efficientnet_cc_b1_8e',
                               'tf_efficientnet_el',
                               'tf_efficientnet_em',
                               'tf_efficientnet_es',
                               'tf_efficientnet_l2_ns',
                               'tf_efficientnet_l2_ns_475',
                               'tf_efficientnet_lite0',
                               'tf_efficientnet_lite1',
                               'tf_efficientnet_lite2',
                               'tf_efficientnet_lite3',
                               'tf_efficientnet_lite4',
                               'tf_mixnet_l',
                               'tf_mixnet_m',
                               'tf_mixnet_s',
                               'tf_mobilenetv3_large_075',
                               'tf_mobilenetv3_large_100',
                               'tf_mobilenetv3_large_minimal_100',
                               'tf_mobilenetv3_small_075',
                               'tf_mobilenetv3_small_100',
                               'tf_mobilenetv3_small_minimal_100',
                               'tv_densenet121', ]:
            model = timm.create_model(model_type, pretrained=True)
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, self.num_classes)
            self.model = model

        elif self.model_type in ['adv_inception_v3',  # fc
                                 'dla34',
                                 'dla46_c',
                                 'dla46x_c',
                                 'dla60',
                                 'dla60_res2net',
                                 'dla60_res2next',
                                 'dla60x',
                                 'dla60x_c',
                                 'dla102',
                                 'dla102x',
                                 'dla102x2',
                                 'dla169',
                                 'ecaresnet26t',
                                 'ecaresnet50d',
                                 'ecaresnet50d_pruned',
                                 'ecaresnet50t',
                                 'ecaresnet101d',
                                 'ecaresnet101d_pruned',
                                 'ecaresnet269d',
                                 'ecaresnetlight',
                                 'gluon_inception_v3',
                                 'gluon_resnet18_v1b',
                                 'gluon_resnet34_v1b',
                                 'gluon_resnet50_v1b',
                                 'gluon_resnet50_v1c',
                                 'gluon_resnet50_v1d',
                                 'gluon_resnet50_v1s',
                                 'gluon_resnet101_v1b',
                                 'gluon_resnet101_v1c',
                                 'gluon_resnet101_v1d',
                                 'gluon_resnet101_v1s',
                                 'gluon_resnet152_v1b',
                                 'gluon_resnet152_v1c',
                                 'gluon_resnet152_v1d',
                                 'gluon_resnet152_v1s',
                                 'gluon_resnext50_32x4d',
                                 'gluon_resnext101_32x4d',
                                 'gluon_resnext101_64x4d',
                                 'gluon_senet154',
                                 'gluon_seresnext50_32x4d',
                                 'gluon_seresnext101_32x4d',
                                 'gluon_seresnext101_64x4d',
                                 'gluon_xception65',
                                 'ig_resnext101_32x8d',
                                 'ig_resnext101_32x16d',
                                 'ig_resnext101_32x32d',
                                 'ig_resnext101_32x48d',
                                 'inception_v3',
                                 'res2net50_14w_8s',
                                 'res2net50_26w_4s',
                                 'res2net50_26w_6s',
                                 'res2net50_26w_8s',
                                 'res2net50_48w_2s',
                                 'res2net101_26w_4s',
                                 'res2next50',
                                 'resnest14d',
                                 'resnest26d',
                                 'resnest50d',
                                 'resnest50d_1s4x24d',
                                 'resnest50d_4s2x40d',
                                 'resnest101e',
                                 'resnest200e',
                                 'resnest269e',
                                 'resnet18',
                                 'resnet18d',
                                 'resnet26',
                                 'resnet26d',
                                 'resnet34',
                                 'resnet34d',
                                 'resnet50',
                                 'resnet50d',
                                 'resnet101d',
                                 'resnet152d',
                                 'resnet200d',
                                 'resnetblur50',
                                 'resnext50_32x4d',
                                 'resnext50d_32x4d',
                                 'resnext101_32x8d',
                                 'selecsls42b',
                                 'selecsls60',
                                 'selecsls60b',
                                 'seresnet50',
                                 'seresnet152d',
                                 'seresnext26d_32x4d',
                                 'seresnext26t_32x4d',
                                 'seresnext50_32x4d',
                                 'skresnet18',
                                 'skresnet34',
                                 'skresnext50_32x4d',
                                 'ssl_resnet18',
                                 'ssl_resnet50',
                                 'ssl_resnext50_32x4d',
                                 'ssl_resnext101_32x4d',
                                 'ssl_resnext101_32x8d',
                                 'ssl_resnext101_32x16d',
                                 'swsl_resnet18',
                                 'swsl_resnet50',
                                 'swsl_resnext50_32x4d',
                                 'swsl_resnext101_32x4d',
                                 'swsl_resnext101_32x8d',
                                 'swsl_resnext101_32x16d',
                                 'tf_inception_v3',
                                 'tv_resnet34',
                                 'tv_resnet50',
                                 'tv_resnet101',
                                 'tv_resnet152',
                                 'tv_resnext50_32x4d',
                                 'wide_resnet50_2',
                                 'wide_resnet101_2',
                                 'xception', ]:
            model = timm.create_model(model_type, pretrained=True)
            in_features = model.fc.in_features
            model.classifier = nn.Linear(in_features, self.num_classes)
            self.model = model
        elif self.model_type in ['cspdarknet53',  # head.fc
                                 'cspresnet50',
                                 'cspresnext50',
                                 'dm_nfnet_f0',
                                 'dm_nfnet_f1',
                                 'dm_nfnet_f2',
                                 'dm_nfnet_f3',
                                 'dm_nfnet_f4',
                                 'dm_nfnet_f5',
                                 'dm_nfnet_f6',
                                 'ese_vovnet19b_dw',
                                 'ese_vovnet39b',
                                 'gernet_l',
                                 'gernet_m',
                                 'gernet_s',
                                 'nf_regnet_b1',
                                 'nf_resnet50',
                                 'nfnet_l0c',
                                 'regnetx_002',
                                 'regnetx_004',
                                 'regnetx_006',
                                 'regnetx_008',
                                 'regnetx_016',
                                 'regnetx_032',
                                 'regnetx_040',
                                 'regnetx_064',
                                 'regnetx_080',
                                 'regnetx_120',
                                 'regnetx_160',
                                 'regnetx_320',
                                 'regnety_002',
                                 'regnety_004',
                                 'regnety_006',
                                 'regnety_008',
                                 'regnety_016',
                                 'regnety_032',
                                 'regnety_040',
                                 'regnety_064',
                                 'regnety_080',
                                 'regnety_120',
                                 'regnety_160',
                                 'regnety_320',
                                 'repvgg_a2',
                                 'repvgg_b0',
                                 'repvgg_b1',
                                 'repvgg_b1g4',
                                 'repvgg_b2',
                                 'repvgg_b2g4',
                                 'repvgg_b3',
                                 'repvgg_b3g4',
                                 'resnetv2_50x1_bitm',
                                 'resnetv2_50x1_bitm_in21k',
                                 'resnetv2_50x3_bitm',
                                 'resnetv2_50x3_bitm_in21k',
                                 'resnetv2_101x1_bitm',
                                 'resnetv2_101x1_bitm_in21k',
                                 'resnetv2_101x3_bitm',
                                 'resnetv2_101x3_bitm_in21k',
                                 'resnetv2_152x2_bitm',
                                 'resnetv2_152x2_bitm_in21k',
                                 'resnetv2_152x4_bitm',
                                 'resnetv2_152x4_bitm_in21k',
                                 'rexnet_100',
                                 'rexnet_130',
                                 'rexnet_150',
                                 'rexnet_200',
                                 'tresnet_l',
                                 'tresnet_l_448',
                                 'tresnet_m',
                                 'tresnet_m_448',
                                 'tresnet_xl',
                                 'tresnet_xl_448',
                                 'vgg11',
                                 'vgg11_bn',
                                 'vgg13',
                                 'vgg13_bn',
                                 'vgg16',
                                 'vgg16_bn',
                                 'vgg19',
                                 'vgg19_bn',
                                 'xception41',
                                 'xception65',
                                 'xception71', ]:
            model = timm.create_model(model_type, pretrained=True)
            in_features = model.head.fc.in_features
            model.classifier = nn.Linear(in_features, self.num_classes)
            self.model = model
        elif self.model_type in ['ens_adv_inception_resnet_v2',  # classif
                                 'inception_resnet_v2', ]:
            model = timm.create_model(model_type, pretrained=True)
            in_features = model.classif.in_features
            model.classifier = nn.Linear(in_features, self.num_classes)
            self.model = model
        elif self.model_type in ['inception_v4',  # last_linear
                                 'legacy_senet154',
                                 'legacy_seresnet18',
                                 'legacy_seresnet34',
                                 'legacy_seresnet50',
                                 'legacy_seresnet101',
                                 'legacy_seresnet152',
                                 'legacy_seresnext26_32x4d',
                                 'legacy_seresnext50_32x4d',
                                 'legacy_seresnext101_32x4d',
                                 'nasnetalarge',
                                 'pnasnet5large', ]:
            model = timm.create_model(model_type, pretrained=True)
            in_features = model.last_linear.in_features
            model.classifier = nn.Linear(in_features, self.num_classes)
            self.model = model
        elif self.model_type in ['vit_base_patch16_224',  # head
                                 'vit_base_patch16_224_in21k',
                                 'vit_base_patch16_384',
                                 'vit_base_patch32_224_in21k',
                                 'vit_base_patch32_384',
                                 'vit_base_resnet50_224_in21k',
                                 'vit_base_resnet50_384',
                                 'vit_deit_base_distilled_patch16_224',
                                 'vit_deit_base_distilled_patch16_384',
                                 'vit_deit_base_patch16_224',
                                 'vit_deit_base_patch16_384',
                                 'vit_deit_small_distilled_patch16_224',
                                 'vit_deit_small_patch16_224',
                                 'vit_deit_tiny_distilled_patch16_224',
                                 'vit_deit_tiny_patch16_224',
                                 'vit_large_patch16_224',
                                 'vit_large_patch16_224_in21k',
                                 'vit_large_patch16_384',
                                 'vit_large_patch32_224_in21k',
                                 'vit_large_patch32_384',
                                 'vit_small_patch16_224', ]:
            model = timm.create_model(model_type, pretrained=True)
            in_features = model.head.in_features
            model.classifier = nn.Linear(in_features, self.num_classes)
            self.model = model
        elif self.model_type in ['senet154']:
            model = pretrainedmodels.__dict__[model_type](num_classes=1000, pretrained='imagenet')
            model.eval()
            num_features = model.last_linear.in_features
            # Заменяем Fully-Connected слой на наш линейный классификатор
            model.last_linear = nn.Linear(num_features, self.num_classes)
            self.model = model
        else:
            assert (
                False
            ), f"model_type '{self.model_type}' not implemented. Please, choose from {MODELS}"

    def loss(self, logits, labels):
        return self.loss_func(input=logits, target=labels.long())

    # will be used during inference
    def forward(self, x):
        return self.model(x)

    # Using custom or multiple metrics (default_hp_metric=False)
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        train_loss = self.loss(output, y)

        # training metrics
        output = torch.argmax(output, dim=1)

        acc = accuracy(output, y)

        self.log('train_loss', train_loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)

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
            ), f"Optimizer '{self.optimizer}' not implemented. Please, choose from {OPTIMIZERS}"

        if type(self.scheduler).__name__ == 'ExponentialLR':
            gen_sched = {'scheduler': torch.optim.lr_scheduler.ExponentialLR(gen_opt,
                                                                             gamma=self.scheduler.gamma,
                                                                             last_epoch=self.scheduler.last_epoch,
                                                                             verbose=self.scheduler.verbose),
                         'interval': self.scheduler.interval}
        else:
            assert (
                False
            ), f"Scheduler '{self.scheduler}' not implemented. Please, choose from {SCHEDULERS}"

        return [gen_opt], [gen_sched]
