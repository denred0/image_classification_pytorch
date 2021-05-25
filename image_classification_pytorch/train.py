from image_classification_pytorch.datamodule import ICPDataModule
from image_classification_pytorch.model import ICPModel
from image_classification_pytorch.inference import ICPInference
from image_classification_pytorch.optimizers import Adam
from image_classification_pytorch.schedulers import ExponentialLR

from pathlib import Path

# lightning related imports
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import datetime


class ICPTrainer():
    def __init__(self,
                 models=[],
                 data_dir='data',
                 images_ext='jpg',
                 init_lr=1e-5,
                 max_epochs=500,
                 augment_p=0.7,
                 progress_bar_refresh_rate=10,
                 early_stop_patience=6,
                 optimizer=Adam(),
                 scheduler=ExponentialLR()):
        super().__init__()
        self.models = models
        self.data_dir = data_dir
        self.images_ext = images_ext
        self.init_lr = init_lr
        self.max_epochs = max_epochs
        self.augment_p = augment_p
        self.augment_p = augment_p
        self.progress_bar_refresh_rate = progress_bar_refresh_rate
        self.early_stop_patience = early_stop_patience
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.models_for_training = []
        for m in self.models:
            model_data = {'model': m}

            dm = ICPDataModule(data_dir=self.data_dir,
                               augment_p=self.augment_p,
                               images_ext=self.images_ext,
                               model_type=model_data['model']['model_type'],
                               batch_size=model_data['model']['batch_size'],
                               input_resize=model_data['model']['im_size'],
                               input_resize_test=model_data['model']['im_size_test'],
                               mean=model_data['model']['mean'],
                               std=model_data['model']['std'])

            # To access the x_dataloader we need to call prepare_data and setup.
            # dm.prepare_data()
            dm.setup()

            # Init our model
            model = ICPModel(model_type=model_data['model']['model_type'],
                             num_classes=dm.num_classes,
                             optimizer=self.optimizer,
                             scheduler=self.scheduler,
                             learning_rate=self.init_lr)

            # Initialize a trainer
            early_stop_callback = EarlyStopping(
                monitor='val_loss',
                patience=self.early_stop_patience,
                verbose=True,
                mode='min'
            )

            # logs for tensorboard
            experiment_name = model_data['model']['model_type']
            logger = TensorBoardLogger('tb_logs/', name=experiment_name)

            checkpoint_name = experiment_name + '_' + '_{epoch}_{val_loss:.3f}_{val_acc:.3f}_{val_f1_epoch:.3f}'

            checkpoint_callback_loss = ModelCheckpoint(monitor='val_loss',
                                                       mode='min',
                                                       filename=checkpoint_name,
                                                       verbose=True,
                                                       save_top_k=1,
                                                       save_last=False)

            checkpoint_callback_acc = ModelCheckpoint(monitor='val_acc',
                                                      mode='max',
                                                      filename=checkpoint_name,
                                                      verbose=True,
                                                      save_top_k=1,
                                                      save_last=False)

            checkpoints = [checkpoint_callback_acc, checkpoint_callback_loss, early_stop_callback]
            callbacks = checkpoints

            trainer = pl.Trainer(max_epochs=self.max_epochs,
                                 progress_bar_refresh_rate=self.progress_bar_refresh_rate,
                                 gpus=1,
                                 logger=logger,
                                 callbacks=callbacks)

            model_data['icp_datamodule'] = dm
            model_data['icp_model'] = model
            model_data['icp_trainer'] = trainer

            self.models_for_training.append(model_data)

    def fit_test(self):

        for model in self.models_for_training:
            print('##################### START Training ' + model['model']['model_type'] + '... #####################')

            # Train the model ⚡g🚅⚡
            model['icp_trainer'].fit(model['icp_model'], model['icp_datamodule'])

            # Evaluate the model on the held out test set ⚡⚡
            results = model['icp_trainer'].test()[0]

            # save test results
            best_checkpoint = 'best_checkpoint: ' + model['icp_trainer'].checkpoint_callback.best_model_path
            results['best_checkpoint'] = best_checkpoint

            filename = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '__test_acc_' + str(
                round(results.get('test_acc'), 4)) + '.txt'

            path = 'test_logs/' + model['model']['model_type']
            Path(path).mkdir(parents=True, exist_ok=True)

            with open(path + '/' + filename, 'w+') as f:
                print(results, file=f)

            print('##################### END Training ' + model['model']['model_type'] + '... #####################')


def main():
    ens_adv_inception_resnet_v2 = {'model_type': 'ens_adv_inception_resnet_v2',
                                   'im_size': 256,
                                   'im_size_test': 256,
                                   'batch_size': 8,
                                   'mean': [0.5, 0.5, 0.5],
                                   'std': [0.5, 0.5, 0.5]}

    models = [ens_adv_inception_resnet_v2]

    trainer = ICPTrainer(models=models, data_dir='data_simpsons')
    trainer.fit_test()


def inference():
    ICPInference(data_dir='inference',
                 img_size=380,
                 show_accuracy=True,
                 checkpoint='tb_logs/tf_efficientnet_b4_ns/version_4/checkpoints/tf_efficientnet_b4_ns__epoch=2_val_loss=0.922_val_acc=0.830_val_f1_epoch=0.000.ckpt',
                 std=[0.229, 0.224, 0.225],
                 mean=[0.485, 0.456, 0.406],
                 confidence_threshold=1).predict()


if __name__ == '__main__':
    main()
    # inference()
