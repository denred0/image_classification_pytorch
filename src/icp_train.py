from src.icp_datamodule import ICPDataModule
from src.icp_model import ICPModel


from pathlib import Path

# lightning related imports
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import datetime


def main():
    # models
    vgg16_bn = {'model_type': 'vgg16_bn', 'use_normalize': True, 'half_normalize': False, 'im_size': 224,
                'im_size_test': 224, 'batch_size': 16}

    senet154 = {'model_type': 'senet154', 'use_normalize': True, 'half_normalize': False, 'im_size': 224,
                'im_size_test': 224, 'batch_size': 16}

    inceptionv4 = {'model_type': 'inceptionv4', 'use_normalize': True, 'half_normalize': False, 'im_size': 299,
                   'im_size_test': 299, 'batch_size': 16}

    efficientnet_b3 = {'model_type': 'efficientnet-b3', 'use_normalize': True, 'half_normalize': False, 'im_size': 512,
                       'im_size_test': 512, 'batch_size': 8}

    efficientnet_b4 = {'model_type': 'efficientnet-b4', 'use_normalize': True, 'half_normalize': False, 'im_size': 512,
                       'im_size_test': 512, 'batch_size': 6}

    efficientnet_b6 = {'model_type': 'efficientnet-b6', 'use_normalize': True, 'half_normalize': False, 'im_size': 512,
                       'im_size_test': 512, 'batch_size': 4}

    dm_nfnet_f4 = {'model_type': 'dm_nfnet_f4', 'use_normalize': True, 'half_normalize': False, 'im_size': 384,
                   'im_size_test': 512, 'batch_size': 1}

    vit_base_patch16_384 = {'model_type': 'vit_base_patch16_384', 'use_normalize': True, 'half_normalize': True,
                            'im_size': 384,
                            'im_size_test': 384, 'batch_size': 8}

    tf_efficientnet_b3_ns = {'model_type': 'tf_efficientnet_b3_ns', 'use_normalize': True, 'half_normalize': False,
                             'im_size': 300,
                             'im_size_test': 300, 'batch_size': 16}

    tf_efficientnet_b4_ns = {'model_type': 'tf_efficientnet_b4_ns', 'use_normalize': True, 'half_normalize': False,
                             'im_size': 380,
                             'im_size_test': 380, 'batch_size': 8}

    tf_efficientnet_b5_ns = {'model_type': 'tf_efficientnet_b5_ns', 'use_normalize': True, 'half_normalize': False,
                             'im_size': 456,
                             'im_size_test': 456, 'batch_size': 6}

    tf_efficientnet_b6_ns = {'model_type': 'tf_efficientnet_b6_ns', 'use_normalize': True, 'half_normalize': False,
                             'im_size': 528,
                             'im_size_test': 528, 'batch_size': 3}

    tf_efficientnet_b7_ns = {'model_type': 'tf_efficientnet_b7_ns', 'use_normalize': True, 'half_normalize': False,
                             'im_size': 600,
                             'im_size_test': 600, 'batch_size': 2}

    models = [tf_efficientnet_b4_ns]

    # test models cuda memory
    # models.append(tf_efficientnet_b7_ns)

    #############################################################

    data_dir = Path('data_simpsons')
    images_ext = 'jpg'

    # train parameters
    init_lr = 1e-5
    max_epochs = 500
    augment_p = 0.7
    progress_bar_refresh_rate = 10
    early_stop_patience = 6

    # Init our data pipeline

    for m in models:
        print('####################### START Training ' + m['model_type'] + '... #######################')

        model_type = m['model_type']
        batch_size = m['batch_size']
        im_size = m['im_size']
        im_size_test = m['im_size_test']
        use_normalize = m['use_normalize']
        half_normalize = m['half_normalize']

        dm = ICPDataModule(model_type=model_type, batch_size=batch_size, data_dir=data_dir, input_resize=im_size,
                             input_resize_test=im_size_test,
                             use_normalize=use_normalize, half_normalize=half_normalize, augment_p=augment_p,
                             images_ext=images_ext)

        # To access the x_dataloader we need to call prepare_data and setup.
        # dm.prepare_data()
        dm.setup()

        # Init our model
        model = ICPModel(model_type, dm.num_classes, learning_rate=init_lr)

        # Initialize a trainer
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=early_stop_patience,
            verbose=True,
            mode='min'
        )

        # logs for tensorboard
        experiment_name = model_type
        logger = TensorBoardLogger('tb_logs/', name=experiment_name)

        checkpoint_name = experiment_name + '_' + '_{epoch}_{val_loss:.3f}_{val_acc:.3f}_{val_f1_epoch:.3f}'

        checkpoint_callback_loss = ModelCheckpoint(monitor='val_loss', mode='min',
                                                   filename=checkpoint_name,
                                                   verbose=True, save_top_k=1,
                                                   save_last=False)
        checkpoint_callback_acc = ModelCheckpoint(monitor='val_acc', mode='max',
                                                  filename=checkpoint_name,
                                                  verbose=True, save_top_k=1,
                                                  save_last=False)

        # checkpoint_callback_f1 = ModelCheckpoint(monitor='val_f1_epoch', mode='max',
        #                                          filename=checkpoint_name,
        #                                          verbose=True, save_top_k=1,
        #                                          save_last=True)

        checkpoints = [checkpoint_callback_acc, checkpoint_callback_loss, early_stop_callback]
        callbacks = checkpoints

        trainer = pl.Trainer(max_epochs=max_epochs,
                             progress_bar_refresh_rate=progress_bar_refresh_rate,
                             # gpus=1,
                             auto_select_gpus=True,
                             logger=logger,
                             callbacks=callbacks)


        # Train the model ⚡🚅⚡
        trainer.fit(model, dm)

        # Evaluate the model on the held out test set ⚡⚡
        results = trainer.test()[0]

        # save test results
        best_checkpoint = 'best_checkpoint: ' + trainer.checkpoint_callback.best_model_path
        results['best_checkpoint'] = best_checkpoint

        filename = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '__test_acc_' + str(
            round(results.get('test_acc'), 4)) + '.txt'

        path = 'test_logs/' + model_type
        Path(path).mkdir(parents=True, exist_ok=True)

        with open(path + '/' + filename, 'w+') as f:
            print(results, file=f)

        print('####################### END Training ' + m['model_type'] + '... #######################')


if __name__ == '__main__':
    main()
