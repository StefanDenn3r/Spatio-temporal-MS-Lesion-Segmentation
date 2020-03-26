import os

CONFIG = {
    "name": f"{os.path.basename(__file__).split('.')[0]}",
    "n_gpu": 1,

    "arch": {
        "type": "FCDenseNet",
        "args": {
            "in_channels": 4  # 1  # 4
        }
    },
    "dataset": {
        "type": "ISBIDatasetStatic",
        "args": {
            "data_dir": "../ISBIMSlesionChallenge/",
            "preprocess": True,
            "modalities": ['flair', 'mprage', 'pd', 't2']
        }
    },
    "data_loader": {
        "type": "ISBIDataloader",
        "args": {
            "batch_size": 2,
            "shuffle": True,
            "num_workers": 4,
        }
    },
    "optimizer": {
        "type": "Adam",
        "args": {
            "lr": 0.0001,
            "weight_decay": 0,
            "amsgrad": True
        }
    },
    "loss": "mse",
    "metrics": [
        "precision", "recall", "dice_loss", "dice_score", "asymmetric_loss"
    ],
    "lr_scheduler": {
        "type": "StepLR",
        "args": {
            "step_size": 50,
            "gamma": 0.1
        }
    },
    "trainer": {
        "type": "StaticTrainer",
        "epochs": 100,
        "save_dir": "../saved/",
        "save_period": 1,
        "verbosity": 2,
        "monitor": "min val_loss",
        "early_stop": 10,
        "tensorboard": True
    }
}
