# Spatio-temporal Learning from Longitudinal Data for Multiple Sclerosis Lesion Segmentation

This is the code for our paper Spatio-temporal Learning from Longitudinal Data for Multiple Sclerosis Lesion Segmentation which can be found [here](https://arxiv.org/pdf/2004.03675.pdf)

If you use any of our code, please cite:
```
@article{Denner2020,
   author = {Denner, Stefan and Khakzar, Ashkan and Sajid, Moiz and Saleh, Mahdi and Spiclin, Ziga and Kim, Seong Tae and Navab, Nassir},
   title = {Spatio-temporal Learning from Longitudinal Data for Multiple Sclerosis Lesion Segmentation},
   url = {http://arxiv.org/abs/2004.03675},
   year = {2020}
}
``` 

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->
* [Spatio-temporal Learning from Longitudinal Data for Multiple Sclerosis Lesion Segmentation](#spatio-temporal-learning-from-longitudinal-data-for-multiple-sclerosis-lesion-segmentation)
    * [Requirements](#requirements)
    * [Folder Structure](#folder-structure)
    * [Usage](#usage)
        * [Train](#train)
        * [Resuming from checkpoints](#resuming-from-checkpoints)
        * [Test](#test)
    * [Disclaimer](#disclaimer)
    * [License](#license)
    * [Acknowledgements](#acknowledgements)

<!-- /code_chunk_output -->

## Requirements
* Python >= 3.5 (3.6 recommended)
* PyTorch = 1.4 
* tqdm (Optional for `test.py`)
* tensorboard >= 1.14 
* nibabel >= 2.5

## Folder Structure
  ```
  Spatio-temporal-MS-Lesion-Segmentation/
  │
  ├── train.py - main script to start/resume training
  ├── test.py - evaluation of trained model
  ├── test_single_view.py - evaluation of models which use a single model for each plane orientation
  │
  ├── base/ - abstract base classes
  │  
  ├── configs/ - holds all the configurations files for the different models
  │   ├── Longitudinal_Network.py
  │   ├── Longitudinal_Network_with_Pretraining_Finetune.py
  │   ├── Longitudinal_Network_with_Pretraining_Pretrain.py
  │   ├── Longitudinal_Siamese_Network.py
  │   ├── Multitask_Longitudinal_Network.py
  │   ├── Static_Network.py
  │   └── Static_Network_Zhang.py
  │
  ├── data_loader/
  │   └── ISBIDataloader.py - dataloader for the ISBI Dataset
  │
  ├── model/
  │   ├── utils/ - holds additional Modules, losses and metrics
  │   ├── FCDenseNet.py
  │   ├── LongitudinalFCDenseNet.py
  │   ├── MultitaskNetwork.py.py
  │   └── Voxelmorph2DTransfer.py
  │
  └── trainer/ - trainers
      ├── ISBITrainer.py
      ├── LongitudinalDeformationTrainer.py
      ├── LongitudinalMultitaskTrainer.py
      ├── LongitudinalTrainer.py
      ├── LongitudinalTrainerFinetune.py
      └── StaticTrainer.py

  ```

## Usage
Before the models can be trained or tested, the paths in the config files (located in `configs/`) have to be adjusted:
- `data_loader.args.data_dir` specifies where the data is located
-  `trainer.save_dir` specifies where to store the model checkpoints and logs.

### Train
To run the experiments from our paper the following table specifies the commands to run:

| Network                                   | Command                                                              |
|-------------------------------------------|----------------------------------------------------------------------|
| Multitask Longitudinal Network            | python train.py -c Multitask_Longitudinal_Network.py                 |
| Longitudinal Network with Pretraining     | python train.py -c Longitudinal_Network_with_Pretraining_Pretrain.py |
| Longitudinal Network                      | python train.py -c Longitudinal_Network.py                           |
| Static Network                            | python train.py -c Static_Network.py                                 |
| Longitudinal Siamese Network              | python train.py -c Longitudinal_Siamese_Network.py                   |
| Static Network (Zhang et al. [2])         | python train.py -c Static_Network_Zhang.py -s True                   |
| Static Network (Single plane orientation) | python train.py -c Static_Network.py -s True                         |


### Resuming from checkpoints
Resume the training form a certain checkpoint can be done by executing:

  ```
  python train.py --resume path/to/checkpoint
  ```

### Test
A trained model can be tested by executing `test.py` passing the path to the trained checkpoint to the `--resume` argument.
For the networks which use a separate model for each plane orientation, the model can be tested with `test_single_view.py`. 
Here the _parent folder_ of a checkpoint has to be provided as the `--resume` argument.

For testing a **longitudinal** model we perform a majority vote over all possible combinates for a given target image with its reference images. 
A longitudinal model usually has a reference image(timepoint t-1) and the target/follow-up image(timepoint t) as input. 
Our experiments have shown that we achieve the best performance when applying a majority vote over all possible permutations for a certain target image. 
This means, for a patient with four timesteps t ∈ {0; 1; 2; 3} and having t = 1 as the target image, we do 
a majority votes over the outputs of the inputs (reference, target): (0, 1), (2, 1), (3, 1).

### General notes
All hyperparameters are defined in the config file.
Majority vote for merging the outputs of the different plane orientations is only applied in `test.py` and `test_single_view.py`. 
The majority voting (MV) is done on the merged probability maps for a voxel from each view. 
Since it is from great interest to see the actual performance (i.e. after MV) of a model on the validation set, the 
test script has an argument `-e` or `--evaluate` which can be either `train` or `test`. 
This argument specifies which data should be used. For evaluating the performance of the model on the 
train/validation set, this argument has to be `train` else `test`(default).

## Disclaimer
The code has been cleaned and polished for the sake of clarity and reproducibility, and even though it has been checked thoroughly, it might contain bugs or mistakes. Please do not hesitate to open an issue or contact the authors to inform of any problem you may find within this repository.

## License
This project is licensed under the MIT License. See LICENSE for more details

## Acknowledgements
This project is a fork of the project [PyTorch-Template](https://github.com/victoresque/pytorch-template) by [Victor Huang](https://github.com/victoresque)
