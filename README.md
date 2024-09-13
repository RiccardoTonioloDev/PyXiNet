# PyXiNet

Implementing an embeddable deep learning model for Monocular Depth Estimation.
Various versions of PyXiNet were implemented. The best embeddable version is PyXiNetBCBAM1, while the best overall version (on Eigen evaluation metrics) is PyXiNetM2.

# General information

In this repository you can find these main files:

-   `Config.py`: the file that implements the configurations format;
-   `evaluating.py`: the file that implements the evaluation logic;
-   `KittiDataset.py`: it's the dataset implementation (it can be used for other types of datasets other than KITTi, such as CityScapes for example, as long as they have a file with images paths specified that can be used by the `KittiDataset` class);
-   `Losses.py`: here the losses used by the training procedure are implemented;
-   `main.py`: here the main logic is implemented (this is the file that has to be executed in order to train, use or evaluate the models);
-   `PyXiNet.py`: here all the various versions of PyXiNet are implemented;
-   `testing.py`: here the testing logic is implemented;
-   `training.py`: here the training logic is implemented;
-   `using.py`: here the logic for the model usage is implemented;
-   `webcam.py`: here the logic for the model usage through the webcam is implemented.

In this repository you can also find these main folders:

-   `10_test_images`: those are 10 random images from the KITTI dataset, used to evaluate the model on inference time;
-   `Blocks`: here the various blocks used inside of the various versions of PyXiNet are implemented;
-   `Configs`: here the various configurations that were used to train, use or evaluate PyXiNet are implemented (watch the **Configurations** subsection to learn more);
-   `filenames`: here are stored the various files containing the paths for the images of the KITTI and CityScapes dataset;
-   `outputfiles`: this is a utility directory, made to store the outputs of the various procedures (slurm output files, and models checkpoints);
-   `slurm_files`: here are stored the various slurm files used to train the models.

## Info

> Note: `wandb` was used to log the different losses. To use it you'll have to:
>
> -   [create an account](https://wandb.ai/login?signup=true);
> -   install the package locally;
> -   configure the packate with your account information.

# The code

## Requirements

```bash
# Create the conda environment (use your preferred name)
conda create -n <environmentName>
# Activate the conda environment
conda activate <environmentName>
# Install the required packages (I'll use conda for torch)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# Install the required packages (I'll use pip for everything else)
pip install wandb pandas matplotlib Pillow
```

> **WARNING**: if you want to use the `--use=webcam` flag, your system must have the `ffmpeg` command installed and know that this functionality was only tested on a macOS device with an M1 Pro ARM CPU. I had to use it because ARM chips can't use open-cv yet.

> **IMPORTANT**: choose the cuda version based on the cuda version of your system.

## Configurations

To make things smoother to try and test, this project is based on configurations, lowering the amount of cli parameters you have to care for while executing the scripts.

You can find two examples of configurations inside the `Configs` folder. Every configuration parameter that's not obvious it's well documented in the provided examples.

You'll want to create your own configuration or modify the existing ones to specify different parameters, including the dataset path, the image resolution, and so on.

To create a custom configuration, copy one of the examples (i.e. `Configs/ConfigHomeLab.py`) and modify it to your likings.

After you created your own configuration, you have to:

-   Import it inside of `Config.py`, and add the conditional logic to use your specified configuration;
-   Import it inside of `testing.py` and add it as the possible types of the parameter `config` inside of the `evaluate_on_test_set` function;
-   In the `main.py` file you could add to the helper of the parser of the `--env` parameter, the name that has to be provided in order to select your new configuration.

After that you are done!

## Training

**IMPORTANT**: make sure that the program it's using the right configuration as explained in the configurations section.

This will generate the checkpoint of the last epoch and will maintain the checkpoint that had the best performance on the test set, inside the directory specified by the `checkpoint_path` attribute of the selected configuration.

```bash
python3 main.py --mode=train --env=<NameOfTheConfigurationYouWantToUse>
```

## Testing

**IMPORTANT**: make sure that the program it's using the right configuration as explained in the configurations section.

This is used to generate the `disparities.npy` file. It will contain the disparities calculated for the images of the choosen test set.

The file will be placed inside the directory specified by the `output_directory` attribute of the selected configuration.
To execute the testing you should have a checkpoint first, specified by the `checkpoint_to_use_path` attribute of the selected configuration.

```bash
python3 main.py --mode=test --env=<NameOfTheConfigurationYouWantToUse>
```

## Evaluating

**IMPORTANT**: make sure that the program it's using the right configuration as explained in the configurations section.

This is used to evaluate the model on the `disparities.npy` file, generated from the test set (look at the testing section).

It will also measure the average of the inference time, of the model on 10 different images (that you can find inside of the `10_test_images/` folder), using only the CPU as the computing device.

To execute the evalutation you should have a checkpoint first, specified by the `checkpoint_to_use_path` attribute , and a `disparities.npy` file inside the folder specified by the `output_directory` attribute of the selected configuration.

```bash
python3 main.py --mode=eval --env=<NameOfTheConfigurationYouWantToUse>
```

## Using

**IMPORTANT**: make sure that the program it's using the right configuration as explained in the configurations section.

This will create a depth map image in the same folder of the image that was provided to the model.

To use the model on an image you should have a checkpoint first, specified by the `checkpoint_to_use_path` attribute of the selected configuration.

```bash
python3 main.py --mode=use --env=<NameOfTheConfigurationYouWantToUse> --img_path=<pathOfTheImageYouWantToUse>
```

# Resources

The following is a list of the papers I studied to develop the various versions of PyXiNet:

```yaml
PyDNetV1:
    type: Article
    title: Towards real-time unsupervised monocular depth estimation on CPU
    author:
        [
            'Poggi, Matteo',
            'Aleotti, Filippo',
            'Tosi, Fabio',
            'Mattoccia, Stefano',
        ]
    publisher: IEEE/JRS Conference on Intelligent Robots and Systems (IROS)
    date: 2018
PyDNetV2:
    type: Article
    title: Real-time Self-Supervised Monocular Depth Estimation Without GPU
    author:
        [
            'Poggi, Matteo',
            'Tosi, Fabio',
            'Aleotti, Filippo',
            'Mattoccia, Stefano',
        ]
    publisher: IEEE Transactions on Intelligent Transportation Systems
    date: 2022
XiNet:
    type: Article
    title: 'XiNet: Efficient Neural Networks for tinyML'
    author: ['Alberto, Ancilotto', 'Francesco, Paissan', 'Elisabetta, Farella']
    publisher: ICCV
    date: 2023
SelfAttentionComputerVision:
    type: Article
    title: Non-local Neural Networks
    author: ['Xiaolong Wang', 'Ross Girshick', 'Abhinav Gupta', 'Kaiming He']
    publisher: CVPR
    date: 2018
CBAM:
    type: Article
    title: 'CBAM: Convolutional Block Attention Module'
    author: ['Sanghyun Woo', 'Jongchan Park', 'Joon-Young Lee', 'In So Kweon']
    publisher: ECCV
    date: 2018
ResNet:
    type: Article
    title: Deep Residual Learning for Image Recognition
    author: ['Kaiming, He', 'Xiangyu, Zhang', 'Shaoqing, Ren', 'Jian, Sun']
    publisher: CVPR
    date: 2016
```
