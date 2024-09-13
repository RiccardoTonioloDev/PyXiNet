from typing import Tuple, Optional, Literal
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
import pandas as pd
from PIL import Image
import os
import random
from Config import Config
from Utils.HSVconverters import rgb_to_hsv


class KittiDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        filenames_file_path: str,
        image_width: int,
        image_height: int,
        config: Config,
        mode: Literal["train", "test"] = "train",
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            - `data_path`: Path to the dataset folder containing images;
            - `filenames_file_path`: Path to the file containing image filenames;
            - `image_width`: Width to resize the images;
            - `image_height`: Height to resize the images;
            - `config`: the environment configuration;
            - `mode`: can be "train" for training and "test" for testing;
            - `transform`: an optional transformation to be applied to images.
        """
        self.config = config.get_configuration()
        self.filenames_df = pd.read_csv(filenames_file_path, sep=r"\s+", header=None)
        self.leftImagePathPrefix = ""
        self.rightImagePathPrefix = ""
        if "cityscapes" in filenames_file_path:
            self.leftImagePathPrefix = "leftImg8bit"
            self.rightImagePathPrefix = "rightImg8bit"

        self.data_path = data_path
        self.image_tensorizer = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        )  # It allows us to transform each image retrieved from the dataset in a tensor
        self.mode = mode
        self.transform = transform
        self.__horizontal_flipper = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=1)]
        )
        if self.config.VerticalFlipAugmentation:
            self.__vertical_flipper = transforms.Compose(
                [transforms.RandomVerticalFlip(p=1)]
            )
        self.image_width = image_width
        self.image_height = image_height

    def __len__(self) -> int:
        num_rows, _ = self.filenames_df.shape
        return num_rows

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        path_ith_row_left, path_ith_row_right = self.filenames_df.iloc[i]

        left_image_path = os.path.join(
            self.data_path, self.leftImagePathPrefix, path_ith_row_left
        )
        right_image_path = os.path.join(
            self.data_path, self.rightImagePathPrefix, path_ith_row_right
        )

        try:
            with Image.open(left_image_path) as left_image:
                left_image = left_image.convert("RGB")
                left_image_tensor: torch.Tensor = self.image_tensorizer(
                    left_image
                ).unsqueeze(0)
                left_image_tensor = torch.nn.functional.interpolate(
                    left_image_tensor,
                    (self.image_height, self.image_width),
                    mode="area",
                ).squeeze(0)

        except Exception as e:
            raise RuntimeError(f"Error loading left image: {left_image_path}. {e}")
        if self.transform:
            left_image_tensor = self.transform(left_image_tensor)

        # Checking for testing
        if self.mode == "test":
            if self.config.BlackAndWhite_processing:
                left_image_tensor = F.rgb_to_grayscale(left_image_tensor)
            return left_image_tensor

        try:
            with Image.open(right_image_path) as right_image:
                right_image = right_image.convert("RGB")
                right_image_tensor: torch.Tensor = self.image_tensorizer(
                    right_image
                ).unsqueeze(0)
                right_image_tensor = torch.nn.functional.interpolate(
                    right_image_tensor,
                    (self.image_height, self.image_width),
                    mode="area",
                ).squeeze()
        except Exception as e:
            raise RuntimeError(f"Error loading right image: {right_image_path}. {e}")
        if self.transform:
            right_image_tensor = self.transform(right_image_tensor)

        if self.mode == "train":
            # Randomly flipping images
            if random.random() > 0.5:
                left_image_tensor_copy = left_image_tensor
                left_image_tensor = self.__horizontal_flipper(right_image_tensor)
                right_image_tensor = self.__horizontal_flipper(left_image_tensor_copy)

            # Randomly augmenting images
            if (
                random.random() > 0.5 and not self.config.VerticalFlipAugmentation
            ):  # The experimental training on VFlip has to be turned off
                left_image_tensor, right_image_tensor = self.augment_image_pair(
                    left_image_tensor, right_image_tensor
                )

            # Experimental training for vertical flip
            if self.config.VerticalFlipAugmentation and random.random() > 0.5:
                left_image_tensor = self.__vertical_flipper(left_image_tensor)
                right_image_tensor = self.__vertical_flipper(right_image_tensor)

            if self.config.HSV_processing:
                left_image_tensor = rgb_to_hsv(left_image_tensor)
                right_image_tensor = rgb_to_hsv(right_image_tensor)
            if self.config.BlackAndWhite_processing:
                left_image_tensor = F.rgb_to_grayscale(left_image_tensor)
                right_image_tensor = F.rgb_to_grayscale(right_image_tensor)

        return left_image_tensor, right_image_tensor

    def augment_image_pair(
        self, left_image: torch.Tensor, right_image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        It augments a pair of images working on gamma, brightness, colors and saturation.
            - `left_image`: the left image in the stereo pair;
            - `right_image`: the corresponding right image in the stereo pair.

        It returns the augmented pair of images.
        """
        # Shifting with random gamma
        gamma = random.uniform(0.8, 1.2)
        left_image = transforms.functional.adjust_gamma(left_image, gamma)
        right_image = transforms.functional.adjust_gamma(right_image, gamma)

        # Shifting with random brightness
        brightness = random.uniform(0.5, 2)
        left_image = transforms.functional.adjust_brightness(left_image, brightness)
        right_image = transforms.functional.adjust_brightness(right_image, brightness)

        # Shifting with random colors
        random_colors = torch.FloatTensor(3).uniform_(0.8, 1.2)
        white = torch.ones(left_image.size(1), left_image.size(2))
        color_image = torch.stack([white * random_colors[i] for i in range(3)], dim=0)

        # saturate
        left_image = left_image * color_image
        right_image = right_image * color_image

        left_image = torch.clamp(left_image, 0, 1)
        right_image = torch.clamp(right_image, 0, 1)

        return left_image, right_image

    def make_dataloader(
        self,
        batch_size: int = 8,
        shuffle_batch: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> DataLoader:
        """
        It creates a dataloader from the dataset.
            - `batch_size`: the number of samples inside a single batch;
            - `shuffle_batch`: if true the batches will be different in every epoch;
            - `num_workers`: the number of workers used to create batches;
            - `pin_memory`: leave it to true (it's to optimize the flow of information between CPU and GPU).

        Returns the configured dataloader.
        """
        dataloader = DataLoader(
            self,
            batch_size,
            shuffle_batch,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return dataloader

    @staticmethod
    def from_left_to_left_batch(image_tensor: torch.Tensor) -> torch.Tensor:
        """
        To be used during testing, where we suppose to do evaluations only on the left image.
            - `image_tensor`: the tensor we suppose being the left_image.

        Returns a Tensor[2,3,H,W] corresponding to the left image stacked on top of the
        horizontally-flipped-left image.
        """
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor.squeeze(0)
        elif len(image_tensor.shape) == 2:
            image_tensor = image_tensor.unsqueeze(0)
        left_image_tensor_flipped = transforms.functional.hflip(image_tensor)
        return torch.stack([image_tensor, left_image_tensor_flipped], dim=0)
