from typing import List, Literal
from Blocks.UpsizingBlock import UpsizingBlock
from Blocks.DownsizingBlock import DownsizingBlock
from Blocks.LevelConvolutionsBlock import LevelConvolutionsBlock
from Blocks.DownsizingCBAMBlock import DownsizingCBAMBlock
from Blocks.LevelCBAMBlock import LevelCBAMBlock
import torch.nn as nn
import torch.nn.functional as F
import torch
from Blocks.Xavier_initializer import xavier_init
from Blocks.XiEncoder import XiEncoder
from Blocks.LightAttention import LightSelfAttentionCNN1, LightSelfAttentionCNN2
from Configs.ConfigCluster import ConfigCluster
from Configs.ConfigHomeLab import ConfigHomeLab


class PyXiNetA1(nn.Module):
    def __init__(self, config: ConfigHomeLab | ConfigCluster):
        super(PyXiNetA1, self).__init__()

        # LEVEL 1
        input_shape_lv1 = [3, config.image_height, config.image_width]
        self.__enc_1 = XiEncoder(
            input_shape=input_shape_lv1,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=3,
            mid_channels=12,
            out_channels=16,
        )
        self.__conv_block_1 = LevelConvolutionsBlock(16 + 8)

        # LEVEL 2
        input_shape_lv2 = [16, input_shape_lv1[1] // 2, input_shape_lv1[2] // 2]
        self.__enc_2 = XiEncoder(
            input_shape=input_shape_lv2,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=3,
            mid_channels=24,
            out_channels=32,
        )
        self.__conv_block_2 = LevelConvolutionsBlock(32 + 8)

        # LEVEL 3
        input_shape_lv3 = [32, input_shape_lv2[1] // 2, input_shape_lv2[2] // 2]
        self.__enc_3 = XiEncoder(
            input_shape=input_shape_lv3,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=3,
            mid_channels=28,
            out_channels=64,
        )
        self.__conv_block_3 = LevelConvolutionsBlock(64)

        self.__upsizing_3 = UpsizingBlock(8, 8)
        self.__upsizing_2 = UpsizingBlock(8, 8)

        self.apply(xavier_init)

    def __level_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gives back the disparities for left images on the 0th channel and for the right images on the 1th channel.
            - `x`: torch.Tensor[B,C,W,H]

        Returns a torch.Tensor[B,2,W,H]
        """
        return 0.3 * torch.sigmoid(x[:, :2, :, :])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Level's starting blocks
        conv1 = self.__enc_1(x)  # [8, 7, 128, 256]
        conv2 = self.__enc_2(conv1)  # [8, 16, 64, 128]
        conv3 = self.__enc_3(conv2)  # [8, 36, 32, 64]

        # LEVEL 3
        conv3b = self.__conv_block_3(conv3)
        disp3 = self.__level_activations(conv3b)

        conv3b = self.__upsizing_3(conv3b)

        # LEVEL 2
        concat2 = torch.cat((conv2, conv3b), 1)
        conv2b = self.__conv_block_2(concat2)
        disp2 = self.__level_activations(conv2b)

        conv2b = self.__upsizing_2(conv2b)

        # LEVEL 1
        concat1 = torch.cat((conv1, conv2b), 1)
        conv1b = self.__conv_block_1(concat1)
        disp1 = self.__level_activations(conv1b)

        return [
            disp1,
            disp2,
            disp3,
        ]

    @staticmethod
    def scale_pyramid(img_batch: torch.Tensor, num_scales: int = 3) -> torch.Tensor:
        """
        It scales the batch of images to `num_scales` scales, everytime dividing by 2 the height and the width.
            - `img_batch`: torch.Tensor[B,C,H,W]
            - `num_scales`: int

        Returns a List[torch.Tensor[B,C,H,W]] with a length of `num_scales`.
        """
        scaled_imgs = []
        _, _, h, w = img_batch.shape
        for i in range(num_scales):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_img = nn.functional.interpolate(
                img_batch, size=(nh, nw), mode="area"
            )
            scaled_imgs.append(scaled_img)
        return scaled_imgs

    @staticmethod
    def upscale_img(
        img_tensor: torch.Tensor, new_2d_size: tuple[int, int] = (256, 512)
    ) -> torch.Tensor:
        """
        It upscales the `img_tensor` using bilinear interpolation to the specified size `new_2d_size`.
            - `img_tensor`: torch.Tensor[B,C,H,W];
            - `new_2d_size`: tuple(width,height).

        Returns a torch.Tensor[B,C,width, height].
        """
        return F.interpolate(
            img_tensor, size=new_2d_size, mode="bilinear", align_corners=True
        )


class PyXiNetA2(nn.Module):
    def __init__(self, config: ConfigHomeLab | ConfigCluster):
        super(PyXiNetA2, self).__init__()

        # LEVEL 1
        input_shape_lv1 = [3, config.image_height, config.image_width]
        self.__enc_1 = XiEncoder(
            input_shape=input_shape_lv1,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=3,
            mid_channels=12,
            out_channels=16,
        )
        self.__conv_block_1 = LevelConvolutionsBlock(16 + 8)

        # LEVEL 2
        input_shape_lv2 = [16, input_shape_lv1[1] // 2, input_shape_lv1[2] // 2]
        self.__enc_2 = XiEncoder(
            input_shape=input_shape_lv2,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=3,
            mid_channels=24,
            out_channels=32,
        )
        self.__conv_block_2 = LevelConvolutionsBlock(32 + 8)

        # LEVEL 3
        input_shape_lv3 = [32, input_shape_lv2[1] // 2, input_shape_lv2[2] // 2]
        self.__enc_3 = XiEncoder(
            input_shape=input_shape_lv3,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=3,
            mid_channels=28,
            out_channels=64,
        )
        self.__conv_block_3 = LevelConvolutionsBlock(64 + 8)

        # LEVEL 4
        input_shape_lv4 = [64, input_shape_lv3[1] // 2, input_shape_lv3[2] // 2]
        self.__enc_4 = XiEncoder(
            input_shape=input_shape_lv4,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=3,
            mid_channels=32,
            out_channels=96,
        )
        self.__conv_block_4 = LevelConvolutionsBlock(96)

        self.__upsizing_4 = UpsizingBlock(8, 8)
        self.__upsizing_3 = UpsizingBlock(8, 8)
        self.__upsizing_2 = UpsizingBlock(8, 8)

        self.apply(xavier_init)

    def __level_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gives back the disparities for left images on the 0th channel and for the right images on the 1th channel.
            - `x`: torch.Tensor[B,C,W,H]

        Returns a torch.Tensor[B,2,W,H]
        """
        return 0.3 * torch.sigmoid(x[:, :2, :, :])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Level's starting blocks
        conv1 = self.__enc_1(x)
        conv2 = self.__enc_2(conv1)
        conv3 = self.__enc_3(conv2)
        conv4 = self.__enc_4(conv3)

        # LEVEL 4
        conv4b = self.__conv_block_4(conv4)
        disp4 = self.__level_activations(conv4b)

        conv4b = self.__upsizing_4(conv4b)

        # LEVEL 3
        concat3 = torch.cat((conv3, conv4b), 1)
        conv3b = self.__conv_block_3(concat3)
        disp3 = self.__level_activations(conv3b)

        conv3b = self.__upsizing_3(conv3b)

        # LEVEL 2
        concat2 = torch.cat((conv2, conv3b), 1)
        conv2b = self.__conv_block_2(concat2)
        disp2 = self.__level_activations(conv2b)

        conv2b = self.__upsizing_2(conv2b)

        # LEVEL 1
        concat1 = torch.cat((conv1, conv2b), 1)
        conv1b = self.__conv_block_1(concat1)
        disp1 = self.__level_activations(conv1b)

        return [disp1, disp2, disp3, disp4]

    @staticmethod
    def scale_pyramid(img_batch: torch.Tensor, num_scales: int = 3) -> torch.Tensor:
        """
        It scales the batch of images to `num_scales` scales, everytime dividing by 2 the height and the width.
            - `img_batch`: torch.Tensor[B,C,H,W]
            - `num_scales`: int

        Returns a List[torch.Tensor[B,C,H,W]] with a length of `num_scales`.
        """
        scaled_imgs = []
        _, _, h, w = img_batch.shape
        for i in range(num_scales):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_img = nn.functional.interpolate(
                img_batch, size=(nh, nw), mode="area"
            )
            scaled_imgs.append(scaled_img)
        return scaled_imgs

    @staticmethod
    def upscale_img(
        img_tensor: torch.Tensor, new_2d_size: tuple[int, int] = (256, 512)
    ) -> torch.Tensor:
        """
        It upscales the `img_tensor` using bilinear interpolation to the specified size `new_2d_size`.
            - `img_tensor`: torch.Tensor[B,C,H,W];
            - `new_2d_size`: tuple(width,height).

        Returns a torch.Tensor[B,C,width, height].
        """
        return F.interpolate(
            img_tensor, size=new_2d_size, mode="bilinear", align_corners=True
        )


class PyXiNetB1(nn.Module):
    def __init__(self, config: ConfigHomeLab | ConfigCluster):
        super(PyXiNetB1, self).__init__()
        input_shape = [3, config.image_height, config.image_width]

        # LEVEL 1
        self.__enc_1 = DownsizingBlock(3, 16)
        self.__conv_block_1 = LevelConvolutionsBlock(16 + 8)

        # LEVEL 2
        self.__enc_2 = XiEncoder(
            input_shape=input_shape,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=4,
            mid_channels=24,
            out_channels=32,
        )
        self.__conv_block_2 = LevelConvolutionsBlock(32 + 8)

        # LEVEL 3
        self.__enc_3 = XiEncoder(
            input_shape=input_shape,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=5,
            mid_channels=28,
            out_channels=64,
        )
        self.__conv_block_3 = LevelConvolutionsBlock(64)

        self.__upsizing_3 = UpsizingBlock(8, 8)
        self.__upsizing_2 = UpsizingBlock(8, 8)

        self.apply(xavier_init)

    def __level_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gives back the disparities for left images on the 0th channel and for the right images on the 1th channel.
            - `x`: torch.Tensor[B,C,W,H]

        Returns a torch.Tensor[B,2,W,H]
        """
        return 0.3 * torch.sigmoid(x[:, :2, :, :])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Level's starting blocks
        conv1 = self.__enc_1(x)
        conv2 = self.__enc_2(x)
        conv3 = self.__enc_3(x)

        # LEVEL 3
        conv3b = self.__conv_block_3(conv3)
        disp3 = self.__level_activations(conv3b)

        conv3b = self.__upsizing_3(conv3b)

        # LEVEL 2
        concat2 = torch.cat((conv2, conv3b), 1)
        conv2b = self.__conv_block_2(concat2)
        disp2 = self.__level_activations(conv2b)

        conv2b = self.__upsizing_2(conv2b)

        # LEVEL 1
        concat1 = torch.cat((conv1, conv2b), 1)
        conv1b = self.__conv_block_1(concat1)
        disp1 = self.__level_activations(conv1b)

        return [
            disp1,
            disp2,
            disp3,
        ]

    @staticmethod
    def scale_pyramid(img_batch: torch.Tensor, num_scales: int = 3) -> torch.Tensor:
        """
        It scales the batch of images to `num_scales` scales, everytime dividing by 2 the height and the width.
            - `img_batch`: torch.Tensor[B,C,H,W]
            - `num_scales`: int

        Returns a List[torch.Tensor[B,C,H,W]] with a length of `num_scales`.
        """
        scaled_imgs = []
        _, _, h, w = img_batch.shape
        for i in range(num_scales):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_img = nn.functional.interpolate(
                img_batch, size=(nh, nw), mode="area"
            )
            scaled_imgs.append(scaled_img)
        return scaled_imgs

    @staticmethod
    def upscale_img(
        img_tensor: torch.Tensor, new_2d_size: tuple[int, int] = (256, 512)
    ) -> torch.Tensor:
        """
        It upscales the `img_tensor` using bilinear interpolation to the specified size `new_2d_size`.
            - `img_tensor`: torch.Tensor[B,C,H,W];
            - `new_2d_size`: tuple(width,height).

        Returns a torch.Tensor[B,C,width, height].
        """
        return F.interpolate(
            img_tensor, size=new_2d_size, mode="bilinear", align_corners=True
        )


class PyXiNetB2(nn.Module):
    def __init__(self, config: ConfigHomeLab | ConfigCluster):
        super(PyXiNetB2, self).__init__()
        input_shape = [3, config.image_height, config.image_width]

        # LEVEL 1
        self.__enc_1 = DownsizingBlock(3, 16)
        self.__conv_block_1 = LevelConvolutionsBlock(16 + 8)

        # LEVEL 2
        self.__enc_2 = XiEncoder(
            input_shape=input_shape,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=3,
            mid_channels=24,
            out_channels=32,
            with_upscaling=False,
        )
        self.__conv_block_2 = LevelConvolutionsBlock(32 + 8)

        # LEVEL 3
        self.__enc_3 = XiEncoder(
            input_shape=input_shape,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=4,
            mid_channels=28,
            out_channels=64,
            with_upscaling=False,
        )
        self.__conv_block_3 = LevelConvolutionsBlock(64)

        self.__upsizing_3 = UpsizingBlock(8, 8)
        self.__upsizing_2 = UpsizingBlock(8, 8)

        self.apply(xavier_init)

    def __level_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gives back the disparities for left images on the 0th channel and for the right images on the 1th channel.
            - `x`: torch.Tensor[B,C,W,H]

        Returns a torch.Tensor[B,2,W,H]
        """
        return 0.3 * torch.sigmoid(x[:, :2, :, :])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Level's starting blocks
        conv1 = self.__enc_1(x)
        conv2 = self.__enc_2(x)
        conv3 = self.__enc_3(x)

        # LEVEL 3
        conv3b = self.__conv_block_3(conv3)
        disp3 = self.__level_activations(conv3b)

        conv3b = self.__upsizing_3(conv3b)

        # LEVEL 2
        concat2 = torch.cat((conv2, conv3b), 1)
        conv2b = self.__conv_block_2(concat2)
        disp2 = self.__level_activations(conv2b)

        conv2b = self.__upsizing_2(conv2b)

        # LEVEL 1
        concat1 = torch.cat((conv1, conv2b), 1)
        conv1b = self.__conv_block_1(concat1)
        disp1 = self.__level_activations(conv1b)

        return [
            disp1,
            disp2,
            disp3,
        ]

    @staticmethod
    def scale_pyramid(img_batch: torch.Tensor, num_scales: int = 3) -> torch.Tensor:
        """
        It scales the batch of images to `num_scales` scales, everytime dividing by 2 the height and the width.
            - `img_batch`: torch.Tensor[B,C,H,W]
            - `num_scales`: int

        Returns a List[torch.Tensor[B,C,H,W]] with a length of `num_scales`.
        """
        scaled_imgs = []
        _, _, h, w = img_batch.shape
        for i in range(num_scales):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_img = nn.functional.interpolate(
                img_batch, size=(nh, nw), mode="area"
            )
            scaled_imgs.append(scaled_img)
        return scaled_imgs

    @staticmethod
    def upscale_img(
        img_tensor: torch.Tensor, new_2d_size: tuple[int, int] = (256, 512)
    ) -> torch.Tensor:
        """
        It upscales the `img_tensor` using bilinear interpolation to the specified size `new_2d_size`.
            - `img_tensor`: torch.Tensor[B,C,H,W];
            - `new_2d_size`: tuple(width,height).

        Returns a torch.Tensor[B,C,width, height].
        """
        return F.interpolate(
            img_tensor, size=new_2d_size, mode="bilinear", align_corners=True
        )


class PyXiNetB3(nn.Module):
    def __init__(self, config: ConfigHomeLab | ConfigCluster):
        super(PyXiNetB3, self).__init__()
        input_shape = [3, config.image_height, config.image_width]

        # LEVEL 1
        self.__enc_1 = DownsizingBlock(3, 16)
        self.__conv_block_1 = LevelConvolutionsBlock(16 + 8)

        # LEVEL 2
        self.__enc_2 = XiEncoder(
            input_shape=input_shape,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=4,
            mid_channels=24,
            out_channels=32,
        )
        self.__conv_block_2 = LevelConvolutionsBlock(32 + 8)

        # LEVEL 3
        self.__enc_3 = XiEncoder(
            input_shape=input_shape,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=5,
            mid_channels=28,
            out_channels=64,
        )
        self.__conv_block_3 = LevelConvolutionsBlock(64 + 8)

        # LEVEL 4
        self.__enc_4 = DownsizingBlock(64, 96)
        self.__conv_block_4 = LevelConvolutionsBlock(96)

        self.__upsizing_4 = UpsizingBlock(8, 8)
        self.__upsizing_3 = UpsizingBlock(8, 8)
        self.__upsizing_2 = UpsizingBlock(8, 8)

        self.apply(xavier_init)

    def __level_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gives back the disparities for left images on the 0th channel and for the right images on the 1th channel.
            - `x`: torch.Tensor[B,C,W,H]

        Returns a torch.Tensor[B,2,W,H]
        """
        return 0.3 * torch.sigmoid(x[:, :2, :, :])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Level's starting blocks
        conv1 = self.__enc_1(x)
        conv2 = self.__enc_2(x)
        conv3 = self.__enc_3(x)
        conv4 = self.__enc_4(conv3)

        # LEVEL 4
        conv4b = self.__conv_block_4(conv4)
        disp4 = self.__level_activations(conv4b)

        conv4b = self.__upsizing_4(conv4b)

        # LEVEL 3
        concat3 = torch.cat((conv3, conv4b), 1)
        conv3b = self.__conv_block_3(concat3)
        disp3 = self.__level_activations(conv3b)

        conv3b = self.__upsizing_3(conv3b)

        # LEVEL 2
        concat2 = torch.cat((conv2, conv3b), 1)
        conv2b = self.__conv_block_2(concat2)
        disp2 = self.__level_activations(conv2b)

        conv2b = self.__upsizing_2(conv2b)

        # LEVEL 1
        concat1 = torch.cat((conv1, conv2b), 1)
        conv1b = self.__conv_block_1(concat1)
        disp1 = self.__level_activations(conv1b)

        return [
            disp1,
            disp2,
            disp3,
            disp4,
        ]

    @staticmethod
    def scale_pyramid(img_batch: torch.Tensor, num_scales: int = 3) -> torch.Tensor:
        """
        It scales the batch of images to `num_scales` scales, everytime dividing by 2 the height and the width.
            - `img_batch`: torch.Tensor[B,C,H,W]
            - `num_scales`: int

        Returns a List[torch.Tensor[B,C,H,W]] with a length of `num_scales`.
        """
        scaled_imgs = []
        _, _, h, w = img_batch.shape
        for i in range(num_scales):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_img = nn.functional.interpolate(
                img_batch, size=(nh, nw), mode="area"
            )
            scaled_imgs.append(scaled_img)
        return scaled_imgs

    @staticmethod
    def upscale_img(
        img_tensor: torch.Tensor, new_2d_size: tuple[int, int] = (256, 512)
    ) -> torch.Tensor:
        """
        It upscales the `img_tensor` using bilinear interpolation to the specified size `new_2d_size`.
            - `img_tensor`: torch.Tensor[B,C,H,W];
            - `new_2d_size`: tuple(width,height).

        Returns a torch.Tensor[B,C,width, height].
        """
        return F.interpolate(
            img_tensor, size=new_2d_size, mode="bilinear", align_corners=True
        )


class PyXiNetB4(nn.Module):
    def __init__(self, config: ConfigHomeLab | ConfigCluster):
        super(PyXiNetB4, self).__init__()
        input_shape = [3, config.image_height, config.image_width]

        # LEVEL 1
        self.__enc_1 = DownsizingBlock(3, 16)
        self.__conv_block_1 = LevelConvolutionsBlock(16 + 8)

        # LEVEL 2
        self.__enc_2 = XiEncoder(
            input_shape=input_shape,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=3,
            mid_channels=24,
            out_channels=32,
            with_upscaling=False,
        )
        self.__conv_block_2 = LevelConvolutionsBlock(32 + 8)

        # LEVEL 3
        self.__enc_3 = XiEncoder(
            input_shape=input_shape,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=4,
            mid_channels=28,
            out_channels=64,
            with_upscaling=False,
        )
        self.__conv_block_3 = LevelConvolutionsBlock(64 + 8)

        # LEVEL 4
        self.__enc_4 = XiEncoder(
            input_shape=input_shape,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=5,
            mid_channels=32,
            out_channels=96,
            with_upscaling=False,
        )
        self.__conv_block_4 = LevelConvolutionsBlock(96)

        self.__upsizing_4 = UpsizingBlock(8, 8)
        self.__upsizing_3 = UpsizingBlock(8, 8)
        self.__upsizing_2 = UpsizingBlock(8, 8)

        self.apply(xavier_init)

    def __level_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gives back the disparities for left images on the 0th channel and for the right images on the 1th channel.
            - `x`: torch.Tensor[B,C,W,H]

        Returns a torch.Tensor[B,2,W,H]
        """
        return 0.3 * torch.sigmoid(x[:, :2, :, :])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Level's starting blocks
        conv1 = self.__enc_1(x)
        conv2 = self.__enc_2(x)
        conv3 = self.__enc_3(x)
        conv4 = self.__enc_4(x)

        # LEVEL 4
        conv4b = self.__conv_block_4(conv4)
        disp4 = self.__level_activations(conv4b)

        conv4b = self.__upsizing_4(conv4b)

        # LEVEL 3
        concat3 = torch.cat((conv3, conv4b), 1)
        conv3b = self.__conv_block_3(concat3)
        disp3 = self.__level_activations(conv3b)

        conv3b = self.__upsizing_3(conv3b)

        # LEVEL 2
        concat2 = torch.cat((conv2, conv3b), 1)
        conv2b = self.__conv_block_2(concat2)
        disp2 = self.__level_activations(conv2b)

        conv2b = self.__upsizing_2(conv2b)

        # LEVEL 1
        concat1 = torch.cat((conv1, conv2b), 1)
        conv1b = self.__conv_block_1(concat1)
        disp1 = self.__level_activations(conv1b)

        return [disp1, disp2, disp3, disp4]

    @staticmethod
    def scale_pyramid(img_batch: torch.Tensor, num_scales: int = 3) -> torch.Tensor:
        """
        It scales the batch of images to `num_scales` scales, everytime dividing by 2 the height and the width.
            - `img_batch`: torch.Tensor[B,C,H,W]
            - `num_scales`: int

        Returns a List[torch.Tensor[B,C,H,W]] with a length of `num_scales`.
        """
        scaled_imgs = []
        _, _, h, w = img_batch.shape
        for i in range(num_scales):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_img = nn.functional.interpolate(
                img_batch, size=(nh, nw), mode="area"
            )
            scaled_imgs.append(scaled_img)
        return scaled_imgs

    @staticmethod
    def upscale_img(
        img_tensor: torch.Tensor, new_2d_size: tuple[int, int] = (256, 512)
    ) -> torch.Tensor:
        """
        It upscales the `img_tensor` using bilinear interpolation to the specified size `new_2d_size`.
            - `img_tensor`: torch.Tensor[B,C,H,W];
            - `new_2d_size`: tuple(width,height).

        Returns a torch.Tensor[B,C,width, height].
        """
        return F.interpolate(
            img_tensor, size=new_2d_size, mode="bilinear", align_corners=True
        )


class PyXiNetM1(nn.Module):
    def __init__(self, config: ConfigHomeLab | ConfigCluster):
        super(PyXiNetM1, self).__init__()
        input_shape = [3, config.image_height, config.image_width]

        # LEVEL 1
        self.__enc_1 = DownsizingBlock(3, 16)
        self.__lsa_1 = LightSelfAttentionCNN1(16, 128, 256)
        self.__conv_block_1 = LevelConvolutionsBlock(16 + 8)

        # LEVEL 2
        self.__enc_2 = XiEncoder(
            input_shape=input_shape,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=3,
            mid_channels=24,
            out_channels=32,
            with_upscaling=False,
        )
        self.__conv_block_2 = LevelConvolutionsBlock(32 + 8)

        # LEVEL 3
        self.__enc_3 = XiEncoder(
            input_shape=input_shape,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=4,
            mid_channels=28,
            out_channels=64,
            with_upscaling=False,
        )
        self.__conv_block_3 = LevelConvolutionsBlock(64 + 8)

        # LEVEL 4
        self.__enc_4 = XiEncoder(
            input_shape=input_shape,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=5,
            mid_channels=32,
            out_channels=96,
            with_upscaling=False,
        )
        self.__conv_block_4 = LevelConvolutionsBlock(96)

        self.__upsizing_4 = UpsizingBlock(8, 8)
        self.__upsizing_3 = UpsizingBlock(8, 8)
        self.__upsizing_2 = UpsizingBlock(8, 8)

        self.apply(xavier_init)

    def __level_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gives back the disparities for left images on the 0th channel and for the right images on the 1th channel.
            - `x`: torch.Tensor[B,C,W,H]

        Returns a torch.Tensor[B,2,W,H]
        """
        return 0.3 * torch.sigmoid(x[:, :2, :, :])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Level's starting blocks
        conv1 = self.__enc_1(x)
        conv2 = self.__enc_2(x)
        conv3 = self.__enc_3(x)
        conv4 = self.__enc_4(x)

        # LEVEL 4
        conv4b = self.__conv_block_4(conv4)
        disp4 = self.__level_activations(conv4b)

        conv4b = self.__upsizing_4(conv4b)

        # LEVEL 3
        concat3 = torch.cat((conv3, conv4b), 1)
        conv3b = self.__conv_block_3(concat3)
        disp3 = self.__level_activations(conv3b)

        conv3b = self.__upsizing_3(conv3b)

        # LEVEL 2
        concat2 = torch.cat((conv2, conv3b), 1)
        conv2b = self.__conv_block_2(concat2)
        disp2 = self.__level_activations(conv2b)

        conv2b = self.__upsizing_2(conv2b)

        # LEVEL 1
        sa_conv1 = self.__lsa_1(conv1)
        concat1 = torch.cat((sa_conv1, conv2b), 1)
        conv1b = self.__conv_block_1(concat1)
        disp1 = self.__level_activations(conv1b)

        return [disp1, disp2, disp3, disp4]

    @staticmethod
    def scale_pyramid(img_batch: torch.Tensor, num_scales: int = 3) -> torch.Tensor:
        """
        It scales the batch of images to `num_scales` scales, everytime dividing by 2 the height and the width.
            - `img_batch`: torch.Tensor[B,C,H,W]
            - `num_scales`: int

        Returns a List[torch.Tensor[B,C,H,W]] with a length of `num_scales`.
        """
        scaled_imgs = []
        _, _, h, w = img_batch.shape
        for i in range(num_scales):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_img = nn.functional.interpolate(
                img_batch, size=(nh, nw), mode="area"
            )
            scaled_imgs.append(scaled_img)
        return scaled_imgs

    @staticmethod
    def upscale_img(
        img_tensor: torch.Tensor, new_2d_size: tuple[int, int] = (256, 512)
    ) -> torch.Tensor:
        """
        It upscales the `img_tensor` using bilinear interpolation to the specified size `new_2d_size`.
            - `img_tensor`: torch.Tensor[B,C,H,W];
            - `new_2d_size`: tuple(width,height).

        Returns a torch.Tensor[B,C,width, height].
        """
        return F.interpolate(
            img_tensor, size=new_2d_size, mode="bilinear", align_corners=True
        )


class PyXiNetM2(nn.Module):
    def __init__(self, config: ConfigHomeLab | ConfigCluster):
        super(PyXiNetM2, self).__init__()
        input_shape = [3, config.image_height, config.image_width]

        # LEVEL 1
        self.__enc_1 = DownsizingBlock(3, 16)
        self.__conv_block_1 = LevelConvolutionsBlock(16 + 8)
        self.__lsa_1 = LightSelfAttentionCNN1(16 + 8, 128, 256)

        # LEVEL 2
        self.__enc_2 = XiEncoder(
            input_shape=input_shape,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=3,
            mid_channels=24,
            out_channels=32,
            with_upscaling=False,
        )
        self.__conv_block_2 = LevelConvolutionsBlock(32 + 8)

        # LEVEL 3
        self.__enc_3 = XiEncoder(
            input_shape=input_shape,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=4,
            mid_channels=28,
            out_channels=64,
            with_upscaling=False,
        )
        self.__conv_block_3 = LevelConvolutionsBlock(64 + 8)

        # LEVEL 4
        self.__enc_4 = XiEncoder(
            input_shape=input_shape,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=5,
            mid_channels=32,
            out_channels=96,
            with_upscaling=False,
        )
        self.__conv_block_4 = LevelConvolutionsBlock(96)

        self.__upsizing_4 = UpsizingBlock(8, 8)
        self.__upsizing_3 = UpsizingBlock(8, 8)
        self.__upsizing_2 = UpsizingBlock(8, 8)

        self.apply(xavier_init)

    def __level_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gives back the disparities for left images on the 0th channel and for the right images on the 1th channel.
            - `x`: torch.Tensor[B,C,W,H]

        Returns a torch.Tensor[B,2,W,H]
        """
        return 0.3 * torch.sigmoid(x[:, :2, :, :])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Level's starting blocks
        conv1 = self.__enc_1(x)
        conv2 = self.__enc_2(x)
        conv3 = self.__enc_3(x)
        conv4 = self.__enc_4(x)

        # LEVEL 4
        conv4b = self.__conv_block_4(conv4)
        disp4 = self.__level_activations(conv4b)

        conv4b = self.__upsizing_4(conv4b)

        # LEVEL 3
        concat3 = torch.cat((conv3, conv4b), 1)
        conv3b = self.__conv_block_3(concat3)
        disp3 = self.__level_activations(conv3b)

        conv3b = self.__upsizing_3(conv3b)

        # LEVEL 2
        concat2 = torch.cat((conv2, conv3b), 1)
        conv2b = self.__conv_block_2(concat2)
        disp2 = self.__level_activations(conv2b)

        conv2b = self.__upsizing_2(conv2b)

        # LEVEL 1
        concat1 = torch.cat((conv1, conv2b), 1)
        sa_concat1 = self.__lsa_1(concat1)
        conv1b = self.__conv_block_1(sa_concat1)
        disp1 = self.__level_activations(conv1b)

        return [disp1, disp2, disp3, disp4]

    @staticmethod
    def scale_pyramid(img_batch: torch.Tensor, num_scales: int = 3) -> torch.Tensor:
        """
        It scales the batch of images to `num_scales` scales, everytime dividing by 2 the height and the width.
            - `img_batch`: torch.Tensor[B,C,H,W]
            - `num_scales`: int

        Returns a List[torch.Tensor[B,C,H,W]] with a length of `num_scales`.
        """
        scaled_imgs = []
        _, _, h, w = img_batch.shape
        for i in range(num_scales):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_img = nn.functional.interpolate(
                img_batch, size=(nh, nw), mode="area"
            )
            scaled_imgs.append(scaled_img)
        return scaled_imgs

    @staticmethod
    def upscale_img(
        img_tensor: torch.Tensor, new_2d_size: tuple[int, int] = (256, 512)
    ) -> torch.Tensor:
        """
        It upscales the `img_tensor` using bilinear interpolation to the specified size `new_2d_size`.
            - `img_tensor`: torch.Tensor[B,C,H,W];
            - `new_2d_size`: tuple(width,height).

        Returns a torch.Tensor[B,C,width, height].
        """
        return F.interpolate(
            img_tensor, size=new_2d_size, mode="bilinear", align_corners=True
        )


class PyXiNetM3(nn.Module):
    def __init__(self, config: ConfigHomeLab | ConfigCluster):
        super(PyXiNetM3, self).__init__()
        input_shape = [3, config.image_height, config.image_width]

        # LEVEL 1
        self.__enc_1 = DownsizingBlock(3, 16)
        self.__lsa_1 = LightSelfAttentionCNN2(16, 128, 256)
        self.__conv_block_1 = LevelConvolutionsBlock(16 + 8)

        # LEVEL 2
        self.__enc_2 = XiEncoder(
            input_shape=input_shape,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=3,
            mid_channels=24,
            out_channels=32,
            with_upscaling=False,
        )
        self.__conv_block_2 = LevelConvolutionsBlock(32 + 8)

        # LEVEL 3
        self.__enc_3 = XiEncoder(
            input_shape=input_shape,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=4,
            mid_channels=28,
            out_channels=64,
            with_upscaling=False,
        )
        self.__conv_block_3 = LevelConvolutionsBlock(64 + 8)

        # LEVEL 4
        self.__enc_4 = XiEncoder(
            input_shape=input_shape,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=5,
            mid_channels=32,
            out_channels=96,
            with_upscaling=False,
        )
        self.__conv_block_4 = LevelConvolutionsBlock(96)

        self.__upsizing_4 = UpsizingBlock(8, 8)
        self.__upsizing_3 = UpsizingBlock(8, 8)
        self.__upsizing_2 = UpsizingBlock(8, 8)

        self.apply(xavier_init)

    def __level_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gives back the disparities for left images on the 0th channel and for the right images on the 1th channel.
            - `x`: torch.Tensor[B,C,W,H]

        Returns a torch.Tensor[B,2,W,H]
        """
        return 0.3 * torch.sigmoid(x[:, :2, :, :])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Level's starting blocks
        conv1 = self.__enc_1(x)
        conv2 = self.__enc_2(x)
        conv3 = self.__enc_3(x)
        conv4 = self.__enc_4(x)

        # LEVEL 4
        conv4b = self.__conv_block_4(conv4)
        disp4 = self.__level_activations(conv4b)

        conv4b = self.__upsizing_4(conv4b)

        # LEVEL 3
        concat3 = torch.cat((conv3, conv4b), 1)
        conv3b = self.__conv_block_3(concat3)
        disp3 = self.__level_activations(conv3b)

        conv3b = self.__upsizing_3(conv3b)

        # LEVEL 2
        concat2 = torch.cat((conv2, conv3b), 1)
        conv2b = self.__conv_block_2(concat2)
        disp2 = self.__level_activations(conv2b)

        conv2b = self.__upsizing_2(conv2b)

        # LEVEL 1
        sa_conv1 = self.__lsa_1(conv1)
        concat1 = torch.cat((sa_conv1, conv2b), 1)
        conv1b = self.__conv_block_1(concat1)
        disp1 = self.__level_activations(conv1b)

        return [disp1, disp2, disp3, disp4]

    @staticmethod
    def scale_pyramid(img_batch: torch.Tensor, num_scales: int = 3) -> torch.Tensor:
        """
        It scales the batch of images to `num_scales` scales, everytime dividing by 2 the height and the width.
            - `img_batch`: torch.Tensor[B,C,H,W]
            - `num_scales`: int

        Returns a List[torch.Tensor[B,C,H,W]] with a length of `num_scales`.
        """
        scaled_imgs = []
        _, _, h, w = img_batch.shape
        for i in range(num_scales):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_img = nn.functional.interpolate(
                img_batch, size=(nh, nw), mode="area"
            )
            scaled_imgs.append(scaled_img)
        return scaled_imgs

    @staticmethod
    def upscale_img(
        img_tensor: torch.Tensor, new_2d_size: tuple[int, int] = (256, 512)
    ) -> torch.Tensor:
        """
        It upscales the `img_tensor` using bilinear interpolation to the specified size `new_2d_size`.
            - `img_tensor`: torch.Tensor[B,C,H,W];
            - `new_2d_size`: tuple(width,height).

        Returns a torch.Tensor[B,C,width, height].
        """
        return F.interpolate(
            img_tensor, size=new_2d_size, mode="bilinear", align_corners=True
        )


class PyXiNetM4(nn.Module):
    def __init__(self, config: ConfigHomeLab | ConfigCluster):
        super(PyXiNetM4, self).__init__()
        input_shape = [3, config.image_height, config.image_width]

        # LEVEL 1
        self.__enc_1 = DownsizingBlock(3, 16)
        self.__conv_block_1 = LevelConvolutionsBlock(16 + 8)
        self.__lsa_1 = LightSelfAttentionCNN2(16 + 8, 128, 256)

        # LEVEL 2
        self.__enc_2 = XiEncoder(
            input_shape=input_shape,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=3,
            mid_channels=24,
            out_channels=32,
            with_upscaling=False,
        )
        self.__conv_block_2 = LevelConvolutionsBlock(32 + 8)

        # LEVEL 3
        self.__enc_3 = XiEncoder(
            input_shape=input_shape,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=4,
            mid_channels=28,
            out_channels=64,
            with_upscaling=False,
        )
        self.__conv_block_3 = LevelConvolutionsBlock(64 + 8)

        # LEVEL 4
        self.__enc_4 = XiEncoder(
            input_shape=input_shape,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=5,
            mid_channels=32,
            out_channels=96,
            with_upscaling=False,
        )
        self.__conv_block_4 = LevelConvolutionsBlock(96)

        self.__upsizing_4 = UpsizingBlock(8, 8)
        self.__upsizing_3 = UpsizingBlock(8, 8)
        self.__upsizing_2 = UpsizingBlock(8, 8)

        self.apply(xavier_init)

    def __level_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gives back the disparities for left images on the 0th channel and for the right images on the 1th channel.
            - `x`: torch.Tensor[B,C,W,H]

        Returns a torch.Tensor[B,2,W,H]
        """
        return 0.3 * torch.sigmoid(x[:, :2, :, :])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Level's starting blocks
        conv1 = self.__enc_1(x)
        conv2 = self.__enc_2(x)
        conv3 = self.__enc_3(x)
        conv4 = self.__enc_4(x)

        # LEVEL 4
        conv4b = self.__conv_block_4(conv4)
        disp4 = self.__level_activations(conv4b)

        conv4b = self.__upsizing_4(conv4b)

        # LEVEL 3
        concat3 = torch.cat((conv3, conv4b), 1)
        conv3b = self.__conv_block_3(concat3)
        disp3 = self.__level_activations(conv3b)

        conv3b = self.__upsizing_3(conv3b)

        # LEVEL 2
        concat2 = torch.cat((conv2, conv3b), 1)
        conv2b = self.__conv_block_2(concat2)
        disp2 = self.__level_activations(conv2b)

        conv2b = self.__upsizing_2(conv2b)

        # LEVEL 1
        concat1 = torch.cat((conv1, conv2b), 1)
        sa_concat1 = self.__lsa_1(concat1)
        conv1b = self.__conv_block_1(sa_concat1)
        disp1 = self.__level_activations(conv1b)

        return [disp1, disp2, disp3, disp4]

    @staticmethod
    def scale_pyramid(img_batch: torch.Tensor, num_scales: int = 3) -> torch.Tensor:
        """
        It scales the batch of images to `num_scales` scales, everytime dividing by 2 the height and the width.
            - `img_batch`: torch.Tensor[B,C,H,W]
            - `num_scales`: int

        Returns a List[torch.Tensor[B,C,H,W]] with a length of `num_scales`.
        """
        scaled_imgs = []
        _, _, h, w = img_batch.shape
        for i in range(num_scales):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_img = nn.functional.interpolate(
                img_batch, size=(nh, nw), mode="area"
            )
            scaled_imgs.append(scaled_img)
        return scaled_imgs

    @staticmethod
    def upscale_img(
        img_tensor: torch.Tensor, new_2d_size: tuple[int, int] = (256, 512)
    ) -> torch.Tensor:
        """
        It upscales the `img_tensor` using bilinear interpolation to the specified size `new_2d_size`.
            - `img_tensor`: torch.Tensor[B,C,H,W];
            - `new_2d_size`: tuple(width,height).

        Returns a torch.Tensor[B,C,width, height].
        """
        return F.interpolate(
            img_tensor, size=new_2d_size, mode="bilinear", align_corners=True
        )


class PyXiNetBCBAM1(nn.Module):
    def __init__(self, config: ConfigHomeLab | ConfigCluster):
        super(PyXiNetBCBAM1, self).__init__()
        input_shape = [3, config.image_height, config.image_width]

        # LEVEL 1
        self.__enc_1 = DownsizingBlock(3, 16)
        self.__conv_block_1 = LevelCBAMBlock(16 + 8)

        # LEVEL 2
        self.__enc_2 = XiEncoder(
            input_shape=input_shape,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=4,
            mid_channels=24,
            out_channels=32,
        )
        self.__conv_block_2 = LevelConvolutionsBlock(32 + 8)

        # LEVEL 3
        self.__enc_3 = XiEncoder(
            input_shape=input_shape,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=5,
            mid_channels=28,
            out_channels=64,
        )
        self.__conv_block_3 = LevelConvolutionsBlock(64 + 8)

        # LEVEL 4
        self.__enc_4 = DownsizingBlock(64, 96)
        self.__conv_block_4 = LevelConvolutionsBlock(96)

        self.__upsizing_4 = UpsizingBlock(8, 8)
        self.__upsizing_3 = UpsizingBlock(8, 8)
        self.__upsizing_2 = UpsizingBlock(8, 8)

        self.apply(xavier_init)

    def __level_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gives back the disparities for left images on the 0th channel and for the right images on the 1th channel.
            - `x`: torch.Tensor[B,C,W,H]

        Returns a torch.Tensor[B,2,W,H]
        """
        return 0.3 * torch.sigmoid(x[:, :2, :, :])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Level's starting blocks
        conv1 = self.__enc_1(x)
        conv2 = self.__enc_2(x)
        conv3 = self.__enc_3(x)
        conv4 = self.__enc_4(conv3)

        # LEVEL 4
        conv4b = self.__conv_block_4(conv4)
        disp4 = self.__level_activations(conv4b)

        conv4b = self.__upsizing_4(conv4b)

        # LEVEL 3
        concat3 = torch.cat((conv3, conv4b), 1)
        conv3b = self.__conv_block_3(concat3)
        disp3 = self.__level_activations(conv3b)

        conv3b = self.__upsizing_3(conv3b)

        # LEVEL 2
        concat2 = torch.cat((conv2, conv3b), 1)
        conv2b = self.__conv_block_2(concat2)
        disp2 = self.__level_activations(conv2b)

        conv2b = self.__upsizing_2(conv2b)

        # LEVEL 1
        concat1 = torch.cat((conv1, conv2b), 1)
        conv1b = self.__conv_block_1(concat1)
        disp1 = self.__level_activations(conv1b)

        return [
            disp1,
            disp2,
            disp3,
            disp4,
        ]

    @staticmethod
    def scale_pyramid(img_batch: torch.Tensor, num_scales: int = 3) -> torch.Tensor:
        """
        It scales the batch of images to `num_scales` scales, everytime dividing by 2 the height and the width.
            - `img_batch`: torch.Tensor[B,C,H,W]
            - `num_scales`: int

        Returns a List[torch.Tensor[B,C,H,W]] with a length of `num_scales`.
        """
        scaled_imgs = []
        _, _, h, w = img_batch.shape
        for i in range(num_scales):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_img = nn.functional.interpolate(
                img_batch, size=(nh, nw), mode="area"
            )
            scaled_imgs.append(scaled_img)
        return scaled_imgs

    @staticmethod
    def upscale_img(
        img_tensor: torch.Tensor, new_2d_size: tuple[int, int] = (256, 512)
    ) -> torch.Tensor:
        """
        It upscales the `img_tensor` using bilinear interpolation to the specified size `new_2d_size`.
            - `img_tensor`: torch.Tensor[B,C,H,W];
            - `new_2d_size`: tuple(width,height).

        Returns a torch.Tensor[B,C,width, height].
        """
        return F.interpolate(
            img_tensor, size=new_2d_size, mode="bilinear", align_corners=True
        )


class PyXiNetBCBAM2(nn.Module):
    def __init__(self, config: ConfigHomeLab | ConfigCluster):
        super(PyXiNetBCBAM2, self).__init__()
        input_shape = [3, config.image_height, config.image_width]

        # LEVEL 1
        self.__enc_1 = DownsizingBlock(3, 16)
        self.__conv_block_1 = LevelCBAMBlock(16 + 8)

        # LEVEL 2
        self.__enc_2 = XiEncoder(
            input_shape=input_shape,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=3,
            mid_channels=24,
            out_channels=32,
            with_upscaling=False,
        )
        self.__conv_block_2 = LevelConvolutionsBlock(32 + 8)

        # LEVEL 3
        self.__enc_3 = XiEncoder(
            input_shape=input_shape,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=4,
            mid_channels=28,
            out_channels=64,
            with_upscaling=False,
        )
        self.__conv_block_3 = LevelConvolutionsBlock(64 + 8)

        # LEVEL 4
        self.__enc_4 = XiEncoder(
            input_shape=input_shape,
            alpha=config.alpha,
            gamma=config.gamma,
            num_layers=5,
            mid_channels=32,
            out_channels=96,
            with_upscaling=False,
        )
        self.__conv_block_4 = LevelConvolutionsBlock(96)

        self.__upsizing_4 = UpsizingBlock(8, 8)
        self.__upsizing_3 = UpsizingBlock(8, 8)
        self.__upsizing_2 = UpsizingBlock(8, 8)

        self.apply(xavier_init)

    def __level_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gives back the disparities for left images on the 0th channel and for the right images on the 1th channel.
            - `x`: torch.Tensor[B,C,W,H]

        Returns a torch.Tensor[B,2,W,H]
        """
        return 0.3 * torch.sigmoid(x[:, :2, :, :])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Level's starting blocks
        conv1 = self.__enc_1(x)
        conv2 = self.__enc_2(x)
        conv3 = self.__enc_3(x)
        conv4 = self.__enc_4(x)

        # LEVEL 4
        conv4b = self.__conv_block_4(conv4)
        disp4 = self.__level_activations(conv4b)

        conv4b = self.__upsizing_4(conv4b)

        # LEVEL 3
        concat3 = torch.cat((conv3, conv4b), 1)
        conv3b = self.__conv_block_3(concat3)
        disp3 = self.__level_activations(conv3b)

        conv3b = self.__upsizing_3(conv3b)

        # LEVEL 2
        concat2 = torch.cat((conv2, conv3b), 1)
        conv2b = self.__conv_block_2(concat2)
        disp2 = self.__level_activations(conv2b)

        conv2b = self.__upsizing_2(conv2b)

        # LEVEL 1
        concat1 = torch.cat((conv1, conv2b), 1)
        conv1b = self.__conv_block_1(concat1)
        disp1 = self.__level_activations(conv1b)

        return [disp1, disp2, disp3, disp4]

    @staticmethod
    def scale_pyramid(img_batch: torch.Tensor, num_scales: int = 3) -> torch.Tensor:
        """
        It scales the batch of images to `num_scales` scales, everytime dividing by 2 the height and the width.
            - `img_batch`: torch.Tensor[B,C,H,W]
            - `num_scales`: int

        Returns a List[torch.Tensor[B,C,H,W]] with a length of `num_scales`.
        """
        scaled_imgs = []
        _, _, h, w = img_batch.shape
        for i in range(num_scales):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_img = nn.functional.interpolate(
                img_batch, size=(nh, nw), mode="area"
            )
            scaled_imgs.append(scaled_img)
        return scaled_imgs

    @staticmethod
    def upscale_img(
        img_tensor: torch.Tensor, new_2d_size: tuple[int, int] = (256, 512)
    ) -> torch.Tensor:
        """
        It upscales the `img_tensor` using bilinear interpolation to the specified size `new_2d_size`.
            - `img_tensor`: torch.Tensor[B,C,H,W];
            - `new_2d_size`: tuple(width,height).

        Returns a torch.Tensor[B,C,width, height].
        """
        return F.interpolate(
            img_tensor, size=new_2d_size, mode="bilinear", align_corners=True
        )


class PydnetC1(nn.Module):
    def __init__(self):
        super(PydnetC1, self).__init__()

        # LEVEL 1
        self.__downsizing_block_1 = DownsizingCBAMBlock(3, 16)
        self.__conv_block_1 = LevelCBAMBlock(16 + 8)

        # LEVEL 2
        self.__downsizing_block_2 = DownsizingCBAMBlock(16, 32)
        self.__conv_block_2 = LevelCBAMBlock(32 + 8)

        # LEVEL 3
        self.__downsizing_block_3 = DownsizingCBAMBlock(32, 64)
        self.__conv_block_3 = LevelCBAMBlock(64 + 8)

        # LEVEL 4
        self.__downsizing_block_4 = DownsizingCBAMBlock(64, 96)
        self.__conv_block_4 = LevelCBAMBlock(96)

        self.__upsizing_4 = UpsizingBlock(8, 8)
        self.__upsizing_3 = UpsizingBlock(8, 8)
        self.__upsizing_2 = UpsizingBlock(8, 8)

        self.apply(xavier_init)

    def __level_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gives back the disparities for left images on the 0th channel and for the right images on the 1th channel.
            - `x`: torch.Tensor[B,C,W,H]

        Returns a torch.Tensor[B,2,W,H]
        """
        return 0.3 * torch.sigmoid(x[:, :2, :, :])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Level's starting blocks
        conv1 = self.__downsizing_block_1(x)  # [8, 16, 128, 256]
        conv2 = self.__downsizing_block_2(conv1)  # [8, 32, 64, 128]
        conv3 = self.__downsizing_block_3(conv2)  # [8, 64, 32, 64]
        conv4 = self.__downsizing_block_4(conv3)  # [8, 96, 16, 32]

        # LEVEL 4
        conv4b = self.__conv_block_4(conv4)
        disp4 = self.__level_activations(conv4b)

        conv4b = self.__upsizing_4(conv4b)

        # LEVEL 3
        concat3 = torch.cat((conv3, conv4b), 1)
        conv3b = self.__conv_block_3(concat3)
        disp3 = self.__level_activations(conv3b)

        conv3b = self.__upsizing_3(conv3b)

        # LEVEL 2
        concat2 = torch.cat((conv2, conv3b), 1)
        conv2b = self.__conv_block_2(concat2)
        disp2 = self.__level_activations(conv2b)

        conv2b = self.__upsizing_2(conv2b)

        # LEVEL 1
        concat1 = torch.cat((conv1, conv2b), 1)
        conv1b = self.__conv_block_1(concat1)
        disp1 = self.__level_activations(conv1b)

        return [
            disp1,
            disp2,
            disp3,
            disp4,
        ]

    @staticmethod
    def scale_pyramid(img_batch: torch.Tensor, num_scales: int = 4) -> torch.Tensor:
        """
        It scales the batch of images to `num_scales` scales, everytime dividing by 2 the height and the width.
            - `img_batch`: torch.Tensor[B,C,H,W]
            - `num_scales`: int

        Returns a List[torch.Tensor[B,C,H,W]] with a length of `num_scales`.
        """
        scaled_imgs = []
        _, _, h, w = img_batch.shape
        for i in range(num_scales):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_img = nn.functional.interpolate(
                img_batch, size=(nh, nw), mode="area"
            )
            scaled_imgs.append(scaled_img)
        return scaled_imgs

    @staticmethod
    def upscale_img(
        img_tensor: torch.Tensor, new_2d_size: tuple[int, int] = (256, 512)
    ) -> torch.Tensor:
        """
        It upscales the `img_tensor` using bilinear interpolation to the specified size `new_2d_size`.
            - `img_tensor`: torch.Tensor[B,C,H,W];
            - `new_2d_size`: tuple(width,height).

        Returns a torch.Tensor[B,C,width, height].
        """
        return F.interpolate(
            img_tensor, size=new_2d_size, mode="bilinear", align_corners=True
        )


# xe = XiEncoder([3, 256, 512], 0.4, 4, 4, 64, 64)
# print(sum(p.numel() for p in xe.parameters()))
# print(xe(torch.rand([8, 3, 256, 512])).size())
# xe = PyXiNetB4(ConfigHomeLab())
# print(sum(p.numel() for p in xe.parameters()))
# print(xe(torch.rand([8, 3, 256, 512]))[0].size())
# xe = PyXiNetBCBAM1(ConfigHomeLab())
# print(sum(p.numel() for p in xe.parameters()))
# print(xe(torch.rand([8, 3, 256, 512]))[0].size())
#xe = PydnetC1()
#print(sum(p.numel() for p in xe.parameters()))
#print(xe(torch.rand([9, 3, 256, 512]))[0].size())
