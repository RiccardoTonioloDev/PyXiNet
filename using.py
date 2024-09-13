from typing import Literal
from PyXiNet import PyXiNetA1
from Config import Config
from KittiDataset import KittiDataset
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import time
import numpy as np

image_to_single_batch_tensor = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)


def tensor_resizer(
    tensor: torch.Tensor,
    width: int,
    height: int,
    mode: Literal["bicubic", "area"] = "area",
) -> torch.Tensor:
    """
    It resizes the tensor to specified width and height.
        - `tensor`: torch.Tensor[B,C,H,W];
        - `width`: output tensor width;
        - `height`: output tensor height;
        - `mode`: it's the selection of the interpolation method.

    Returns torch.Tensor[B,C,heigth,width]
    """
    tensor = tensor if tensor.size().__len__() > 3 else tensor.unsqueeze(0)
    tensor = F.interpolate(tensor, size=(height, width), mode=mode)
    return tensor


def from_image_to_tensor(img: Image.Image) -> torch.Tensor:
    """
    Converts a pillow image to a torch.Tensor
        - `img`: pillow image.

    Returns image torch.Tensor[1,C,H,W].
    """
    img = img.convert("RGB")
    img_tensor: torch.Tensor = image_to_single_batch_tensor(img).unsqueeze(0)
    return img_tensor


def post_process_disparity(disp: torch.Tensor) -> torch.Tensor:
    """
    It uses the entire disparity tensor, coming as the prediction of the model using the batch composed
    of the left and the horizontally-flipped-left image. It combines the two results to make a more accurate
    depth map.
        - `disp`: torch.Tensor[B,H,W].

    Returns the processed disparity tensor.
    """
    _, h, w = disp.shape
    l_disp = disp[0, :, :]
    r_disp = torch.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l = torch.Tensor(l).to(disp.device)
    l_mask = 1.0 - torch.clamp(20 * (l - 0.05), 0, 1)
    r_mask = torch.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def use_with_path(
    env: Literal["HomeLab", "Cluster"], img_path: str, model: torch.nn.Module
) -> None:
    """
    It makes use of the provided model, to make a depth map of the image located in the `img_path` path.
    The depth map will be saved in the same folder of the `img_path`.
        - `env`: environment configuration;
        - `img_path`: the path where the image il located;
        - `model`: the model that will be used to make the depth map.
    """
    # Configurations and checks
    config = Config(env).get_configuration()
    if config.checkpoint_to_use_path == None or config.checkpoint_to_use_path == "":
        print("You have to select a checkpoint to correctly configure the model.")
        exit(0)
    if img_path == None or img_path == "":
        print("You have to select an image to create the corresponding depth heatmap.")
        exit(0)
    folder = os.path.dirname(img_path)
    filename = os.path.basename(img_path)
    name, ext = os.path.splitext(filename)
    depth_map_filename = f"{name}_depth_map{ext}"
    depth_map_path = os.path.join(folder, depth_map_filename)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model configuration
    model = model.to(device)
    checkpoint = torch.load(
        config.checkpoint_to_use_path,
        map_location=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    try:
        with Image.open(img_path) as img:
            original_width, original_height = img.size
            start_time = time.time()
            disp_to_img: np.ndarray = (
                use(
                    model,
                    img,
                    config.image_width,
                    config.image_height,
                    original_width,
                    original_height,
                    device,
                )
                .cpu()
                .numpy()
            )
            elapsed_time = time.time() - start_time
            print(f"Execution time: {elapsed_time:.2f}s")
            # Salva l'immagine
            plt.imsave(depth_map_path, disp_to_img, cmap="plasma")

            print(f"Depth map salvata al seguente path:\n{depth_map_path}")

    except Exception as e:
        raise RuntimeError(f"Error loading image: {img_path}. {e}")


def use(
    model: torch.nn.Module,
    img: Image.Image | torch.Tensor,
    downscale_width: int,
    downscale_height: int,
    original_width: int,
    original_height: int,
    device: torch.device,
) -> torch.Tensor:
    """
    It will use a Pillow image as an input and provide a pillow depth map as an output.
        - `model`: the Pydnet model used;
        - `img`: it can be a pillow image or a torch.Tensor representation of an image;
        - `downscale_width`: it's the width that the model accepts in the input;
        - `downscale_height`: it's the height that the model accepts in the input;
        - `original_width`: it's the width of the output image;
        - `original_height`: it's the height of the output image;
        - `device`: the device where the model it's hosted.

    Returns the torch tensor of the depth map with size [H,W]
    """
    model.eval()
    model.to(device)
    img_tensor = img
    if isinstance(img, Image.Image):
        # In case of pillow image:
        # - It turns the image into a tensor with batch dimension of 1;
        # - It resizes it with the right dimensions.
        img_tensor = from_image_to_tensor(img).to(device)
        img_tensor = tensor_resizer(img_tensor, downscale_width, downscale_height)
    else:
        # In case of pillow image:
        # - It makes sure the tensor have the right dimensions;
        # - It resizes the tensor if necessary.
        img_tensor = img if len(img.shape) == 4 else img.unsqueeze(0)
        img_tensor = (
            img
            if downscale_height == img.shape[2] and downscale_width == img.shape[3]
            else tensor_resizer(img_tensor, downscale_width, downscale_height)
        )
    img_tensor_batch = KittiDataset.from_left_to_left_batch(img_tensor)
    with torch.no_grad():
        img_disparities: torch.Tensor = model(img_tensor_batch)[0][
            :, 0, :, :
        ]  # [2, H, W]
        img_disparities = (
            post_process_disparity(img_disparities)
            .unsqueeze(0)
            .unsqueeze(0)  # [1,1,H,W]
        )
        # img_disparities = tensor_resizer(
        #     img_disparities, original_width, original_height, mode="bicubic"
        # ).squeeze() TODO: Remove this if you find out that the code below is right
        img_disparities = (
            img_disparities
            if original_width == img_disparities.shape[2]
            and original_height == img_disparities.shape[3]
            else tensor_resizer(
                img_disparities, original_width, original_height, mode="bicubic"
            )
        ).squeeze()

    return img_disparities


def inference_time_avg_10(dir: str, model: torch.nn.Module, rsz_h: int, rsz_w: int):
    file_list = []
    if os.path.isdir(dir):
        file_list = [
            os.path.abspath(os.path.join(dir, f))
            for f in os.listdir(dir)
            if os.path.isfile(os.path.join(dir, f))
        ]
    else:
        raise Exception(
            f"The provided directory: {dir} is non existent or it's not a directory."
        )
    recorded_time = []
    for img_path in file_list:
        with Image.open(img_path) as img:
            original_width, original_height = img.size
            start_time = time.time()
            use(
                model,
                img,
                rsz_w,
                rsz_h,
                original_width,
                original_height,
                torch.device("cpu"),
            ).cpu().numpy()
            elapsed_time = time.time() - start_time
            recorded_time.append(elapsed_time)
    print(
        f"The avg on 10 images for inference time is: {(sum(recorded_time) / len(recorded_time)):.02f}s"
    )
