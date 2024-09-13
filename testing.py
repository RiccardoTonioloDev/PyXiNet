import torch
import numpy as np
import os
from typing import List, Literal
from KittiDataset import KittiDataset
from PyXiNet import PyXiNetA1
from Losses import L_total, generate_image_left, generate_image_right
from Configs.ConfigCluster import ConfigCluster
from Configs.ConfigHomeLab import ConfigHomeLab
from Config import Config
from using import use


def evaluate_on_test_set(
    dataset: KittiDataset,
    config: ConfigHomeLab | ConfigCluster,
    model: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """
    It performs an evaluation of the model on the test set.
        - `dataset`: test dataset;
        - `config`: environment configuration;
        - `model`: model used for the evaluation process;
        - `device`: it should be the same device of the model.

    Returns the scalar tensor representing the total loss on the test set.
    """
    model.eval()
    test_dataloader = dataset.make_dataloader(config.batch_size, False)
    total_loss_tower = []
    with torch.no_grad():
        for i, (left_img_batch, right_img_batch) in enumerate(test_dataloader):
            left_img_batch, right_img_batch = left_img_batch.to(
                device
            ), right_img_batch.to(device)

            # Model input
            model_output: List[torch.Tensor] = model(left_img_batch)
            left_disp_pyramid = [mo[:, 0, :, :].unsqueeze(1) for mo in model_output]
            right_disp_pyramid = [mo[:, 1, :, :].unsqueeze(1) for mo in model_output]

            # Creating pyramid of various resolutions for left and right image batches
            left_img_batch_pyramid = model.scale_pyramid(left_img_batch)  # [B, C, H, W]
            right_img_batch_pyramid = model.scale_pyramid(
                right_img_batch
            )  # [B, C, H, W]

            # Using disparities to generate corresponding left and right warped image batches (at various resolutions)
            est_batch_pyramid_left = [
                generate_image_left(img, disp)
                for img, disp in zip(right_img_batch_pyramid, left_disp_pyramid)
            ]
            est_batch_pyramid_right = [
                generate_image_right(img, disp)
                for img, disp in zip(left_img_batch_pyramid, right_disp_pyramid)
            ]
            # Calculating the loss based on the total loss function
            total_loss, _, _, _ = L_total(
                est_batch_pyramid_left,
                est_batch_pyramid_right,
                left_img_batch_pyramid,
                right_img_batch_pyramid,
                left_disp_pyramid,
                right_disp_pyramid,
                weight_SSIM=config.weight_SSIM,
                weight_ap=config.weight_ap,
                weight_lr=config.weight_lr,
                weight_df=config.weight_df,
            )
            total_loss_tower.append(total_loss)
        tower_len = len(total_loss_tower)
        avg_loss = sum(total_loss_tower) / tower_len
        return avg_loss


def generate_test_disparities(
    env: Literal["HomeLab", "Cluster"], model: torch.nn.Module
) -> None:
    """
    It generates the `disparities.npy` file that will be used for the true evaluation step.
        - `env`: the selected environment to select the right configuration;
        - `model`: the model that will be used to generate the various disparities.
    """
    # Configurations and checks
    config = Config(env).get_configuration()
    if config.checkpoint_to_use_path == None or config.checkpoint_to_use_path == "":
        print("You have to select a checkpoint to create the disparities file.")
        exit(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model creation and configuration
    model = model.to(device)
    checkpoint = torch.load(config.checkpoint_to_use_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Data
    test_dataset = KittiDataset(
        config.data_path,
        config.filenames_file_testing,
        config.image_width,
        config.image_height,
        Config(env),
        mode="test",
    )
    test_dataloader = test_dataset.make_dataloader(1, shuffle_batch=False)
    num_of_samples = len(test_dataset)

    # Disparities array creation
    disparities = np.zeros((num_of_samples, 256, 512), dtype=np.float32)

    print(f"Calculating disparities for {num_of_samples} samples.")
    # Populating disparities array
    for i, left_img in enumerate(test_dataloader):
        print("Doing image #", i + 1)
        left_img: torch.Tensor = left_img.to(device)
        with torch.no_grad():
            if True:
                # without post processing
                disp: torch.Tensor = model(left_img)[0][:, 0, :, :].unsqueeze(1)
                upscaled_disparities = PyXiNetA1.upscale_img(disp, (256, 512))
                disparities[i] = upscaled_disparities[0, 0, :, :].cpu().numpy()
            else:
                # with post processing
                disp = use(model, left_img, 512, 256, 512, 256, device)
                disparities[i] = disp.cpu().numpy()
    print("Computing complete.")
    print("Saving disparities.")
    output_directory = "."
    if config.output_directory != None and config.output_directory != "":
        output_directory = config.output_directory
    np.save(os.path.join(output_directory, "disparities.npy"), disparities)
    print("Disparities saved.")
