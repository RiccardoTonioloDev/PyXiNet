import os
import time
from typing import List, Literal
import torch
import torch.utils
import torch.utils.data
import wandb
import torch.optim as optim
from KittiDataset import KittiDataset
from PyXiNet import PyXiNetA1
from Config import Config
from Losses import L_total, generate_image_left, generate_image_right
from testing import evaluate_on_test_set


def train(env: Literal["HomeLab", "Cluster"], model: torch.nn.Module) -> None:
    """
    Function used to train the model.
        - `env`: is the selected configuration that will be use to configure the model, the dataset, the optimizer,
            the checkpoint logic and the training process;
        - `model`: the model that will be used in the training process.
    """
    # Configurations
    config = Config(env).get_configuration()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuring wandb
    wandb.init(
        project=config.model_name,
        config={
            "num_epochs": config.num_epochs,
            "learning_rate": config.learning_rate,
        },
    )

    # Data
    train_dataset = KittiDataset(
        config.data_path,
        config.filenames_file_training,
        config.image_width,
        config.image_height,
        Config(env),
    )
    test_dataset = KittiDataset(
        config.data_path,
        config.filenames_file_testing,
        config.image_width,
        config.image_height,
        Config(env),
    )
    train_dataloader = train_dataset.make_dataloader(
        config.batch_size, config.shuffle_batch
    )

    # Model configuration
    model = model.to(device)
    num_of_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: ", num_of_params)

    # Optimizer creation and configuration
    optimizer = optim.Adam(
        model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-8
    )

    # Dynamic learning rate configuration
    start_epoch = 0
    num_epochs = config.num_epochs
    lr_lambda = lambda epoch: (
        0.5 ** (1 + int(epoch >= 0.8 * num_epochs)) if epoch >= 0.6 * num_epochs else 1
    )
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Stats to be used for logging
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * num_epochs
    start_time = time.time() / 3600  # in hours

    # Variables to be used to manage checkpoints
    is_first_evaluation = True
    min_loss = torch.Tensor([0]).to(device)
    checkpoint_files = []

    # Checkpoint loading for training if set to true
    if (
        config.retrain == False
        and config.checkpoint_to_use_path != None
        and config.checkpoint_to_use_path != ""
    ):
        print("Retraining disabled. Loading checkpoint.")
        checkpoint = torch.load(config.checkpoint_to_use_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Fine tuning started.")

    # Training cycle
    for epoch in range(start_epoch, num_epochs):
        model.train()
        for i, (left_img_batch, right_img_batch) in enumerate(train_dataloader):
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
            total_loss, img_loss, disp_grad_loss, lr_loss = L_total(
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
            # Doing the forward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            wandb.log(
                {
                    "image_loss": img_loss.item(),
                    "disp_gradient_loss": disp_grad_loss.item(),
                    "lr_loss": lr_loss.item(),
                    "total_loss": total_loss.item(),
                }
            )
            steps_done = i + epoch * steps_per_epoch

            if steps_done % 100 == 0 and i != 0:
                elapsed_time = (time.time() / 3600) - start_time  # in hours
                steps_to_do = total_steps - steps_done
                time_remaining = (elapsed_time / steps_done) * steps_to_do
                # Logging stats
                print(
                    f"Epoch [{epoch+1}/{num_epochs}]| Steps: {steps_done}| Loss: {total_loss.item():.4f}| Learning rate: {config.learning_rate * lr_lambda(epoch)}| Elapsed time: {elapsed_time:.2f}h| Time to finish: ~{time_remaining}h|"
                )

        # Evaluation and checkpoint saving if the evaluation score is better
        eval_loss = evaluate_on_test_set(test_dataset, config, model, device)
        if is_first_evaluation or min_loss > eval_loss:
            is_first_evaluation = False
            min_loss = eval_loss

            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": total_loss,
            }
            # Saving checkpoint
            checkpoint_filename = os.path.join(
                config.checkpoint_path, f"checkpoint_e{epoch:03d}.pth.tar"
            )
            torch.save(state, checkpoint_filename)
            checkpoint_files.append(checkpoint_filename)

            # Remove older checkpoints
            for checkpoint_file in checkpoint_files[:-1]:
                if os.path.exists(checkpoint_file):
                    os.remove(checkpoint_file)
            checkpoint_files = checkpoint_files[-1:]
        print(f"Test loss: {eval_loss.item()}")

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": total_loss,
    }
    # Saving checkpoint of the last epoch
    checkpoint_filename = os.path.join(
        config.checkpoint_path, f"checkpoint_e{num_epochs:03d}.pth.tar"
    )
    torch.save(state, checkpoint_filename)
