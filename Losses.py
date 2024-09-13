from typing import List, Tuple
import torch
from torch import nn
from torch.nn.functional import pad


def L1_loss(est_batch: torch.Tensor, img_batch: torch.Tensor) -> torch.Tensor:
    """
    # L1 diffence
    - `est_batch`: Tensor[B, C, H, W] The warped image batch;
    - `img_batch`: Tensor[B, C, H, W] The original image batch.
    """
    return torch.abs(est_batch - img_batch)  # [B, C, H, W]


def SSIM_loss(est_batch: torch.Tensor, img_batch: torch.Tensor) -> torch.Tensor:
    """
    # Structural Similarity Index Measure
    - `est_batch`: Tensor[B, C, H, W] The warped image batch;
    - `img_batch`: Tensor[B, C, H, W] The original image batch.
    """
    C1 = 0.01**2
    C2 = 0.03**2

    mus_est = nn.functional.avg_pool2d(est_batch, 3, 1, 0)  # [B, C, H - 2, W - 2]
    mu_img = nn.functional.avg_pool2d(img_batch, 3, 1, 0)  # [B, C, H - 2, W - 2]

    sigma_est = nn.functional.avg_pool2d(est_batch**2, 3, 1, 0) - mus_est**2
    # [B, C, H - 2, W - 2]
    sigma_img = nn.functional.avg_pool2d(img_batch**2, 3, 1, 0) - mu_img**2
    # [B, C, H - 2, W - 2]
    sigma_est_img = (
        nn.functional.avg_pool2d(est_batch * img_batch, 3, 1, 0) - mus_est * mu_img
    )  # [B, C, H - 2, W - 2]

    SSIM_n = (2 * mus_est * mu_img + C1) * (2 * sigma_est_img + C2)
    # [B, C, H - 2, W - 2]
    SSIM_d = (mus_est**2 + mu_img**2 + C1) * (sigma_est + sigma_img + C2)
    # [B, C, H - 2, W - 2]

    SSIM = SSIM_n / SSIM_d  # [B, C, H - 2, W - 2]

    return torch.clamp((1 - SSIM) / 2, 0, 1)  # [B, C, H - 2, W - 2]


def L_ap(
    est_batch_pyramid: List[torch.Tensor],
    img_batch_pyramid: List[torch.Tensor],
    alpha: float,
) -> torch.Tensor:
    """
    # Reconstruction error loss (otherwise called image loss)
        - `est_batch_pyramid`: List[Tensor[B, C, H, W]] Warped image batch at different scales;
        - `img_batch_pyramid`: List[Tensor[B, C, H, W]] Corresponding image batch at different scales.
    """
    SSIM_pyramid = [
        SSIM_loss(est, img).mean() * alpha
        for est, img in zip(est_batch_pyramid, img_batch_pyramid)
    ]  #  List[Tensor[B]]
    #  Doing the SSIM on each image of the batch at the same time, for each resolution
    L1_pyramid = [
        L1_loss(est, img).mean() * (1 - alpha)
        for est, img in zip(est_batch_pyramid, img_batch_pyramid)
    ]  # List[Tensor[B]]
    #  Doing the L1 on each image of the batch at the same time, for each resolution

    return sum(SSIM_pyramid + L1_pyramid)  #  Tensor[Scalar]


def gradient_x(tensor: torch.Tensor):
    """
    Generates the horizontal gradient tensor of the input tensor.
        - `tensor`: Tensor[B,C,H,W] The input tensor.

    Returns a Tensor[B,C,H,W-1]
    """
    gx = tensor[:, :, :, :-1] - tensor[:, :, :, 1:]
    return gx


def gradient_y(tensor: torch.Tensor):
    """
    Generates the vertical gradient tensor of the input tensor.
        - `tensor`: Tensor[B,C,H,W] The input tensor.

    Returns a Tensor[B,C,H-1,W]
    """
    gy = tensor[:, :, :-1, :] - tensor[:, :, 1:, :]
    return gy


def disparity_smoothness(
    disparity_batch: torch.Tensor, image_batch: torch.Tensor
) -> torch.Tensor:
    """
    Disparity smoothness
        - `disparity_batch`: Tensor[B, 1, H, W] (the channels dimension is externally artificially added).
            Normally it would be [B, H, W] because disparities are single numbers in a specific
            pixel location.
            This tensor represents the disparities for a specific side;
        - `image_batch`: Tensor[B, C, H, W] Images batch of the corresponding side.

    IMPORTANT: only the horizontal gradient has been calculated, as the original code makes only use
    of the horizontal gradient to compute disparity smoothness. I've left the code (commented) of the
    vertical gradient computation in case of necessity.

    Returns a scalar Tensor representing the disparity smoothness.
    """
    disparity_grad_x = torch.abs(gradient_x(disparity_batch))  # [B, 1, H, W - 1]
    # disparity_grad_y = torch.abs(gradient_y(disparity_batch))  # [B, 1, H - 1, W]

    image_grad_x = gradient_x(image_batch)  # [B, C, H, W - 1]
    # image_grad_y = gradient_y(image_batch)  # [B, C, H - 1, W]

    weights_x = torch.exp(
        -torch.mean(torch.abs(image_grad_x), 1, keepdim=True)
    )  # [B, 1, H, W - 1]
    # weights_y = torch.exp(
    #    -torch.mean(torch.abs(image_grad_y), 1, keepdim=True)
    # )  # [B, 1, H - 1, W]

    smoothness_x = disparity_grad_x * weights_x  # [B, 1, H, W - 1]
    # smoothness_y = disparity_grad_y * weights_y  # [B, 1, H - 1, W]

    return torch.mean(torch.abs(smoothness_x))  # Tensor[Scalar]


def L_df(
    disparity_batch_pyramid: List[torch.Tensor],
    image_batch_pyramid: List[torch.Tensor],
) -> torch.Tensor:
    """
    # Disparity gradient loss
        - `disparity_batch_pyramid`: List[Tensor[B, 1, H, W]] Calculated disparities for a specific side (at various resolutions);
        - `image_batch_pyramid`: List[Tensor[B, C, H, W]] Images batch of the corresponding side (at various resolutions).
    """
    return sum(
        [
            disparity_smoothness(disp, img) / 2**i
            for i, (disp, img) in enumerate(
                zip(disparity_batch_pyramid, image_batch_pyramid)
            )
        ]
    )


def bilinear_sampler_1d_h(
    input_images, x_offset, wrap_mode="border", tensor_type="torch.cuda.FloatTensor"
) -> torch.Tensor:
    """
    It does bilinear sampling on the `input_images` using the `x_offset` (disparities) for the coordinates.
        - `input_images`: the images that will be sampled;
        - `x_offset`: the disparities that will be used to sample the input images;
        - `torch.cuda.FloatTensor`: leave it as default.

    Returns torch.Tensor[B,C,H,W].
    """
    num_batch, num_channels, height, width = input_images.size()

    # Handle both texture border types
    edge_size = 0
    if wrap_mode == "border":
        edge_size = 1
        # Pad last and second-to-last dimensions by 1 from both sides
        input_images = pad(input_images, (1, 1, 1, 1))
    elif wrap_mode == "edge":
        edge_size = 0
    else:
        return None

    # Put channels to slowest dimension and flatten batch with respect to others
    input_images = input_images.permute(1, 0, 2, 3).contiguous()
    im_flat = input_images.reshape(num_channels, -1)

    # Create meshgrid for pixel indicies (PyTorch doesn't have dedicated
    # meshgrid function)
    x = torch.linspace(0, width - 1, width).repeat(height, 1).type(tensor_type)
    y = (
        torch.linspace(0, height - 1, height)
        .repeat(width, 1)
        .transpose(0, 1)
        .type(tensor_type)
    )
    # Take padding into account
    x = x + edge_size
    y = y + edge_size
    # Flatten and repeat for each image in the batch
    x = x.reshape(-1).repeat(1, num_batch)
    y = y.reshape(-1).repeat(1, num_batch)

    # Now we want to sample pixels with indicies shifted by disparity in X direction
    # For that we convert disparity from % to pixels and add to X indicies
    x = x + x_offset.contiguous().reshape(-1) * width
    # Make sure we don't go outside of image
    x = torch.clamp(x, 0.0, width - 1 + 2 * edge_size)
    # Round disparity to sample from integer-valued pixel grid
    y0 = torch.floor(y)
    # In X direction round both down and up to apply linear interpolation
    # between them later
    x0 = torch.floor(x)
    x1 = x0 + 1
    # After rounding up we might go outside the image boundaries again
    x1 = x1.clamp(max=(width - 1 + 2 * edge_size))

    # Calculate indices to draw from flattened version of image batch
    dim2 = width + 2 * edge_size
    dim1 = (width + 2 * edge_size) * (height + 2 * edge_size)
    # Set offsets for each image in the batch
    base = dim1 * torch.arange(num_batch).type(tensor_type)
    base = base.reshape(-1, 1).repeat(1, height * width).reshape(-1)
    # One pixel shift in Y  direction equals dim2 shift in flattened array
    base_y0 = base + y0 * dim2
    # Add two versions of shifts in X direction separately
    idx_l = base_y0 + x0
    idx_r = base_y0 + x1

    # Sample pixels from images
    pix_l = im_flat.gather(1, idx_l.repeat(num_channels, 1).long())
    pix_r = im_flat.gather(1, idx_r.repeat(num_channels, 1).long())

    # Apply linear interpolation to account for fractional offsets
    weight_l = x1 - x
    weight_r = x - x0
    output = weight_l * pix_l + weight_r * pix_r

    # Reshape back into image batch and permute back to (N,C,H,W) shape
    output = output.reshape(num_channels, num_batch, height, width).permute(1, 0, 2, 3)

    return output


def generate_image_left(
    right_img_batch: torch.Tensor, right_disp_batch: torch.Tensor
) -> torch.Tensor:
    """
    It generates a left image, starting from the right one, using disparities to sample.
        - `right_img_batch`: [B, C, H, W] Right images to warp into left ones using the corresponding disparities;
        - `right_disp_batch`: [B, 1, H, W] The corresponding disparities to use for right images.
    """
    return bilinear_sampler_1d_h(right_img_batch, -right_disp_batch)


def generate_image_right(
    left_img_batch: torch.Tensor, left_disp_batch: torch.Tensor
) -> torch.Tensor:
    """
    It generates a right image, starting from the left one, using disparities to sample.
        - `left_img_batch`: [B, C, H, W] Left images to warp into right ones using the corresponding disparities;
        - `left_disp_batch`: [B, 1, H, W] The corresponding disparities to use for left images.
    """
    return bilinear_sampler_1d_h(left_img_batch, left_disp_batch)


def L_lr(
    disp_l_batch_pyramid: List[torch.Tensor],
    disp_r_batch_pyramid: List[torch.Tensor],
) -> torch.Tensor:
    """
    # Left-righ consistency loss
        - `disp_l_batch_pyramid`: List[Tensor[B, 1, H, W]] Disparities for left images (at different scales);
        - `disp_r_batch_pyramid`: List[Tensor[B, 1, H, W]] Disparities for right images (at different scales).
    """
    right_to_left_disp = [
        generate_image_left(disp_r, disp_l)  # [B, 1, H, W]
        for disp_r, disp_l in zip(disp_r_batch_pyramid, disp_l_batch_pyramid)
    ]
    left_to_right_disp = [
        generate_image_right(disp_l, disp_r)  # [B, 1, H, W]
        for disp_l, disp_r in zip(disp_l_batch_pyramid, disp_r_batch_pyramid)
    ]
    lr_left_loss = [
        torch.mean(torch.abs(rtl_disp - disp_l))
        for rtl_disp, disp_l in zip(right_to_left_disp, disp_l_batch_pyramid)
    ]
    lr_right_loss = [
        torch.mean(torch.abs(ltr_disp - disp_r))
        for ltr_disp, disp_r in zip(left_to_right_disp, disp_r_batch_pyramid)
    ]
    return sum(lr_left_loss + lr_right_loss)


def L_total(
    est_batch_pyramid_l: torch.Tensor,
    est_batch_pyramid_r: torch.Tensor,
    img_batch_pyramid_l: torch.Tensor,
    img_batch_pyramid_r: torch.Tensor,
    disp_batch_pyramid_l: torch.Tensor,
    disp_batch_pyramid_r: torch.Tensor,
    weight_SSIM=0.85,
    weight_ap=1,
    weight_lr=1,
    weight_df=0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    # Total loss
    It combines the various losses to obtain the total loss.

    Images constructed by using disparities (at various resolutions):
        - `est_batch_pyramid_l`: List[Tensor] Reconstructed (warped) left images;
        - `est_batch_pyramid_r`: List[Tensor] Reconstructed (warped) right images.

    Images (at various resolutions):
        - `img_batch_pyramid_l`: List[Tensor] Images in left batch;
        - `img_batch_pyramid_r`: List[Tensor] Images in right batch.

    Disparities (at various resolutions)
        - `disp_batch_pyramid_l`: List[Tensor] Disparities calculated on left batch;
        - `disp_batch_pyramid_r`: List[Tensor] Disparities calculated on right batch.
    """
    L_ap_tot = L_ap(est_batch_pyramid_l, img_batch_pyramid_l, weight_SSIM) + L_ap(
        est_batch_pyramid_r, img_batch_pyramid_r, weight_SSIM
    )
    L_df_tot = L_df(
        disp_batch_pyramid_l,
        img_batch_pyramid_l,
    ) + L_df(
        disp_batch_pyramid_r,
        img_batch_pyramid_r,
    )
    L_lr_tot = L_lr(disp_batch_pyramid_l, disp_batch_pyramid_r)
    L_total = L_ap_tot * weight_ap + L_df_tot * weight_df + L_lr_tot * weight_lr
    return L_total, L_ap_tot, L_df_tot, L_lr_tot
