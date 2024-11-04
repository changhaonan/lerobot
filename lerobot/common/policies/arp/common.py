import albumentations as A
from scipy.spatial.transform import Rotation
import cv2
import torch
import math
import torch.nn as nn
from torchvision.transforms.functional import pil_to_tensor
from albumentations.pytorch import ToTensorV2
import numpy as np
from torchvision.utils import Optional, Tuple, Union, ImageDraw, List, Image


def augmentation_recipe(
    target_size=None,
    color=0.25,
    spatial=0.25,
    hflip=0.5,
    vflip=0.1,
    mask=0,
    keypoints=False,
    normalize=True,
    to_tensor=False,
    additional_targets=None,
):
    T_flip = A.Compose([A.HorizontalFlip(p=hflip), A.VerticalFlip(p=vflip)], additional_targets=additional_targets)

    T_mask = A.XYMasking(
        num_masks_x=10,
        num_masks_y=10,
        mask_x_length=6,
        mask_y_length=6,
        fill_value=0,
        mask_fill_value=0,
        always_apply=False,
        p=mask,
    )

    T_spatial = A.OneOf(
        [
            # A.Affine(rotate=(-45, 45), translate_percent=(0.1, 0.3), scale=(0.5, 0.95)),
            # A.Perspective(scale=(0.2, 0.4)),
            # A.Perspective(scale=1.0),
            A.SafeRotate(limit=60, border_mode=cv2.BORDER_CONSTANT),
            # A.Rotate(limit=60),  #FIXME: this will cause the change of keypoints.
        ],
        p=1.0,
    )
    T_spatial = A.OneOf(T_spatial, p=spatial)

    T_color = A.OneOf(
        [
            A.ElasticTransform(alpha=80.0),
            A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 5.0)),
            A.Posterize(num_bits=2, p=1.0),
            A.ColorJitter(
                brightness=(0.5, 1),
                contrast=(0.5, 1),
                saturation=(0.5, 1),
                hue=(-0.1, 0.1),
                p=1.0,
            ),
            A.Equalize(p=1.0),
        ],
        p=color,
    )

    Ts = []
    if normalize:
        Ts.append(A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)))
    if to_tensor:
        Ts.append(ToTensorV2())

    if color > 0:
        Ts = [T_color] + Ts

    if spatial > 0:
        Ts = [T_spatial] + Ts

    if max(vflip, hflip) > 0:
        Ts = [T_flip] + Ts

    if mask > 0:
        Ts = [T_mask] + Ts

    if target_size is not None:
        if isinstance(target_size, (list, tuple)):
            H, W = target_size
        else:
            H = W = target_size
        Ts = [
            A.Resize(H, W),
        ] + Ts

    if keypoints:
        return A.Compose(Ts, keypoint_params=A.KeypointParams(format="xy", remove_invisible=False), additional_targets=additional_targets)
    else:
        return A.Compose(Ts, additional_targets=additional_targets)


@torch.no_grad()
def draw_keypoints(
    image: torch.Tensor,
    keypoints: torch.Tensor,
    connectivity: Optional[List[Tuple[int, int]]] = None,
    colors: Optional[Union[str, Tuple[int, int, int]]] = (255, 0, 0),
    line_color="white",
    radius: int = 2,
    width: int = 3,
    output_pil=True,
    transparency=1.0,
    line_under=True,
) -> torch.Tensor:
    def is_valid(*args):
        return all([a >= 0 for a in args])

    if isinstance(image, Image.Image):
        image = pil_to_tensor(image)

    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
        if image.shape[-1] == 3:
            image = image.permute(2, 0, 1)

    POINT_SIZE = keypoints.shape[-1]
    if isinstance(keypoints, np.ndarray):
        keypoints = torch.from_numpy(keypoints)

    keypoints = keypoints.reshape(1, -1, POINT_SIZE)

    ndarr = image.permute(1, 2, 0).cpu().numpy()
    img_to_draw = Image.fromarray(ndarr)
    if transparency < 1.0:
        draw = ImageDraw.Draw(img_to_draw, "RGBA")
    else:
        draw = ImageDraw.Draw(img_to_draw, None if POINT_SIZE == 2 else "RGBA")
    keypoints = keypoints.clone()
    if POINT_SIZE == 3:
        keypoints[:, :, -1] *= 255
    img_kpts = keypoints.to(torch.int64).tolist()

    for kpt_id, kpt_inst in enumerate(img_kpts):
        kpt_size = len(kpt_inst[0])

        def draw_line():
            if connectivity is not None:
                for connection in connectivity:
                    start_pt_x = kpt_inst[connection[0]][0]
                    start_pt_y = kpt_inst[connection[0]][1]

                    end_pt_x = kpt_inst[connection[1]][0]
                    end_pt_y = kpt_inst[connection[1]][1]

                    if not is_valid(start_pt_x, start_pt_y, end_pt_x, end_pt_y):
                        continue

                    if transparency < 1.0:
                        kp_line_color = line_color + (int(255 * (1 - transparency)),)
                    else:
                        kp_line_color = line_color

                    draw.line(((start_pt_x, start_pt_y), (end_pt_x, end_pt_y)), width=width, fill=kp_line_color)

        def draw_points():
            for inst_id, kpt in enumerate(kpt_inst):
                if not is_valid(*kpt):
                    continue
                x1 = kpt[0] - radius
                x2 = kpt[0] + radius
                y1 = kpt[1] - radius
                y2 = kpt[1] + radius
                if len(kpt) == 3:
                    kp_color = colors + (int(kpt[2]),)
                elif transparency < 1.0:
                    kp_color = colors + (int(255 * (1 - transparency)),)
                else:
                    kp_color = colors
                draw.ellipse([x1, y1, x2, y2], fill=kp_color, outline=None, width=0)

        if line_under:
            draw_line()
            draw_points()
        else:
            draw_points()
            draw_line()

    if output_pil:
        return img_to_draw
    else:
        return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)


@torch.no_grad()
def draw_kps_horizon(
    image: torch.Tensor,
    kpts_horizon: torch.Tensor,
    connectivity: list[tuple[int, int]] | None = None,
    colors: list[tuple[int, int, int]] | tuple[int, int, int] = (255, 0, 0),
    kp_colors: list[tuple[int, int, int]] | tuple[int, int, int] = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)],
    radius: int = 2,
    width: int = 3,
    output_pil=True,
    line_under=True,
):
    """Draw keypoints with a horizon.
    Args:
        kps_horizon: torch.Tensor of shape (1, N, H, 3/2) where B is the batch size, H is the horizon, N is the number of keypoints, and 3 is the x, y, and visibility.
    """

    def is_valid(*args):
        return all([a >= 0 for a in args])

    if isinstance(image, Image.Image):
        image = pil_to_tensor(image)

    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
        if image.shape[-1] == 3:
            image = image.permute(2, 0, 1)

    POINT_SIZE = kpts_horizon.shape[-1]
    NUM_KPS = kpts_horizon.shape[1]
    HORIZON = kpts_horizon.shape[0]
    kpts_horizon = kpts_horizon.reshape(HORIZON, NUM_KPS, POINT_SIZE)
    ndarr = image.permute(1, 2, 0).cpu().numpy()
    img_to_draw = Image.fromarray(ndarr)
    draw = ImageDraw.Draw(img_to_draw, "RGBA")
    kpts_horizon = kpts_horizon.clone()
    if POINT_SIZE == 3:
        kpts_horizon[..., -1] *= 255
    img_kpts_horizon = kpts_horizon.to(torch.int64).tolist()

    for horizon_id, kpts in enumerate(img_kpts_horizon):
        transparency = (horizon_id / HORIZON) * 0.5
        # draw lines within one frame
        if connectivity is not None:
            for connection in connectivity:
                start_pt_x = kpts[connection[0]][0]
                start_pt_y = kpts[connection[0]][1]

                end_pt_x = kpts[connection[1]][0]
                end_pt_y = kpts[connection[1]][1]

                if not is_valid(start_pt_x, start_pt_y, end_pt_x, end_pt_y):
                    continue

                kp_line_color = (255, 255, 255) + (int(255 * (1 - transparency)),)
                draw.line(((start_pt_x, start_pt_y), (end_pt_x, end_pt_y)), width=1, fill=kp_line_color)
        for kpt_id, kpt_inst in enumerate(kpts):
            kpt_size = len(kpt_inst)
            if kpt_size == 2:
                kpt_inst.append(255)
            if not is_valid(*kpt_inst):
                continue
            # draw kpts
            x1 = kpt_inst[0] - radius
            x2 = kpt_inst[0] + radius
            y1 = kpt_inst[1] - radius
            y2 = kpt_inst[1] + radius
            kp_color = kp_colors[kpt_id % len(kp_colors)] + (int(255 * (1 - transparency)),)
            draw.ellipse([x1, y1, x2, y2], fill=kp_color, outline=None, width=0)

    # # draw lines across frames, connecting the same keypoint across different frames
    # for keypoint_id in range(NUM_KPS):
    #     for horizon_id in range(HORIZON - 1):
    #         start_pt_x = img_kpts_horizon[horizon_id][keypoint_id][0]
    #         start_pt_y = img_kpts_horizon[horizon_id][keypoint_id][1]

    #         end_pt_x = img_kpts_horizon[horizon_id + 1][keypoint_id][0]
    #         end_pt_y = img_kpts_horizon[horizon_id + 1][keypoint_id][1]

    #         if not is_valid(start_pt_x, start_pt_y, end_pt_x, end_pt_y):
    #             continue

    #         if transparency < 1.0:
    #             kp_line_color = colors + (int(255 * (1 - transparency)),)
    #         else:
    #             kp_line_color = colors

    #         draw.line(((start_pt_x, start_pt_y), (end_pt_x, end_pt_y)), width=width, fill=kp_line_color)

    if output_pil:
        return img_to_draw
    else:
        return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)


def generate_heatmap_from_screen_pts(pt, res, sigma=1.5, thres_sigma_times=3):  # 2d label smoothing
    """
    Pytorch code to generate heatmaps from point. Points with values less than
    thres are made 0
    :type pt: torch.FloatTensor of size (num_pt, 2)
    :type res: int or (int, int)
    :param sigma: the std of the gaussian distribition. if it is -1, we
        generate a hm with one hot vector
    :type sigma: float
    :type thres: float
    """
    num_pt, x = pt.shape
    assert x == 2
    assert sigma > 0

    if isinstance(res, int):
        resx = resy = res
    else:
        resx, resy = res

    _hmx = torch.arange(0, resy).to(pt.device)
    _hmx = _hmx.view([1, resy]).repeat(resx, 1).view([resx, resy, 1])
    _hmy = torch.arange(0, resx).to(pt.device)
    _hmy = _hmy.view([resx, 1]).repeat(1, resy).view([resx, resy, 1])
    hm = torch.cat([_hmx, _hmy], dim=-1)
    hm = hm.view([1, resx, resy, 2]).repeat(num_pt, 1, 1, 1)  # one HxW heatmap for each point?

    pt = pt.view([num_pt, 1, 1, 2])
    hm = torch.exp(-1 * torch.sum((hm - pt) ** 2, -1) / (2 * (sigma**2)))  # RBF Kernel
    thres = np.exp(-1 * (thres_sigma_times**2) / 2)  # truncated
    hm[hm < thres] = 0.0

    hm /= torch.sum(hm, (1, 2), keepdim=True) + 1e-6  # normalization
    return hm  # (n_pt, h, w)


class SinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

    The variation is that the position indices are normalized in [0, 2π] (not quite: the lower bound is 1/H
    for the vertical direction, and 1/W for the horizontal direction.
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: The desired dimension of the embeddings.
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # Inverse "common ratio" for the geometric progression in sinusoid frequencies.
        self._temperature = 10000

    def forward(self, x):
        """
        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi
        inverse_frequency = self._temperature ** (2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension)
        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)
        return pos_embed


def denormalize_bchw_image(x):
    pixel_std = torch.as_tensor((0.26862954, 0.26130258, 0.27577711)) * 255
    pixel_mean = torch.as_tensor((0.48145466, 0.4578275, 0.40821073)) * 255

    pixel_std = pixel_std.to(x.device)
    pixel_mean = pixel_mean.to(x.device)

    return (x * pixel_std[None, :, None, None]) + pixel_mean[None, :, None, None]


def normalize_bchw_image(x, scaler=255):
    pixel_std = torch.as_tensor((0.26862954, 0.26130258, 0.27577711)) * scaler
    pixel_mean = torch.as_tensor((0.48145466, 0.4578275, 0.40821073)) * scaler

    pixel_std = pixel_std.to(x.device)
    pixel_mean = pixel_mean.to(x.device)

    return (x - pixel_mean[None, :, None, None]) / pixel_std[None, :, None, None]


def pose7_to_frame(pose, scale=0.5):
    pose = pose.copy()
    R = Rotation.from_quat(pose[:, 3:]).as_matrix() * scale
    R = np.transpose(R, (0, 2, 1))
    t = pose[:, :3]
    return np.concatenate([t[:, None, :], R + t[:, None, :]], axis=1)
