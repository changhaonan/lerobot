from lerobot.common.policies.arp import arp
from einops import rearrange
import math
import torch
import torch.nn as nn
from torch import Tensor
import torchvision
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.models._utils import IntermediateLayerGetter
from collections import deque
import numpy as np
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)
from dataclasses import dataclass, field
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import DictConfig, OmegaConf


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


class ARPNetwork(nn.Module):
    def __init__(
        self,
        dim_model=384,
        dropout=0.1,
        pre_norm=False,
        mlp_ratio=4,
        n_heads=8,
        n_encoder_layers=4,
        n_arp_layers=4,
        num_gmm_latents=4,
        action_from_glb_only=True,
        visual_guide_downsample=4,
        horizon=15,
        num_guide_points=15,
    ):
        super().__init__()
        self.name = "arp"
        backbone_model = getattr(torchvision.models, "resnet18")(
            replace_stride_with_dilation=[False, False, False],
            weights="ResNet18_Weights.IMAGENET1K_V1",
            norm_layer=FrozenBatchNorm2d,
        )
        self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_model,
                dropout=dropout,
                norm_first=pre_norm,
                batch_first=False,
                activation="relu",
                dim_feedforward=int(dim_model * mlp_ratio),
                nhead=n_heads,
            ),
            num_layers=n_encoder_layers,
            norm=nn.LayerNorm(dim_model),
        )

        self.robot_state_input_proj = nn.Linear(8, dim_model)
        self.visual_input_proj = nn.Conv2d(backbone_model.fc.in_features, dim_model, kernel_size=1)
        self.robot_state_pos_embed = nn.Embedding(1, dim_model)
        self.visual_pos_embed = SinusoidalPositionEmbedding2d(dim_model // 2)

        stride = 32
        self.policy = arp.AutoRegressivePolicy(
            arp.ModelConfig(
                n_embd=dim_model,
                embd_pdrop=dropout,
                layers=[
                    arp.LayerType.make(
                        n_head=n_heads,
                        mlp_ratio=mlp_ratio,
                        AdaLN=True,
                        mlp_dropout=dropout,
                        attn_kwargs={"attn_pdrop": dropout, "resid_pdrop": dropout},
                        cond_attn_kwargs={"attn_pdrop": dropout, "resid_pdrop": dropout},
                        condition_on="visual-tokens",
                    )
                ]
                * n_arp_layers,
                tokens=[
                    arp.TokenType.make(name="state", is_control=True, is_continuous=True, dim=8, embedding="linear"),
                    arp.TokenType.make(
                        name="prompt-features", dim=1, embedding="discrete", is_control=True, embedding_kwargs={"embed_from": "prompt-features"}
                    ),
                    arp.TokenType.make(
                        name="visual-guide",
                        is_continuous=True,
                        dim=2,
                        embedding="position_2d",
                        predictor="upsample_from_2d_attn",
                        predictor_kwargs={
                            "attn_with": "visual-featmap",
                            "upscale_ratio": stride // visual_guide_downsample,
                            "label_name": "smooth-heatmap",
                        },
                    ),
                    arp.TokenType.make(
                        name="action",
                        is_continuous=True,
                        dim=8,
                        embedding="linear",
                        predictor="gmm",
                        predictor_kwargs={"num_latents": num_gmm_latents},
                    ),
                ],
            )
        )
        self.visual_guide_downsample = visual_guide_downsample
        self.action_from_glb_only = action_from_glb_only
        self.horizon = horizon
        self.num_guide_points = num_guide_points

    def forward(self, batch):
        """
        observation.images: (bs, num_obs: 1, num_cam: 1, 3, 512, 512)
        observation.state: (bs, 1, 8)
        action: (bs, L, 8)
        action_is_pad: (bs, L)
        visual_guide: (bs, l, 2) only needed during training, l <= L
        """
        num_guide_points = self.num_guide_points
        horizon = self.horizon
        if self.training:
            batch = self._preprocess(batch)
            assert (
                num_guide_points == batch["visual_guide"].shape[1]
            ), f"Number of guide points mismatch: {num_guide_points} vs {batch['visual_guide'].shape[1]}"
            assert horizon == batch["action"].shape[1], f"Horizon mismatch: {horizon} vs {batch['action'].shape[1]}"
        dev = batch["observation.images"].device

        images = batch["observation.images"][:, 0, 0]  # [bs, 3, 512, 512]
        H, W = images.shape[-2:]
        visual_features = self.backbone(images)["feature_map"]
        bs, _, fh, fw = visual_features.shape
        visual_pos_embed = self.visual_pos_embed(visual_features).to(dtype=visual_features.dtype)
        visual_features = self.visual_input_proj(visual_features)

        robot_state_embed = self.robot_state_input_proj(batch["observation.state"][:, 0])
        encoder_in = torch.cat([robot_state_embed[None, ...], rearrange(visual_features, "b c h w -> (h w) b c")])
        pos_embed = torch.cat([self.robot_state_pos_embed.weight.unsqueeze(1), visual_pos_embed.flatten(2).permute(2, 0, 1)])

        encoder_out = self.encoder(encoder_in + pos_embed)
        encoder_out = encoder_out.permute(1, 0, 2)  # (B, fh*fw, C)
        visual_featmap = encoder_out[:, 1:, :].permute(0, 2, 1).reshape(bs, 1, -1, fh, fw).repeat(1, num_guide_points, 1, 1, 1).flatten(0, 1)
        visual_tokens = encoder_out[:, 1:, :]
        global_feat = encoder_out[:, :1, :]

        name2id = self.policy.token_name_2_ids
        tk_names = ["visual-guide"] * num_guide_points + ["prompt-features"] + ["action"] * horizon
        tk_ids = [name2id[name] for name in tk_names]
        chk_ids = [0] * num_guide_points + [1] + [2] * horizon

        if self.training:
            tk_is_pad_mask = torch.cat([torch.full([bs, 1 + num_guide_points], fill_value=False, device=dev), batch["action_is_pad"]], dim=1)
            vH, vW = H // self.visual_guide_downsample, W // self.visual_guide_downsample
            visual_guide = batch["visual_guide"].float() / self.visual_guide_downsample
            heatmap = generate_heatmap_from_screen_pts(visual_guide.flatten(0, 1), (vH, vW)).reshape(bs, num_guide_points, vH, vW)
            chk_ids = torch.as_tensor(chk_ids, device=dev)[None, :]
            tk_vals = arp.cat_uneven_blc_tensors(visual_guide, torch.zeros(bs, 1, 1, device=dev), batch["action"])
            tk_ids = torch.as_tensor(tk_ids).to(dev)[None, :, None].repeat(bs, 1, 1)
            tks = torch.cat([tk_vals, tk_ids], dim=-1)
            if self.action_from_glb_only:
                loss_dict1 = self.policy.compute_loss(
                    tks[:, :num_guide_points],
                    chk_ids[:, :num_guide_points],
                    contexts={"visual-tokens": visual_tokens, "visual-featmap": visual_featmap, "smooth-heatmap": heatmap.flatten(0, 1)},
                )
                loss_dict2 = self.policy.compute_loss(
                    tks,
                    chk_ids,
                    contexts={"visual-tokens": None, "prompt-features": global_feat},
                    skip_tokens=[name2id["visual-guide"]],
                    valid_tk_mask=~tk_is_pad_mask,
                )
                loss_dict = {**loss_dict1, **loss_dict2}
            else:
                loss_dict = self.policy.compute_loss(
                    tks,
                    chk_ids,
                    contexts={
                        "visual-tokens": visual_tokens,
                        "visual-featmap": visual_featmap,
                        "prompt-features": global_feat,
                        "smooth-heatmap": heatmap.flatten(0, 1),
                    },
                    valid_tk_mask=~tk_is_pad_mask,
                )
            # "loss" is the sum of all losses
            loss_dict["loss"] = sum(loss_dict.values())
            return loss_dict
        else:
            prompt_tks = torch.zeros(bs, 0, 2, device=dev)
            future_tk_chk_ids = [{"tk_id": tk_id, "chk_id": chk_id} for tk_id, chk_id in zip(tk_ids, chk_ids)]
            future_tk_chk_ids[num_guide_points]["tk_val"] = 0  # prompt features

            if self.action_from_glb_only:
                visual_tokens = {frozenset(chk_ids[: num_guide_points + 1]): visual_tokens, frozenset(chk_ids[num_guide_points + 1 :]): None}

            pred_tks = self.policy.generate(
                prompt_tks,
                future_tk_chk_ids,
                contexts={"visual-tokens": visual_tokens, "visual-featmap": visual_featmap, "prompt-features": global_feat},
                sample=True,
            )
            return pred_tks[:, num_guide_points + 1 :, :-1]

    def _preprocess(self, batch):
        bs = batch["action"].shape[0]
        proj = batch["observation.camera_proj.front"].reshape(bs, 3, 4)
        action_points = batch["action"][:, :, :3]
        action_points = torch.cat([action_points, torch.ones((bs, action_points.shape[1], 1), device=action_points.device)], dim=-1)
        proj_p = torch.matmul(proj, action_points.permute(0, 2, 1)).permute(0, 2, 1)
        proj_p = proj_p[:, :, :2] / (proj_p[:, :, 2:] + 1e-6)
        batch["visual_guide"] = proj_p[:, : self.num_guide_points, :]
        # [DEBUG]
        # import cv2
        # for i in range(bs):
        #     img = batch["observation.images.front"][i, 0].cpu().numpy().transpose(1, 2, 0)
        #     img = (img * 255).astype(np.uint8)
        #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #     for j in range(self.num_guide_points):
        #         cv2.circle(img, tuple(proj_p[i, j].int().tolist()), 5, (0, 255, 0), -1)
        #     cv2.imwrite(f"debug_{i}.png", img)
        return batch


@dataclass
class ARPConfig:
    n_obs_steps: int = 1
    n_action_steps: int = 1
    input_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "observation.image": [3, 96, 96],
            "observation.state": [2],
        }
    )
    output_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "action": [2],
        }
    )

    # Normalization / Unnormalization
    input_normalization_modes: dict[str, str] = field(
        default_factory=lambda: {
            "observation.image": "mean_std",
            "observation.state": "min_max",
        }
    )
    output_normalization_modes: dict[str, str] = field(default_factory=lambda: {"action": "min_max"})

    # Network parameters
    dim_model: int = 384
    dropout: float = 0.1
    pre_norm: bool = False
    mlp_ratio: int = 4
    n_heads: int = 8
    n_encoder_layers: int = 4
    n_arp_layers: int = 4
    num_gmm_latents: int = 4
    action_from_glb_only: bool = True
    visual_guide_downsample: int = 4
    horizon: int = 15
    num_guide_points: int = 15


class ARPPolicy(nn.Module, PyTorchModelHubMixin, library_name="lerobot", repo_url="https://github.com/mlzxy/arp", tags=["robotics", "arp"]):
    """ARP Policy as per https://arxiv.org/abs/2410.03132.
    Policy wrapper aligned with lerobot.
    """

    name = "arp"

    def __init__(
        self,
        config: ARPConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__()
        self.config = config
        self.normalize_inputs = Normalize(config.input_shapes, config.input_normalization_modes, dataset_stats)
        self.normalize_targets = Normalize(config.output_shapes, config.output_normalization_modes, dataset_stats)
        self.unnormalize_outputs = Unnormalize(config.output_shapes, config.output_normalization_modes, dataset_stats)

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None
        self.model = ARPNetwork(
            dim_model=config.dim_model,
            dropout=config.dropout,
            pre_norm=config.pre_norm,
            mlp_ratio=config.mlp_ratio,
            n_heads=config.n_heads,
            n_encoder_layers=config.n_encoder_layers,
            n_arp_layers=config.n_arp_layers,
            num_gmm_latents=config.num_gmm_latents,
            action_from_glb_only=config.action_from_glb_only,
            visual_guide_downsample=config.visual_guide_downsample,
            horizon=config.horizon,
            num_guide_points=config.num_guide_points,
        )
        self.expected_image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]
        self.use_env_state = "observation.environment_state" in config.input_shapes

        self.reset()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if len(self.expected_image_keys) > 0:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
        if self.use_env_state:
            self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)

    def forward(self, batch: dict[str, Tensor]) -> dict:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        if len(self.expected_image_keys) > 0:
            batch = dict(batch)
            batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)
        log_dict = self.model(batch)
        return log_dict

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        batch = self.normalize_inputs(batch)
        if len(self.expected_image_keys) > 0:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)
        # Note: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues["action"]) == 0:
            # stack n latest observations from the queue
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
            actions = self.model(batch)

            # Chunk action
            actions = self.unnormalize_outputs({"action": actions})["action"]
            start = 0
            end = start + self.config.n_action_steps
            actions = actions[:, start:end]
            self._queues["action"].extend(actions.transpose(0, 1))

        action = self._queues["action"].popleft()
        return action


# def make_policy_arp(hydra_cfg: DictConfig, pretrained_policy_name_or_path: str | None = None, dataset_stats=None):
#     if pretrained_policy_name_or_path is None:
#         policy = ARPPolicy(hydra_cfg.policy, dataset_stats)
#     else:
#         policy = ARPPolicy(hydra_cfg.policy)
#         policy.load_state_dict(
#             ARPPolicy.from_pretrained(pretrained_policy_name_or_path).state_dict(),
#         )
#     # ARPNetwork(**OmegaConf.to_container(hydra_cfg.policy, resolve=True))
#     policy.to(hydra_cfg.device)
#     return policy


if __name__ == "__main__":
    import cv2
    import numpy as np
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    delta_timestamps = {
        "observation.images.front": [0.0],
        "observation.camera_ext.front": [0.0],
        "observation.camera_int.front": [0.0],
        "observation.state": [0.0],
        "action": [0.0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.48, 0.52, 0.56, 0.60],
    }

    dataset = LeRobotDataset(
        "changhaonan/RPLBenchData",
        split="train",
        delta_timestamps=delta_timestamps,
        image_transforms=None,
        video_backend="pyav",
        root="/home/harvey/Project/LGMCTSpp/log/Oct_18_1",
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=0)
    batch = next(iter(data_loader))
    model = ARPNetwork(action_from_glb_only=True, horizon=16, num_guide_points=15)
    model.train()

    print("=== Training ===")
    log_dict = model(batch)
    print(log_dict)

    print("=== Inference ===")
    with torch.no_grad():
        model.eval()
        pred_action = model(batch)
    print(pred_action)
