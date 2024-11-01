from collections import defaultdict
from einops import rearrange
import os
import torch
import torch.nn as nn
import threading
from torch import Tensor
import torchvision
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.models._utils import IntermediateLayerGetter

# from torchvision.models import vit_b_16, ViT_B_16_Weights, vit_s_16, ViT_S_16_Weights
from copy import deepcopy
from transformers import Dinov2Model
from collections import deque
import numpy as np
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)
from lerobot.common.utils.io_utils import write_video
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass, field
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms.functional import to_pil_image

try:
    from common import (
        draw_keypoints,
        denormalize_bchw_image,
        generate_heatmap_from_screen_pts,
        SinusoidalPositionEmbedding2d,
        pose7_to_frame,
        Rotation,
        augmentation_recipe,
        normalize_bchw_image,
    )
    import arp
except ImportError:
    from lerobot.common.policies.arp.common import (
        draw_keypoints,
        denormalize_bchw_image,
        generate_heatmap_from_screen_pts,
        SinusoidalPositionEmbedding2d,
        pose7_to_frame,
        Rotation,
        augmentation_recipe,
        normalize_bchw_image,
    )
    from lerobot.common.policies.arp import arp
import cv2
import random


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
        visual_guide_downsample=2,
        horizon=15,
        reverse_plan=True,
        initial_chunk_size=1,
        backbones="vit",
        **kwargs,
    ):
        super().__init__()
        if backbones == "resnet18":
            backbone_model = getattr(torchvision.models, "resnet18")(
                replace_stride_with_dilation=[False, False, False],
                weights="ResNet18_Weights.IMAGENET1K_V1",
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
            self.visual_input_proj = nn.Conv2d(backbone_model.fc.in_features, dim_model, kernel_size=1)
        elif backbones == "vit":
            lora_config = LoraConfig(target_modules=["fc1", "fc2", "query", "value", "key", "dense", "projection"])
            dinov2 = Dinov2Model.from_pretrained(f"facebook/dinov2-small")
            self.backbone = get_peft_model(dinov2, lora_config)
            self.visual_input_proj = nn.Linear(self.backbone.config.hidden_size, dim_model)
        else:
            raise ValueError(f"Backbone {backbones} is not supported.")
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
        self.visual_pos_embed = SinusoidalPositionEmbedding2d(dim_model // 2)
        self.cls_embed = nn.Embedding(1, dim_model)
        self.cls_pos_embed = nn.Embedding(1, dim_model)

        stride = 32 if backbones == "resnet18" else 14  # vit: change to 14
        self.policy = arp.AutoRegressivePolicy(
            arp.ModelConfig(
                n_embd=dim_model,
                embd_pdrop=dropout,
                layers=[
                    arp.LayerType.make(
                        n_head=n_heads,
                        mlp_ratio=mlp_ratio,
                        AdaLN=True,
                        norm_before_AdaLN=True,
                        mlp_dropout=dropout,
                        attn_kwargs={"attn_pdrop": dropout, "resid_pdrop": dropout},
                        cond_attn_kwargs={"attn_pdrop": dropout, "resid_pdrop": dropout},
                        condition_on="visual-tokens",
                    )
                ]
                * n_arp_layers,
                tokens=[
                    arp.TokenType.make(
                        name="init",
                        is_continuous=True,
                        dim=2,
                        embedding="position_2d",
                        predictor="upsample_from_2d_attn",
                        predictor_kwargs={
                            "attn_with": f"visual-featmap@init",
                            "upscale_ratio": stride // visual_guide_downsample,
                            "label_name": f"smooth-heatmap@init",
                        },
                    ),
                    arp.TokenType.make(
                        name="rest",
                        is_continuous=True,
                        dim=2,
                        embedding="position_2d",
                        predictor="upsample_from_2d_attn",
                        predictor_kwargs={
                            "attn_with": f"visual-featmap@rest",
                            "upscale_ratio": stride // visual_guide_downsample,
                            "label_name": f"smooth-heatmap@rest",
                        },
                    ),
                    arp.TokenType.make(name="gripper", is_continuous=False, dim=1, dict_sizes=[2], embedding="discrete", predictor="class"),
                ],
            )
        )
        self.visual_guide_downsample = visual_guide_downsample
        self.reverse_plan = reverse_plan
        self.horizon = horizon
        self.initial_chunk_size = initial_chunk_size

    def forward(self, batch):
        """
        observation.images.front: (bs, 1, 3, H, W)
        action_is_pad: (bs, L)

        # label
        origin, xaxis, yaxis, zaxis: (bs, L, 2),
        gripper_open: (bs, L)
        """
        horizon = self.horizon
        # Convert original batch to action in image.
        dev = batch["observation.images.front"].device

        images = batch["observation.images.front"][:, 0]  # [bs, 3, H, W]
        H, W = images.shape[-2:]
        bs = images.shape[0]
        fw, fh = W // 14, H // 14
        visual_features = self.backbone(images).last_hidden_state
        visual_feature_map = visual_features[:, 1:, :].permute(0, 2, 1).reshape(bs, -1, fh, fw)
        # bs, _, fh, fw = visual_features.shape
        visual_pos_embed = self.visual_pos_embed(visual_feature_map).to(dtype=visual_features.dtype)
        visual_features = self.visual_input_proj(visual_features)

        encoder_in = torch.cat([self.cls_embed.weight.unsqueeze(1).repeat(1, bs, 1), rearrange(visual_feature_map, "b c h w -> (h w) b c")])
        pos_embed = torch.cat([self.cls_pos_embed.weight.unsqueeze(1), visual_pos_embed.flatten(2).permute(2, 0, 1)])

        encoder_out = self.encoder(encoder_in + pos_embed)
        encoder_out = encoder_out.permute(1, 0, 2)  # (B, fh*fw, C)
        visual_featmap = encoder_out[:, 1:, :].permute(0, 2, 1).reshape(bs, 1, -1, fh, fw)
        visual_tokens = encoder_out

        name2id = self.policy.token_name_2_ids
        tk_names = (
            [
                "init",
            ]
            * 4
            * self.initial_chunk_size
            + ["rest"] * 4 * (horizon - self.initial_chunk_size)
            + ["gripper"] * horizon
        )
        tk_ids = [name2id[name] for name in tk_names]
        # Devide the rest into multiple chunks
        chk_ids = [0] * 4 * self.initial_chunk_size
        num_reset_chunk = 4
        assert (horizon - self.initial_chunk_size) % num_reset_chunk == 0, "The horizon should be divisible by num_reset_chunk"
        for i in range(num_reset_chunk):
            chk_ids += [1 + i] * (4 * ((horizon - self.initial_chunk_size) // num_reset_chunk))
        chk_ids += [chk_ids[-1] + 1] * (horizon - len(chk_ids) // 4) * 4
        chk_ids += [chk_ids[-1] + 1] * horizon

        if self.training:
            assert horizon == batch["origin"].shape[1]
            if self.reverse_plan:
                for k in ["origin", "xaxis", "yaxis", "zaxis"]:
                    batch[k] = torch.flip(batch[k], [1])

            vH, vW = H // self.visual_guide_downsample, W // self.visual_guide_downsample

            actions = defaultdict(list)
            contexts = defaultdict(list)
            for k in ["origin", "xaxis", "yaxis", "zaxis"]:
                visual_action = batch[k].float() / self.visual_guide_downsample
                actions["init"].append(visual_action[:, : self.initial_chunk_size])
                actions["rest"].append(visual_action[:, self.initial_chunk_size :])

                heatmap = generate_heatmap_from_screen_pts(visual_action.flatten(0, 1), (vH, vW)).reshape(bs, horizon, 1, vH, vW)
                contexts[f"smooth-heatmap@init"].append(heatmap[:, : self.initial_chunk_size])
                contexts[f"smooth-heatmap@rest"].append(heatmap[:, self.initial_chunk_size :])

            for k in ["init", "rest"]:
                actions[k] = torch.cat([a[:, :, None, :] for a in actions[k]], dim=2)  # bs, L, 4, 2
                hm = torch.cat(contexts[f"smooth-heatmap@{k}"], dim=2).flatten(1, 2)
                contexts[f"smooth-heatmap@{k}"] = hm.flatten(0, 1)
                contexts[f"visual-featmap@{k}"] = visual_featmap.repeat(1, hm.shape[1], 1, 1, 1).flatten(0, 1)

            actions["gripper"] = batch["gripper_open"].unsqueeze(-1).repeat(1, 1, 2).float()
            contexts["visual-tokens"] = visual_tokens

            tk_vals = torch.cat([torch.cat([actions["init"], actions["rest"]], dim=1).flatten(1, 2), actions["gripper"]], dim=1)
            chk_ids = torch.as_tensor(chk_ids, device=dev)[None, :]
            tk_ids = torch.as_tensor(tk_ids).to(dev)[None, :, None].repeat(bs, 1, 1)
            tks = torch.cat([tk_vals, tk_ids], dim=-1)
            loss_dict = self.policy.compute_loss(tks, chk_ids, contexts=contexts)

            DEBUG = False
            if DEBUG:
                images = denormalize_bchw_image(images)
                visualized_images = []
                for bi in range(bs):
                    img = to_pil_image(images[bi])
                    img = draw_keypoints(
                        img, torch.cat([actions["init"][bi, :, 0], actions["rest"][bi, :, 0]]).detach().cpu() * self.visual_guide_downsample
                    )
                    visualized_images.append(img)
                loss_dict["visualized_images"] = visualized_images
            # Compute a sum loss
            loss_dict["loss"] = sum([v for k, v in loss_dict.items() if "loss" in k])
            return loss_dict
        else:
            prompt_tks = torch.zeros(bs, 0, 3, device=dev)
            future_tk_chk_ids = [{"tk_id": tk_id, "chk_id": chk_id} for tk_id, chk_id in zip(tk_ids, chk_ids)]
            pred_tks = self.policy.generate(
                prompt_tks,
                future_tk_chk_ids,
                contexts={
                    "visual-tokens": visual_tokens,
                    "visual-featmap@init": visual_featmap.repeat(1, 4 * self.initial_chunk_size, 1, 1, 1).flatten(0, 1),
                    "visual-featmap@rest": visual_featmap.repeat(1, 4 * (horizon - self.initial_chunk_size) // num_reset_chunk, 1, 1, 1).flatten(
                        0, 1
                    ),
                },
                sample=True,
            )
            # at this moment, the pred tks are visual actions only
            pred_tks = pred_tks[:, : self.horizon * 4]
            keypoints = pred_tks[:, ::4, :-1] * self.visual_guide_downsample

            images = denormalize_bchw_image(images)

            RETURN_VIDEO = False
            if RETURN_VIDEO:
                random_id = random.randint(0, 100000)
                video_dir = f"./tmp/arp_video/{random_id}"
                os.makedirs(video_dir, exist_ok=True)
            threads = []
            visualized_images = []
            # observe the first 16 images
            for bi in range(min(bs, 16)):
                img = to_pil_image(images[bi])
                if not RETURN_VIDEO:
                    img = draw_keypoints(img, keypoints[bi].detach().cpu(), colors=(255, 0, 0))
                    # draw gt keypoints
                    if "origin" in batch:
                        gt_keypoints = batch["origin"]
                        img = draw_keypoints(img, gt_keypoints[bi].detach().cpu(), colors=(0, 255, 0))
                    visualized_images.append(img)
                else:
                    video_imgs = []
                    for kp, gt_kp in zip(keypoints[bi].detach().cpu(), batch["origin"][bi].detach().cpu()):
                        _img = deepcopy(img)
                        _img = draw_keypoints(_img, kp[None], colors=(255, 0, 0))
                        _img = draw_keypoints(_img, gt_kp[None], colors=(0, 255, 0))
                        video_imgs.append(_img)
                    # write video
                    video_path = f"{video_dir}/{bi}.mp4"
                    thread = threading.Thread(
                        target=write_video,
                        args=(str(video_path), video_imgs, 25),
                    )
                    thread.start()
                    threads.append(thread)
                    visualized_images.append(video_path)

            if RETURN_VIDEO:
                for thread in threads:
                    thread.join()
            return {
                "action": pred_tks,
                "visualized_images": visualized_images,
            }


######################################## For lerobot ########################################
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
            actions = self.model(batch)["action"]

            # Chunk action
            actions = self.unnormalize_outputs({"action": actions})["action"]
            start = 0
            end = start + self.config.n_action_steps
            actions = actions[:, start:end]
            self._queues["action"].extend(actions.transpose(0, 1))

        action = self._queues["action"].popleft()
        return action


#################################### Wrap augmentation into a function ####################################
# count = 0


def ImageActionAug(data, aug_prob=0.5, target_size=None, action_rep="origin"):
    """Project action to image-points.
    action_rep: action representation. "origin" or "origin+depth" or "origin+depth+rot".
    """
    horizon = len(data["action"])
    intrinsics = data["observation.camera_int.front"].reshape(3, 3).numpy().astype(np.float64)
    extrinsics = data["observation.camera_ext.front"].reshape(4, 4).numpy().astype(np.float64)
    extrinsics = np.linalg.inv(extrinsics)
    extrinsics[0] = -extrinsics[0]
    n_obs = data["observation.images.front"].shape[0]
    additional_targets = {f"image{i}": "image" for i in range(1, n_obs)}
    aug_transform = augmentation_recipe(
        target_size=target_size,
        spatial=aug_prob if aug_prob > 0 else 0.0,
        color=0.0,
        mask=aug_prob if aug_prob > 0 else 0.0,
        hflip=aug_prob if aug_prob > 0 else 0.0,
        vflip=0,
        normalize=True,
        keypoints=True,
        to_tensor=True,
        additional_targets=additional_targets,
    )

    # kpts in actions.
    action_rots = Rotation.from_euler("XYZ", data["action"][:, 3:6], degrees=False)
    action_pose7d = np.concatenate([data["action"][:, :3], action_rots.as_quat()], axis=1)
    action_kpts_3d = pose7_to_frame(action_pose7d)  # 3d keypoints
    action_kpts = cv2.projectPoints(action_kpts_3d.reshape(-1, 3), extrinsics[:3, :3], extrinsics[:3, 3], intrinsics, None)[0].reshape(horizon, 4, 2)
    num_action_kpts = action_kpts.shape[0]
    # kpts in states.
    state_rots = Rotation.from_euler("XYZ", data["observation.state"][:, 3:6], degrees=False)
    state_pose7d = np.concatenate([data["observation.state"][:, :3], state_rots.as_quat()], axis=1)
    state_kpts = pose7_to_frame(state_pose7d)
    state_kpts = cv2.projectPoints(state_kpts.reshape(-1, 3), extrinsics[:3, :3], extrinsics[:3, 3], intrinsics, None)[0].reshape(-1, 4, 2)
    num_state_kpts = state_kpts.shape[0]
    # merge kpts
    whole_kpts = np.concatenate([state_kpts.reshape(-1, 2), action_kpts.reshape(-1, 2)], axis=0)
    # THIS IMAGE is ranged in [0...1], or [0...255]
    if n_obs == 1:
        image = (data["observation.images.front"][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        _ = aug_transform(image=image, keypoints=whole_kpts.reshape(-1, 2))
        aug_image = _["image"]
        aug_images = aug_image[None, ...]
    else:
        # assemble additional images
        image = (data["observation.images.front"][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        additional_images = {
            f"image{i}": (data[f"observation.images.front"][i].permute(1, 2, 0).numpy() * 255).astype(np.uint8) for i in range(1, n_obs)
        }
        _ = aug_transform(image=image, keypoints=whole_kpts.reshape(-1, 2), **additional_images)
        aug_images = [_["image"]] + [_[f"image{i}"] for i in range(1, n_obs)]
        aug_images = torch.stack(aug_images, dim=0)
    # FIXME: there is bug here.
    # print(f"aug kpts shape: {len(_['keypoints'])}, whole kpts shape: {len(whole_kpts)}")
    # if len(_["keypoints"]) == (9 * horizon * 4):
    #     _["keypoints"] = _["keypoints"][4 * (horizon * 4) : 5 * (horizon * 4)]
    #     # global count
    #     # count += 1
    #     draw_keypoints(to_pil_image(denormalize_bchw_image(_["image"])[0]), _["keypoints"]).save(f"bad.{count}.png")

    width, height = target_size
    aug_whole_kpts = _["keypoints"].reshape(-1, 4, 2)
    # clip by image size
    aug_whole_kpts[..., 0] = np.clip(aug_whole_kpts[..., 0], 0, width - 1)
    aug_whole_kpts[..., 1] = np.clip(aug_whole_kpts[..., 1], 0, height - 1)
    # normalize to [-1, 1]
    aug_whole_kpts[..., 0] = aug_whole_kpts[..., 0] / (width - 1) * 2 - 1  # [-1, 1]
    aug_whole_kpts[..., 1] = aug_whole_kpts[..., 1] / (height - 1) * 2 - 1  # [-1, 1]
    aug_whole_kpts = torch.from_numpy(aug_whole_kpts)
    # Split augmented keypoints
    aug_state_kpts = aug_whole_kpts[:num_state_kpts, ...].reshape(-1, 4, 2)
    aug_action_kpts = aug_whole_kpts[num_state_kpts:, ...].reshape(-1, 4, 2)
    # [DEBUG]
    # print(f"state_kpts: {state_kpts.shape}, aug_state_kpts: {aug_state_kpts.shape}")
    # print(f"action_kpts: {action_kpts.shape}, aug_action_kpts: {aug_action_kpts.shape}")
    # Assemble new data
    # Assemble action
    action_chunks = action_rep.split("+")
    actions = []
    for chunk in action_chunks:
        if chunk == "origin":
            actions.append(aug_action_kpts[:, 0].reshape(-1, 2))  # (H, 2)
        elif chunk == "xyzaxis":
            # x, y, z axis
            actions.append(aug_action_kpts[:, 1:].reshape(-1, 6))
        elif chunk == "depth":
            # depth
            action_kpts_d = (extrinsics[:3, :3] @ action_kpts_3d.T + extrinsics[:3, 3:4])[:, 2].T
            action_kpts_d = torch.from_numpy(action_kpts_d).reshape(-1, 1)
            actions.append(action_kpts_d)  # (H, 1)
        elif chunk == "rot":
            # rotation
            action_rots = data["action"][:, 3:6]
            actions.append(action_rots)  # (H, 3)
        elif chunk == "gripper":
            # gripper
            actions.append(data["action"][:, -1].reshape(-1, 1))
    actions = torch.concat(actions, dim=-1)
    aug_data = {
        "observation.images.front": aug_images,
        "action": actions,  # only origin
        "action_is_pad": data["action_is_pad"],
        "observation.camera_ext.front": data["observation.camera_ext.front"],
        "observation.camera_int.front": data["observation.camera_int.front"],
        "observation.state": aug_state_kpts.reshape(-1, 8),
        "gripper_open": data["action"][:, -1] >= 0.02,
        "origin": aug_action_kpts[:, 0],
        "xaxis": aug_action_kpts[:, 1],
        "yaxis": aug_action_kpts[:, 2],
        "zaxis": aug_action_kpts[:, 3],
    }
    return aug_data


if __name__ == "__main__":
    import cv2
    import numpy as np
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    delta_timestamps = {
        "observation.images.front": [0.0],
        "observation.camera_ext.front": [0.0],
        "observation.camera_int.front": [0.0],
        "observation.state": [0.0],
        "action": [0.04 * i for i in range(16)],
    }

    dataset = LeRobotDataset(
        "changhaonan/RPLBenchData",
        split="train",
        delta_timestamps=delta_timestamps,
        image_transforms=None,
        video_backend="pyav",
        root=".",
    )
    data = dataset[0]
    horizon = len(data["action"])
    intrinsics = data["observation.camera_int.front"][0].reshape(3, 3).numpy().astype(np.float64)
    extrinsics = data["observation.camera_ext.front"][0].reshape(4, 4).numpy().astype(np.float64)
    extrinsics = np.linalg.inv(extrinsics)
    extrinsics[0] = -extrinsics[0]
    aug_transform = augmentation_recipe(spatial=1.0, color=0.0, mask=1.0, hflip=0, vflip=0, normalize=True, keypoints=True, to_tensor=True)

    rotations = Rotation.from_euler("xyz", data["action"][:, 3:6], degrees=False)
    pose7 = np.concatenate([data["action"][:, :3], rotations.as_quat()], axis=1)
    frames = pose7_to_frame(pose7)
    frames2d = cv2.projectPoints(frames.reshape(-1, 3), extrinsics[:3, :3], extrinsics[:3, 3], intrinsics, None)[0].reshape(horizon, 4, 2)

    # # THIS IMAGE is ranged in [0...1], or [0...255]
    _ = aug_transform(image=data["observation.images.front"][0].permute(1, 2, 0).numpy(), keypoints=frames2d.reshape(-1, 2))
    aug_image = _["image"]
    aug_frames2d = _["keypoints"].reshape(horizon, 4, 2)
    aug_frames2d = torch.from_numpy(aug_frames2d.clip(0, 511))
    frames2d = torch.from_numpy(frames2d)

    origin_batch = {
        "observation.images.front": normalize_bchw_image(data["observation.images.front"])[None],
        "action_is_pad": data["action_is_pad"][None],
        "gripper_open": data["action"][None, :, -1] >= 0.02,
        "origin": frames2d[None, :, 0],
        "xaxis": frames2d[None, :, 1],
        "yaxis": frames2d[None, :, 2],
        "zaxis": frames2d[None, :, 3],
    }

    aug_batch = {
        "observation.images.front": aug_image[None, None],
        "action_is_pad": data["action_is_pad"][None],
        "gripper_open": data["action"][None, :, -1] >= 0.02,
        "origin": aug_frames2d[None, :, 0],
        "xaxis": aug_frames2d[None, :, 1],
        "yaxis": aug_frames2d[None, :, 2],
        "zaxis": aug_frames2d[None, :, 3],
    }

    batch = {}
    for k in origin_batch.keys():
        batch[k] = torch.cat([origin_batch[k], aug_batch[k]], dim=0)

    model = ARPNetwork(horizon=16)
    model.train()

    print("=== Training ===")
    log_dict = model(batch)
    for i, img in enumerate(log_dict["visualized_images"]):
        img.save(f"vis{i}.jpg")

    print(log_dict)

    print("=== Inference ===")
    for k in ["action_is_pad", "gripper_open", "origin", "xaxis", "yaxis", "zaxis"]:
        batch.pop(k)

    with torch.no_grad():
        model.eval()
        out = model(batch)
        for i, img in enumerate(out["visualized_images"]):
            img.save(f"vis{i}.eval.jpg")
