import os
import warnings
from typing import Dict

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

try:
    from transformers import AutoImageProcessor, AutoModel
except ImportError:  # pragma: no cover
    AutoImageProcessor = None
    AutoModel = None

try:
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
except ImportError:  # pragma: no cover
    HfHubHTTPError = Exception
    RepositoryNotFoundError = Exception


class LastLevelMaxPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [nn.functional.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


def _make_group_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = 32
    while num_channels % num_groups != 0 and num_groups > 1:
        num_groups //= 2
    return nn.GroupNorm(num_groups, num_channels)


class SimpleFeaturePyramid(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int = 256,
            scale_factors=(4.0, 2.0, 1.0, 0.5),
            top_block=None,
    ):
        super().__init__()
        self.scale_factors = scale_factors
        self.top_block = top_block
        self.stages = nn.ModuleList()
        self._out_features = []
        self.out_channels = out_channels

        base_stride = 16
        for scale in scale_factors:
            layers = []
            stage_in_channels = in_channels

            if scale == 4.0:
                layers.extend(
                    [
                        nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2, bias=False),
                        _make_group_norm(in_channels // 2),
                        nn.GELU(),
                        nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2, bias=False),
                    ]
                )
                stage_in_channels = in_channels // 4
            elif scale == 2.0:
                layers.append(nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2, bias=False))
                stage_in_channels = in_channels // 2
            elif scale == 1.0:
                pass
            elif scale == 0.5:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                raise NotImplementedError("Unsupported scale factor: {}".format(scale))

            layers.extend(
                [
                    nn.Conv2d(stage_in_channels, out_channels, kernel_size=1, bias=False),
                    _make_group_norm(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    _make_group_norm(out_channels),
                ]
            )

            self.stages.append(nn.Sequential(*layers))
            stride = int(base_stride / scale)
            stage_num = int(torch.log2(torch.tensor(stride)).item())
            self._out_features.append("p{}".format(stage_num))

        if self.top_block is not None:
            last_stage = int(torch.log2(torch.tensor(int(base_stride / scale_factors[-1]))).item())
            for stage_id in range(last_stage, last_stage + self.top_block.num_levels):
                self._out_features.append("p{}".format(stage_id + 1))

        self._init_weights()

    def forward(self, feature_map: torch.Tensor) -> Dict[str, torch.Tensor]:
        results = [stage(feature_map) for stage in self.stages]
        if self.top_block is not None:
            results.extend(self.top_block(results[-1]))
        return {name: feat for name, feat in zip(self._out_features, results)}

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")


class DINOv3Backbone(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        if AutoModel is None:
            raise ImportError(
                "transformers is required to instantiate the DINOv3 backbone. "
                "Install `transformers` in the training environment."
            )
        from_pretrained_kwargs = self._get_from_pretrained_kwargs()
        try:
            self.model = AutoModel.from_pretrained(model_name, **from_pretrained_kwargs)
        except (OSError, RepositoryNotFoundError, HfHubHTTPError) as exc:
            raise RuntimeError(self._format_model_load_error(model_name, exc)) from exc
        self.processor = self._try_load_image_processor(model_name, from_pretrained_kwargs)
        self.out_channels = self.model.config.hidden_size
        self.image_mean, self.image_std = self._resolve_image_stats()
        patch_size = getattr(self.model.config, "patch_size", 16)
        if isinstance(patch_size, (tuple, list)):
            self.patch_size = (int(patch_size[0]), int(patch_size[1]))
        else:
            patch_size = int(patch_size)
            self.patch_size = (patch_size, patch_size)

    @staticmethod
    def _try_load_image_processor(model_name: str, from_pretrained_kwargs: Dict[str, object]):
        if AutoImageProcessor is None:
            return None
        try:
            return AutoImageProcessor.from_pretrained(model_name, **from_pretrained_kwargs)
        except ValueError as exc:
            warnings.warn(
                "Falling back to ImageNet normalization because no Hugging Face image processor "
                "is registered for '{}': {}".format(model_name, exc),
                RuntimeWarning,
            )
            return None

    def _resolve_image_stats(self):
        if self.processor is not None:
            return (
                tuple(float(value) for value in self.processor.image_mean),
                tuple(float(value) for value in self.processor.image_std),
            )
        # DINO-family vision backbones typically use ImageNet normalization.
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    @staticmethod
    def _get_from_pretrained_kwargs() -> Dict[str, object]:
        kwargs: Dict[str, object] = {}
        token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
        )
        if token:
            kwargs["use_auth_token"] = token

        offline_env = os.environ.get("TRANSFORMERS_OFFLINE", "").strip().lower()
        if offline_env in {"1", "true", "yes"}:
            kwargs["local_files_only"] = True
        return kwargs

    @staticmethod
    def _format_model_load_error(model_name: str, exc: Exception) -> str:
        details = str(exc)
        hints = [
            "Failed to load backbone '{}' via Hugging Face.".format(model_name),
            "This usually means one of the following:",
            "1. The repo id is wrong or no longer exists.",
            "2. The repo is gated/private and the current token is missing or expired.",
            "3. The job is running in offline mode without the model cached locally.",
            "Set `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` before launch if the repo requires auth.",
            "If the checkpoint was trained with a different backbone, pass that exact model name via "
            "`--detector_backbone_model` or store `backbone_name` in the checkpoint.",
        ]
        if "expired" in details.lower():
            hints.append("The current Hugging Face token appears to be expired.")
        hints.append("Original error: {}".format(details))
        return "\n".join(hints)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=x)
        tokens = outputs.last_hidden_state
        batch_size, seq_len, channels = tokens.shape

        patch_h, patch_w = self.patch_size
        grid_h = max(1, x.shape[-2] // patch_h)
        grid_w = max(1, x.shape[-1] // patch_w)
        spatial_tokens = grid_h * grid_w
        prefix_tokens = seq_len - spatial_tokens

        if prefix_tokens < 0:
            raise ValueError(
                "Backbone produced fewer tokens ({}) than expected spatial grid {}x{} ({}).".format(
                    seq_len, grid_h, grid_w, spatial_tokens
                )
            )

        patch_tokens = tokens[:, prefix_tokens:, :]
        if patch_tokens.shape[1] != spatial_tokens:
            raise ValueError(
                "Token/grid mismatch after removing {} prefix tokens: got {}, expected {}.".format(
                    prefix_tokens, patch_tokens.shape[1], spatial_tokens
                )
            )

        return patch_tokens.permute(0, 2, 1).contiguous().view(batch_size, channels, grid_h, grid_w)


class BackboneWithSimpleFPN(nn.Module):
    def __init__(self, base_backbone: nn.Module, fpn: nn.Module, freeze_backbone: bool = True):
        super().__init__()
        self.base_backbone = base_backbone
        self.fpn = fpn
        self.out_channels = fpn.out_channels
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for parameter in self.base_backbone.parameters():
                parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.base_backbone(x)
        else:
            features = self.base_backbone(x)
        return self.fpn(features)


def create_scenesayer_detector_model(
        num_classes: int,
        backbone_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        freeze_backbone: bool = True,
        min_size: int = 600,
        max_size: int = 1000,
):
    base_backbone = DINOv3Backbone(backbone_name)
    fpn = SimpleFeaturePyramid(
        in_channels=base_backbone.out_channels,
        out_channels=256,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        top_block=LastLevelMaxPool(),
    )
    backbone = BackboneWithSimpleFPN(base_backbone, fpn, freeze_backbone=freeze_backbone)
    featmap_names = backbone.fpn._out_features

    anchor_generator = AnchorGenerator(
        sizes=((16, 32, 64, 128, 256, 512, 1024),) * len(featmap_names),
        aspect_ratios=((0.5, 1.0, 2.0),) * len(featmap_names),
    )
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=featmap_names,
        output_size=7,
        sampling_ratio=2,
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=min_size,
        max_size=max_size,
        image_mean=base_backbone.image_mean,
        image_std=base_backbone.image_std,
    )
    return model
