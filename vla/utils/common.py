import math
import warnings
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as transform_functional


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class StandardViTPreprocessSpec:
    """Pixel-wise preprocessing contract for an ordinary RGB ViT."""

    rescale_factor: float
    image_mean: Tuple[float, float, float]
    image_std: Tuple[float, float, float]


@dataclass(frozen=True)
class CudaViTTransformCompatibility:
    """Result of checking whether a processor fits the CUDA ViT path."""

    supported: bool
    reason: str
    spec: Optional[StandardViTPreprocessSpec] = None


def _three_channel_processor_values(
    value: Any,
    *,
    name: str,
) -> Tuple[float, float, float]:
    if isinstance(value, Real) and not isinstance(value, bool):
        values = (float(value),) * 3
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(float(item) for item in value)
        if len(values) == 1:
            values = values * 3
    else:
        raise TypeError(f"{name} must be a number or a one/three-value sequence")
    if len(values) != 3:
        raise ValueError(f"{name} must contain one or three values, got {len(values)}")
    if not all(math.isfinite(item) for item in values):
        raise ValueError(f"{name} must contain only finite values")
    return values


def check_cuda_vit_transform_compatibility(
    processor: Any,
    *,
    vision_adapter: str = "standard",
) -> CudaViTTransformCompatibility:
    """Check and resolve a processor for fixed-square CUDA preprocessing.

    The standard custom pipeline already overrides processor resize and crop:
    every image is longest-side resized and padded before pixel-wise processor
    operations.  Consequently, a processor is compatible when its remaining
    contract can be represented by RGB rescale and normalization alone.
    """

    if vision_adapter != "standard":
        return CudaViTTransformCompatibility(
            supported=False,
            reason=(
                "the CUDA ViT transform only supports vision_adapter='standard'; "
                f"got {vision_adapter!r}"
            ),
        )
    if processor is None:
        return CudaViTTransformCompatibility(
            supported=False,
            reason="the model has no image_processor",
        )

    data_format = getattr(processor, "data_format", None)
    data_format = getattr(data_format, "value", data_format)
    if data_format not in (None, "channels_first"):
        return CudaViTTransformCompatibility(
            supported=False,
            reason=f"processor data_format must be channels_first, got {data_format!r}",
        )
    if getattr(processor, "do_pad", False) is True:
        return CudaViTTransformCompatibility(
            supported=False,
            reason="processor applies additional padding after the fixed-square transform",
        )
    for option in (
        "crop_to_patches",
        "dynamic_image_size",
        "use_thumbnail",
    ):
        if getattr(processor, option, False) is True:
            return CudaViTTransformCompatibility(
                supported=False,
                reason=f"processor enables unsupported dynamic image option {option!r}",
            )

    do_rescale = getattr(processor, "do_rescale", None)
    do_normalize = getattr(processor, "do_normalize", None)
    if not isinstance(do_rescale, bool):
        return CudaViTTransformCompatibility(
            supported=False,
            reason="processor.do_rescale must resolve to a boolean",
        )
    if not isinstance(do_normalize, bool):
        return CudaViTTransformCompatibility(
            supported=False,
            reason="processor.do_normalize must resolve to a boolean",
        )

    try:
        if do_rescale:
            raw_factor = getattr(processor, "rescale_factor", None)
            if not isinstance(raw_factor, Real) or isinstance(raw_factor, bool):
                raise TypeError("processor.rescale_factor must be a number")
            rescale_factor = float(raw_factor)
            if not math.isfinite(rescale_factor):
                raise ValueError("processor.rescale_factor must be finite")
        else:
            rescale_factor = 1.0

        if do_normalize:
            image_mean = _three_channel_processor_values(
                getattr(processor, "image_mean", None),
                name="processor.image_mean",
            )
            image_std = _three_channel_processor_values(
                getattr(processor, "image_std", None),
                name="processor.image_std",
            )
            if any(value <= 0.0 for value in image_std):
                raise ValueError("processor.image_std values must be positive")
        else:
            image_mean = (0.0, 0.0, 0.0)
            image_std = (1.0, 1.0, 1.0)
    except (TypeError, ValueError) as exc:
        return CudaViTTransformCompatibility(supported=False, reason=str(exc))

    return CudaViTTransformCompatibility(
        supported=True,
        reason="processor uses the standard fixed-square RGB pixel contract",
        spec=StandardViTPreprocessSpec(
            rescale_factor=rescale_factor,
            image_mean=image_mean,
            image_std=image_std,
        ),
    )


class CudaStandardViTTransform:
    """Batch ordinary RGB frames and preprocess them entirely on CUDA."""

    def __init__(
        self,
        processor: Any,
        input_size: int,
        device: torch.device,
        *,
        vision_adapter: str = "standard",
    ) -> None:
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError(f"CudaStandardViTTransform requires CUDA, got {device}")
        compatibility = check_cuda_vit_transform_compatibility(
            processor,
            vision_adapter=vision_adapter,
        )
        if not compatibility.supported or compatibility.spec is None:
            raise ValueError(
                "image processor is incompatible with CUDA ViT preprocessing: "
                + compatibility.reason
            )

        self.input_size = int(input_size)
        self.device = device
        self.spec = compatibility.spec
        mean = torch.tensor(
            self.spec.image_mean,
            device=device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            self.spec.image_std,
            device=device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        self.channel_scale = self.spec.rescale_factor / std
        self.channel_bias = -mean / std

    def __call__(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        if not frames:
            raise ValueError("frames must contain at least one RGB image")
        arrays = [np.asarray(frame) for frame in frames]
        expected_shape = arrays[0].shape
        for index, array in enumerate(arrays):
            if array.dtype != np.uint8:
                raise TypeError(
                    f"frame {index} must have dtype uint8, got {array.dtype}"
                )
            if array.ndim != 3 or array.shape[-1] != 3:
                raise ValueError(
                    f"frame {index} must have shape [H,W,3], got {array.shape}"
                )
            if array.shape != expected_shape:
                raise ValueError(
                    "all frames in a CUDA ViT batch must have the same shape; "
                    f"frame 0 is {expected_shape}, frame {index} is {array.shape}"
                )

        # Habitat observations originate in host memory. All pixel operations
        # after this single batched upload execute on the selected CUDA device.
        pixel_values = torch.from_numpy(np.stack(arrays, axis=0)).to(
            self.device,
            non_blocking=True,
        )
        pixel_values = pixel_values.permute(0, 3, 1, 2).contiguous()
        _, _, height, width = pixel_values.shape
        resized_height, resized_width, top, bottom, left, right = (
            resize_longest_side_and_pad_geometry(
                height,
                width,
                self.input_size,
            )
        )
        if (height, width) != (resized_height, resized_width):
            pixel_values = transform_functional.resize(
                pixel_values,
                [resized_height, resized_width],
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            )
        pixel_values = transform_functional.pad(
            pixel_values,
            [left, top, right, bottom],
            fill=0,
        )
        return (
            pixel_values.to(dtype=torch.float32)
            .mul_(self.channel_scale)
            .add_(self.channel_bias)
        )


def apply_qwen_vl_chat_template(
    processor: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    model_type: str,
    add_generation_prompt: bool = True,
) -> str:
    """Render a Qwen-VL prompt, with a minimal Qwen3 fallback.

    Official Qwen3-VL Instruct checkpoints ship ``chat_template.json``, but
    converted or selectively copied model directories sometimes omit it.  The
    TrackVLA prompt is a single user turn containing images followed by text,
    so that narrow format can be rendered without depending on the external
    template file. Other model types retain the processor's native behavior.
    """
    processor_template = getattr(processor, "chat_template", None)
    if processor_template:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    tokenizer = getattr(processor, "tokenizer", None)
    tokenizer_template = getattr(tokenizer, "chat_template", None)
    if tokenizer_template:
        return processor.apply_chat_template(
            messages,
            chat_template=tokenizer_template,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    if str(model_type).lower() != "qwen3vl":
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    if len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError(
            "The built-in Qwen3-VL TrackVLA chat-template fallback supports "
            "exactly one user message"
        )
    content = messages[0].get("content")
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        raise ValueError(
            "The built-in Qwen3-VL TrackVLA chat-template fallback requires "
            "a sequence of image/text content items"
        )

    parts = ["<|im_start|>user\n"]
    for item in content:
        if not isinstance(item, Mapping):
            raise ValueError("Qwen3-VL message content items must be mappings")
        item_type = item.get("type")
        if item_type == "image" or "image" in item or "image_url" in item:
            parts.append("<|vision_start|><|image_pad|><|vision_end|>")
        elif item_type == "text" or "text" in item:
            parts.append(str(item.get("text", "")))
        else:
            raise ValueError(
                "Unsupported Qwen3-VL TrackVLA message content type: "
                f"{item_type!r}"
            )
    parts.append("<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    warnings.warn(
        "Qwen3-VL processor has no chat template; using the built-in "
        "single-turn TrackVLA template. Copy chat_template.json from the "
        "official Instruct checkpoint to restore the full template.",
        stacklevel=2,
    )
    return "".join(parts)


def resize_longest_side_and_pad_geometry(
    height: int,
    width: int,
    output_size: int,
) -> Tuple[int, int, int, int, int, int]:
    """Return resize and symmetric padding geometry for a square canvas.

    The longest image side is resized to ``output_size``. The other side is
    scaled without changing the aspect ratio and then padded equally on both
    sides. When the remaining pixel count is odd, the extra pixel is placed on
    the bottom or right.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"Expected a non-empty image, got {height}x{width}")
    if output_size <= 0:
        raise ValueError(f"output_size must be positive, got {output_size}")

    if width >= height:
        resized_width = output_size
        resized_height = max(1, round(height * output_size / width))
    else:
        resized_height = output_size
        resized_width = max(1, round(width * output_size / height))

    pad_height = output_size - resized_height
    pad_width = output_size - resized_width
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left
    return resized_height, resized_width, top, bottom, left, right


class ResizeLongestSideAndPad:
    """Resize the longest side and symmetrically pad to a square image."""

    def __init__(
        self,
        output_size: int,
        interpolation=transforms.InterpolationMode.BICUBIC,
        fill: int = 0,
    ) -> None:
        if output_size <= 0:
            raise ValueError(f"output_size must be positive, got {output_size}")
        self.output_size = int(output_size)
        self.interpolation = interpolation
        self.fill = fill

    def __call__(self, image):
        _, height, width = transform_functional.get_dimensions(image)
        resized_height, resized_width, top, bottom, left, right = (
            resize_longest_side_and_pad_geometry(
                height,
                width,
                self.output_size,
            )
        )
        image = transform_functional.resize(
            image,
            [resized_height, resized_width],
            interpolation=self.interpolation,
            antialias=True,
        )
        return transform_functional.pad(
            image,
            [left, top, right, bottom],
            fill=self.fill,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(output_size={self.output_size}, "
            f"interpolation={self.interpolation}, fill={self.fill})"
        )


def preprocess_custom_images(
    processor: Any,
    images: Sequence[Image.Image],
    input_size: int,
) -> torch.Tensor:
    """Prepare fixed-square inputs for ordinary ViTs and Qwen3-VL vision towers.

    Ordinary image processors return ``[N,C,H,W]``.  Qwen3-VL processors return
    patchified ``[N*P,D]`` plus ``image_grid_thw``; retain one patch matrix per
    source image so the standard custom dataloader can still batch it safely.
    """
    if processor is None or not callable(processor):
        raise ValueError("A callable image processor is required for model_type='custom'")
    square_transform = ResizeLongestSideAndPad(input_size)
    square_images = [square_transform(image) for image in images]
    processed = processor(
        images=square_images,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    )
    if "pixel_values" not in processed:
        raise KeyError("The custom ViT image processor returned no pixel_values")
    pixel_values = processed["pixel_values"]
    image_grid_thw = processed.get("image_grid_thw")
    if image_grid_thw is not None:
        if (
            not isinstance(pixel_values, torch.Tensor)
            or pixel_values.ndim != 2
            or not isinstance(image_grid_thw, torch.Tensor)
            or image_grid_thw.ndim != 2
            or tuple(image_grid_thw.shape) != (len(images), 3)
        ):
            raise ValueError(
                "A patchified custom processor must return pixel_values [N*P,D] "
                "and image_grid_thw [N,3]"
            )
        patches_per_image = image_grid_thw.prod(dim=1)
        if not torch.equal(patches_per_image, patches_per_image[:1].expand_as(patches_per_image)):
            raise ValueError(
                "Custom Qwen3-VL preprocessing requires the same patch grid for every frame"
            )
        patches = int(patches_per_image[0].item())
        if patches <= 0 or pixel_values.shape[0] != len(images) * patches:
            raise ValueError(
                "Custom Qwen3-VL pixel_values do not agree with image_grid_thw"
            )
        return pixel_values.reshape(len(images), patches, pixel_values.shape[-1])
    if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 4:
        raise ValueError("The custom ViT processor must return pixel_values [H,C,S,S]")
    if tuple(pixel_values.shape[-2:]) != (input_size, input_size):
        raise ValueError(
            "The custom ViT processor changed the fixed input geometry: "
            f"expected {(input_size, input_size)}, got {tuple(pixel_values.shape[-2:])}"
        )
    return pixel_values


def build_custom_multimodal_prompt(
    tokenizer: Any,
    image_token: str,
    image_token_counts: Sequence[int],
    prompt: str,
) -> str:
    """Render fixed visual slots using the image token resolved by the LLM loader."""
    if not isinstance(image_token, str) or not image_token:
        raise ValueError("image_token must be a non-empty string")
    if not image_token_counts or any(int(count) <= 0 for count in image_token_counts):
        raise ValueError("image_token_counts must contain positive integers")
    all_special_tokens = set(getattr(tokenizer, "all_special_tokens", ()) or ())
    use_internvl_wrapper = (
        image_token == "<IMG_CONTEXT>"
        and "<img>" in all_special_tokens
        and "</img>" in all_special_tokens
    )
    blocks = []
    for count in image_token_counts:
        body = image_token * int(count)
        blocks.append(f"<img>{body}</img>" if use_internvl_wrapper else body)
    image_blocks = "\n".join(blocks)
    return f"{image_blocks}\nUser: {prompt}\nAssistant:"

def build_transform(input_size: int = 448):
    """InternVL image transform: longest-side resize + pad + normalize."""
    return transforms.Compose([
        ResizeLongestSideAndPad(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_prompt(tokenizer, num_images: int, question: str = "Predict the future trajectory of the ego vehicle.") -> str:
    """Build a text prompt with <image> placeholders for each history frame."""
    image_tokens = "\n".join(["<image>"] * num_images)
    return f"{image_tokens}\nUser: {question}\nAssistant:"


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")
