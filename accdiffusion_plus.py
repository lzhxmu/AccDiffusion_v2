# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
import numpy as np
import random
import warnings
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from utils import *

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    LoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import AutoencoderKL, UNet2DConditionModel, ControlNetModel
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    is_accelerate_available,
    is_accelerate_version,
    is_invisible_watermark_available,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor, is_compiled_module
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from diffusers import ControlNetModel, AutoencoderKL
from accelerate.utils import set_seed
from tqdm import tqdm
if is_invisible_watermark_available():
    from .watermark import StableDiffusionXLWatermarker
import cv2

from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionXLPipeline

        >>> pipe = StableDiffusionXLPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""



def gaussian_kernel(kernel_size=3, sigma=1.0, channels=3):
    x_coord = torch.arange(kernel_size)
    gaussian_1d = torch.exp(-(x_coord - (kernel_size - 1) / 2) ** 2 / (2 * sigma ** 2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
    kernel = gaussian_2d[None, None, :, :].repeat(channels, 1, 1, 1)
    
    return kernel

def gaussian_filter(latents, kernel_size=3, sigma=1.0):
    channels = latents.shape[1]
    kernel = gaussian_kernel(kernel_size, sigma, channels).to(latents.device, latents.dtype)
    blurred_latents = F.conv2d(latents, kernel, padding=kernel_size//2, groups=channels)
    return blurred_latents

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class AccDiffusionv2SDXLPipeline(DiffusionPipeline, FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin):
    """
    Pipeline for text-to-image generation using Stable Diffusion XL.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    In addition the pipeline inherits the following loading methods:
        - *LoRA*: [`StableDiffusionXLPipeline.load_lora_weights`]
        - *Ckpt*: [`loaders.FromSingleFileMixin.from_single_file`]

    as well as the following saving methods:
        - *LoRA*: [`loaders.StableDiffusionXLPipeline.save_lora_weights`]

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion XL uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([` CLIPTextModelWithProjection`]):
            Second frozen text-encoder. Stable Diffusion XL uses the text and pool portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):
            Whether the negative prompt embeddings shall be forced to always be set to 0. Also see the config of
            `stabilityai/stable-diffusion-xl-base-1-0`.
        add_watermarker (`bool`, *optional*):
            Whether to use the [invisible_watermark library](https://github.com/ShieldMnt/invisible-watermark/) to
            watermark output images. If not defined, it will default to True if the package is installed, otherwise no
            watermarker will be used.
    """
    model_cpu_offload_seq = "text_encoder->text_encoder_2->unet->vae"

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
        scheduler: KarrasDiffusionSchedulers,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.default_sample_size = self.unet.config.sample_size
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

        if add_watermarker:
            self.watermark = StableDiffusionXLWatermarker()
        else:
            self.watermark = None
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_tiling
    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_tiling
    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def encode_prompt(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            adjust_lora_scale_text_encoder(self.text_encoder_2, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            # textual inversion: procecss multi-vector tokens if necessary
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, tokenizer)

                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )

                prompt_embeds = text_encoder(
                    text_input_ids.to(device),
                    output_hidden_states=True,
                )

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt, negative_prompt_2]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    negative_prompt = self.maybe_convert_prompt(negative_prompt, tokenizer)

                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device),
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        num_images_per_prompt=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

        if max(height, width) % 1024 != 0:
            raise ValueError(f"the larger one of `height` and `width` has to be divisible by 1024 but are {height} and {width}.")

        if num_images_per_prompt != 1:
            warnings.warn("num_images_per_prompt != 1 is not supported by AccDiffusion v2 and will be ignored.")
            num_images_per_prompt = 1

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + self.text_encoder_2.config.projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. \
                The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def get_views(self, height, width, window_size=128, stride=64, random_jitter=False):
        # Here, we define the mappings F_i (see Eq. 7 in the MultiDiffusion paper https://arxiv.org/abs/2302.08113)
        # if panorama's height/width < window_size, num_blocks of height/width should return 1
        height //= self.vae_scale_factor
        width //= self.vae_scale_factor
        num_blocks_height = int((height - window_size) / stride - 1e-6) + 2 if height > window_size else 1
        num_blocks_width = int((width - window_size) / stride - 1e-6) + 2 if width > window_size else 1
        total_num_blocks = int(num_blocks_height * num_blocks_width)
        views = []
        for i in range(total_num_blocks):
            h_start = int((i // num_blocks_width) * stride)
            h_end = h_start + window_size
            w_start = int((i % num_blocks_width) * stride)
            w_end = w_start + window_size

            if h_end > height:
                h_start = int(h_start + height - h_end)
                h_end = int(height)
            if w_end > width:
                w_start = int(w_start + width - w_end)
                w_end = int(width)
            if h_start < 0:
                h_end = int(h_end - h_start)
                h_start = 0
            if w_start < 0:
                w_end = int(w_end - w_start)
                w_start = 0

            if random_jitter:
                jitter_range = (window_size - stride) // 4
                w_jitter = 0
                h_jitter = 0
                if (w_start != 0) and (w_end != width):
                    w_jitter = random.randint(-jitter_range, jitter_range)
                elif (w_start == 0) and (w_end != width):
                    w_jitter = random.randint(-jitter_range, 0)
                elif (w_start != 0) and (w_end == width):
                    w_jitter = random.randint(0, jitter_range)

                if (h_start != 0) and (h_end != height):
                    h_jitter = random.randint(-jitter_range, jitter_range)
                elif (h_start == 0) and (h_end != height):
                    h_jitter = random.randint(-jitter_range, 0)
                elif (h_start != 0) and (h_end == height):
                    h_jitter = random.randint(0, jitter_range)
                # When using jitter, the noise will be padded by jitterrange, so we need to add it to the view.
                h_start = h_start + h_jitter + jitter_range
                h_end = h_end + h_jitter + jitter_range
                w_start = w_start + w_jitter + jitter_range
                w_end = w_end + w_jitter + jitter_range
            
            views.append((h_start, h_end, w_start, w_end))
        return views

        
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)


    def register_attention_control(self, controller):
        attn_procs = {}
        cross_att_count = 0
        ori_attn_processors = self.unet.attn_processors
        for name in self.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue
            cross_att_count += 1
            attn_procs[name] = P2PCrossAttnProcessor(controller=controller, place_in_unet=place_in_unet)

        self.unet.set_attn_processor(attn_procs)
        controller.num_att_layers = cross_att_count
        return ori_attn_processors

    def recover_attention_control(self, ori_attn_processors):
        self.unet.set_attn_processor(ori_attn_processors)



    # Overrride to properly handle the loading and unloading of the additional text encoder.
    def load_lora_weights(self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]], **kwargs):
        # We could have accessed the unet config from `lora_state_dict()` too. We pass
        # it here explicitly to be able to tell that it's coming from an SDXL
        # pipeline.

        # Remove any existing hooks.
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate.hooks import AlignDevicesHook, CpuOffload, remove_hook_from_module
        else:
            raise ImportError("Offloading requires `accelerate v0.17.0` or higher.")

        is_model_cpu_offload = False
        is_sequential_cpu_offload = False
        recursive = False
        for _, component in self.components.items():
            if isinstance(component, torch.nn.Module):
                if hasattr(component, "_hf_hook"):
                    is_model_cpu_offload = isinstance(getattr(component, "_hf_hook"), CpuOffload)
                    is_sequential_cpu_offload = isinstance(getattr(component, "_hf_hook"), AlignDevicesHook)
                    logger.info(
                        "Accelerate hooks detected. Since you have called `load_lora_weights()`, the previous hooks will be first removed. Then the LoRA parameters will be loaded and the hooks will be applied again."
                    )
                    recursive = is_sequential_cpu_offload
                    remove_hook_from_module(component, recurse=recursive)
        state_dict, network_alphas = self.lora_state_dict(
            pretrained_model_name_or_path_or_dict,
            unet_config=self.unet.config,
            **kwargs,
        )
        self.load_lora_into_unet(state_dict, network_alphas=network_alphas, unet=self.unet)

        text_encoder_state_dict = {k: v for k, v in state_dict.items() if "text_encoder." in k}
        if len(text_encoder_state_dict) > 0:
            self.load_lora_into_text_encoder(
                text_encoder_state_dict,
                network_alphas=network_alphas,
                text_encoder=self.text_encoder,
                prefix="text_encoder",
                lora_scale=self.lora_scale,
            )

        text_encoder_2_state_dict = {k: v for k, v in state_dict.items() if "text_encoder_2." in k}
        if len(text_encoder_2_state_dict) > 0:
            self.load_lora_into_text_encoder(
                text_encoder_2_state_dict,
                network_alphas=network_alphas,
                text_encoder=self.text_encoder_2,
                prefix="text_encoder_2",
                lora_scale=self.lora_scale,
            )

        # Offload back.
        if is_model_cpu_offload:
            self.enable_model_cpu_offload()
        elif is_sequential_cpu_offload:
            self.enable_sequential_cpu_offload()

    @classmethod
    def save_lora_weights(
        self,
        save_directory: Union[str, os.PathLike],
        unet_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
        text_encoder_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
        text_encoder_2_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = True,
    ):
        state_dict = {}

        def pack_weights(layers, prefix):
            layers_weights = layers.state_dict() if isinstance(layers, torch.nn.Module) else layers
            layers_state_dict = {f"{prefix}.{module_name}": param for module_name, param in layers_weights.items()}
            return layers_state_dict

        if not (unet_lora_layers or text_encoder_lora_layers or text_encoder_2_lora_layers):
            raise ValueError(
                "You must pass at least one of `unet_lora_layers`, `text_encoder_lora_layers` or `text_encoder_2_lora_layers`."
            )

        if unet_lora_layers:
            state_dict.update(pack_weights(unet_lora_layers, "unet"))

        if text_encoder_lora_layers and text_encoder_2_lora_layers:
            state_dict.update(pack_weights(text_encoder_lora_layers, "text_encoder"))
            state_dict.update(pack_weights(text_encoder_2_lora_layers, "text_encoder_2"))

        self.write_lora_layers(
            state_dict=state_dict,
            save_directory=save_directory,
            is_main_process=is_main_process,
            weight_name=weight_name,
            save_function=save_function,
            safe_serialization=safe_serialization,
        )

    def _remove_text_encoder_monkey_patch(self):
        self._remove_text_encoder_monkey_patch_classmethod(self.text_encoder)
        self._remove_text_encoder_monkey_patch_classmethod(self.text_encoder_2)

        # Copied from diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline.prepare_image
    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    def decode_latents(self, latents: torch.FloatTensor, result_path: str, t):
        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        if needs_upcasting:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        if height > 2048 or width > 2048:
            self.enable_vae_tiling()
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

        image = self.image_processor.postprocess(image, output_type='pil')

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        if os.path.exists(result_path) == False:
            os.makedirs(result_path)

        image[0].save(f'{result_path}/{t}.png')


    def viusalize_latents(self, latents: torch.FloatTensor, result_path: str, t: int):
        mean_latents = latents.mean(dim=1).squeeze().to(torch.float32)
        mean_latents_np = mean_latents.cpu().numpy()
        mean_latents_np = (mean_latents_np - mean_latents_np.min()) / (mean_latents_np.max() - mean_latents_np.min()) * 255
        image = Image.fromarray(mean_latents_np)
        image = image.convert("L")
        if os.path.exists(result_path) == False:
            os.makedirs(result_path)
        image.save(f'{result_path}/mean_latents_{t}.png')

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = False,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        ################### AccDiffusion v2 specific parameters ####################
        image_lr: Optional[torch.FloatTensor] = None,
        view_batch_size: int = 16,
        multi_decoder: bool = True,
        stride: Optional[int] = 64,
        cosine_scale_1: Optional[float] = 3.,
        cosine_scale_2: Optional[float] = 1.,
        cosine_scale_3: Optional[float] = 1.,
        sigma: Optional[float] = 1.0,
        lowvram: bool = False,
        multi_guidance_scale: Optional[float] = 7.5,
        use_guassian: bool = True,
        upscale_mode: Union[str, List[str]] = 'bicubic_latent',
        use_multidiffusion: bool = True,
        use_dilated_sampling : bool = True,
        use_skip_residual: bool = True,
        use_progressive_upscaling: bool = True,
        shuffle: bool = False,
        result_path: str = './outputs/AccDiffusionv2',
        debug: bool = False,
        use_md_prompt: bool = False,
        attn_res=None,
        save_attention_map: bool = False,
        seed: Optional[int] = None,
        c : Optional[float] = 0.3,
        use_controlnet = True,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        controlnet_conditioning_scale: Union[float, List[float]] = 0.5,
        canny_sigma: Optional[float] = 0.6,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            denoising_end (`float`, *optional*):
                When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
                completed before it is intentionally prematurely terminated. As a result, the returned sample will
                still retain a substantial amount of noise as determined by the discrete timesteps selected by the
                scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
                "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
                Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(width, height)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(width, height)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a target image resolution. It should be as same
                as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            ################### AccDiffusion v2 specific parameters ####################
            # We build AccDiffusion v2 based on Demofusion pipeline (see paper: https://arxiv.org/pdf/2311.16973.pdf)
            image_lr (`torch.FloatTensor`, *optional*, , defaults to None):
                Low-resolution image input for upscaling. 
            view_batch_size (`int`, defaults to 16):
                The batch size for multiple denoising paths. Typically, a larger batch size can result in higher 
                efficiency but comes with increased GPU memory requirements.
            multi_decoder (`bool`, defaults to True):
                Determine whether to use a tiled decoder. Generally, when the resolution exceeds 3072x3072, 
                a tiled decoder becomes necessary.
            stride (`int`, defaults to 64):
                The stride of moving local patches. A smaller stride is better for alleviating seam issues,
                but it also introduces additional computational overhead and inference time.
            cosine_scale_1 (`float`, defaults to 3):
                Control the strength of skip-residual. For specific impacts, please refer to Appendix C
                in the DemoFusion paper. (see paper : https://arxiv.org/pdf/2311.16973.pdf)
            cosine_scale_2 (`float`, defaults to 1):
                Control the strength of dilated sampling. For specific impacts, please refer to Appendix C
                in the DemoFusion paper.(see paper : https://arxiv.org/pdf/2311.16973.pdf)
            cosine_scale_3 (`float`, defaults to 1):
                Control the strength of the gaussion filter. For specific impacts, please refer to Appendix C
                in the DemoFusion paper.(see paper : https://arxiv.org/pdf/2311.16973.pdf)
            sigma (`float`, defaults to 1):
                The standard value of the gaussian filter.
            show_image (`bool`, defaults to False):
                Determine whether to show intermediate results during generation.
            lowvram (`bool`, defaults to False):
                Try to fit in 8 Gb of VRAM, with xformers installed.

        Examples:

        Returns:
            a `list` with the generated images at each phase.
        """

        if debug :
            num_inference_steps = 1

        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
                control_guidance_end
            ]

        
        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        x1_size = self.default_sample_size * self.vae_scale_factor

        height_scale = height / x1_size
        width_scale = width / x1_size
        scale_num = int(max(height_scale, width_scale))
        aspect_ratio = min(height_scale, width_scale) / max(height_scale, width_scale)

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        if attn_res is None:
            attn_res = int(np.ceil(self.default_sample_size * self.vae_scale_factor / 32)), int(np.ceil(self.default_sample_size * self.vae_scale_factor / 32))
        self.attn_res = attn_res

        if lowvram:
            attention_map_device = torch.device("cpu")
        else:
            attention_map_device = self.device

        self.controller = create_controller(
            prompt, cross_attention_kwargs, num_inference_steps, tokenizer=self.tokenizer, device=attention_map_device, attn_res=self.attn_res
        )

        if (save_attention_map or use_md_prompt ):
            ori_attn_processors = self.register_attention_control(self.controller)  # add attention controller

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            num_images_per_prompt,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        self.lowvram = lowvram
        if self.lowvram:
            self.vae.cpu()
            self.unet.cpu()
            self.text_encoder.to(device)
            self.text_encoder_2.to(device)
            # image_lr.cpu()

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height // scale_num,
            width // scale_num,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )


        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds

        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )

        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
            )
        else:
            negative_add_time_ids = add_time_ids

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0).to(device)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0).to(device)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0).to(device).repeat(batch_size * num_images_per_prompt, 1)

        del negative_prompt_embeds, negative_pooled_prompt_embeds, negative_add_time_ids

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)


        # 9 Apply denoising_end
        if denoising_end is not None and isinstance(denoising_end, float) and denoising_end > 0 and denoising_end < 1:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        output_images = []

        # 10 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)
        
    ###################################################### Phase Initialization ########################################################

        if self.lowvram:
            self.text_encoder.cpu()
            self.text_encoder_2.cpu()

        if image_lr == None:
            print("### Phase 1 Denoising ###")
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
    
                    if self.lowvram:
                        self.vae.cpu()
                        self.unet.to(device)
    
                    latents_for_view = latents
    
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        latents.repeat_interleave(2, dim=0)
                        if do_classifier_free_guidance
                        else latents
                    )
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        # cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
    
                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred[::2], noise_pred[1::2]
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    
                    if do_classifier_free_guidance and guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
    
                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
    
                    # # step callback
                    if t == 1 and use_md_prompt:
                        md_prompts, views_attention = get_multidiffusion_prompts(tokenizer=self.tokenizer, prompts=[prompt], 
                                                                                 threthod=c,attention_store=self.controller, 
                                                                                 height=height//scale_num, width =width//scale_num, 
                                                                                 from_where=["up","down"], random_jitter=True, 
                                                                                 scale_num=scale_num)
                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)

            del latents_for_view, latent_model_input, noise_pred, noise_pred_text, noise_pred_uncond
            if (use_md_prompt or save_attention_map):
                self.recover_attention_control(ori_attn_processors=ori_attn_processors)  # recover attention controller
                del self.controller
            torch.cuda.empty_cache()
        else:
            print("### Encoding Real Image ###")
            latents = self.vae.encode(image_lr)
            latents = latents.latent_dist.sample() * self.vae.config.scaling_factor

        anchor_mean = latents.mean()
        anchor_std = latents.std()
        if self.lowvram:
            latents = latents.cpu()
            torch.cuda.empty_cache()
        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            
            if self.lowvram:
                needs_upcasting = False # use madebyollin/sdxl-vae-fp16-fix in lowvram mode!
                self.unet.cpu()
                self.vae.to(device)

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            if self.lowvram and multi_decoder:
                current_width_height = self.unet.config.sample_size * self.vae_scale_factor
                image = self.tiled_decode(latents, current_width_height, current_width_height)
            else:
                image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
               

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
                torch.cuda.empty_cache()
    
        image = self.image_processor.postprocess(image, output_type=output_type)
        if not os.path.exists(f'{result_path}'):
            os.makedirs(f'{result_path}')

        image_lr_save_path = f'{result_path}/{image[0].size[0]}_{image[0].size[1]}.png'

        if use_controlnet:
            canny_image = np.array(image[0])

            canny_image = auto_canny(canny_image, sigma=canny_sigma)

            canny_image = canny_image[:, :, None]
            canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
            canny_image = Image.fromarray(canny_image)
            canny_image.save(f'{result_path}/AccDiffusion_cany.png')
            condition_image = self.prepare_image(
                image=canny_image,
                width=width // scale_num,
                height=height // scale_num,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=False,
            )
        image[0].save(image_lr_save_path)
        output_images.append(image[0])

    ####################################################### Phase Upscaling #####################################################
        if use_progressive_upscaling:
            if image_lr == None:
                starting_scale = 2
            else:
                starting_scale = 1
        else:
            starting_scale = scale_num
        for current_scale_num in range(starting_scale, scale_num + 1):
            if self.lowvram:
                latents = latents.to(device)
                self.unet.to(device)
                torch.cuda.empty_cache()
            
            current_height = self.unet.config.sample_size * self.vae_scale_factor * current_scale_num
            current_width = self.unet.config.sample_size * self.vae_scale_factor * current_scale_num

            if height > width:
                current_width = int(current_width * aspect_ratio)
            else:
                current_height = int(current_height * aspect_ratio)

            
            if upscale_mode == "bicubic_latent" or debug:
                latents = F.interpolate(latents.to(device), size=(int(current_height / self.vae_scale_factor), int(current_width / self.vae_scale_factor)), mode='bicubic')
            else:
                raise NotImplementedError

            if use_controlnet:
                condition_image = F.interpolate(condition_image.to(device), size=(current_height, current_width), mode='bicubic')

            
            print("### Phase {} Denoising ###".format(current_scale_num))
            ############################################# noise inverse #############################################
            noise_latents = []
            noise = torch.randn_like(latents)
            for timestep in timesteps:
                noise_latent = self.scheduler.add_noise(latents, noise, timestep.unsqueeze(0))
                noise_latents.append(noise_latent)
            latents = noise_latents[0]

            ############################################# denoise #############################################
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    count = torch.zeros_like(latents)
                    value = torch.zeros_like(latents)
                    cosine_factor = 0.5 * (1 + torch.cos(torch.pi * (self.scheduler.config.num_train_timesteps - t) / self.scheduler.config.num_train_timesteps)).cpu()
                    if use_skip_residual:
                        c1 = cosine_factor ** cosine_scale_1
                        latents = latents * (1 - c1) + noise_latents[i] * c1
                    
                    if use_multidiffusion:
                        ############################################# MultiDiffusion #############################################
                        if use_md_prompt:
                            md_prompt_embeds_list = []
                            md_add_text_embeds_list = []
                            for md_prompt in md_prompts[current_scale_num]:
                                (  
                                    md_prompt_embeds,
                                    md_negative_prompt_embeds,
                                    md_pooled_prompt_embeds,
                                    md_negative_pooled_prompt_embeds,
                                ) = self.encode_prompt(
                                    prompt=md_prompt,
                                    prompt_2=prompt_2,
                                    device=device,
                                    num_images_per_prompt=num_images_per_prompt,
                                    do_classifier_free_guidance=do_classifier_free_guidance,
                                    negative_prompt=negative_prompt,
                                    negative_prompt_2=negative_prompt_2,
                                    prompt_embeds=None,
                                    negative_prompt_embeds=None,
                                    pooled_prompt_embeds=None,
                                    negative_pooled_prompt_embeds=None,
                                    lora_scale=text_encoder_lora_scale,
                                )
                                md_prompt_embeds_list.append(torch.cat([md_negative_prompt_embeds, md_prompt_embeds], dim=0).to(device))
                                md_add_text_embeds_list.append(torch.cat([md_negative_pooled_prompt_embeds, md_pooled_prompt_embeds], dim=0).to(device))
                                del md_negative_prompt_embeds, md_negative_pooled_prompt_embeds

                        if use_md_prompt:
                            random_jitter = True
                            views = [(h_start*4, h_end*4, w_start*4, w_end*4) for h_start, h_end, w_start, w_end in views_attention[current_scale_num]]
                        else:
                            random_jitter = True
                            views = self.get_views(current_height, current_width, stride=stride, window_size=self.unet.config.sample_size, random_jitter=random_jitter)

                        views_batch = [views[i : i + view_batch_size] for i in range(0, len(views), view_batch_size)]

                        if use_md_prompt:
                            views_prompt_embeds_input = [md_prompt_embeds_list[i : i + view_batch_size] for i in range(0, len(views), view_batch_size)]
                            views_add_text_embeds_input = [md_add_text_embeds_list[i : i + view_batch_size] for i in range(0, len(views), view_batch_size)]

                        if random_jitter:
                            jitter_range = int((self.unet.config.sample_size - stride) // 4)
                            latents_ = F.pad(latents, (jitter_range, jitter_range, jitter_range, jitter_range), 'constant', 0)
                            if use_controlnet:
                                condition_image_ = F.pad(condition_image, (jitter_range * self.vae_scale_factor, jitter_range * self.vae_scale_factor, jitter_range * self.vae_scale_factor, jitter_range * self.vae_scale_factor), 'constant', 0)
                        else:
                            latents_ = latents

                        count_local = torch.zeros_like(latents_)
                        value_local = torch.zeros_like(latents_)
                        
                        for j, batch_view in enumerate(views_batch):
                            vb_size = len(batch_view)
                            # get the latents corresponding to the current view coordinates
                            latents_for_view = torch.cat(
                                [
                                    latents_[:, :, h_start:h_end, w_start:w_end]
                                    for h_start, h_end, w_start, w_end in batch_view
                                ]
                            )
                            if use_controlnet:
                                condition_image_for_view = torch.cat(
                                    [
                                        condition_image_[0:1, :, h_start * self.vae_scale_factor:h_end * self.vae_scale_factor, w_start * self.vae_scale_factor:w_end * self.vae_scale_factor]
                                        for h_start, h_end, w_start, w_end in batch_view
                                    ]
                                )

                            # expand the latents if we are doing classifier free guidance
                            latent_model_input = latents_for_view
                            latent_model_input = (
                                latent_model_input.repeat_interleave(2, dim=0)
                                if do_classifier_free_guidance
                                else latent_model_input
                            )
                            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                            if use_controlnet:
                                condition_image_input = condition_image_for_view
                                condition_image_input = (
                                    condition_image_input.repeat_interleave(2, dim=0)
                                    if do_classifier_free_guidance
                                    else condition_image_input
                                )

                            
                            add_time_ids_input = []
                            for h_start, h_end, w_start, w_end in batch_view:
                                add_time_ids_ = add_time_ids.clone()
                                add_time_ids_[:, 2] = h_start * self.vae_scale_factor
                                add_time_ids_[:, 3] = w_start * self.vae_scale_factor
                                add_time_ids_input.append(add_time_ids_)
                            add_time_ids_input = torch.cat(add_time_ids_input)

                            if not use_md_prompt:
                                prompt_embeds_input = torch.cat([prompt_embeds] * vb_size)
                                add_text_embeds_input = torch.cat([add_text_embeds] * vb_size)
                                # predict the noise residual
                                added_cond_kwargs = {"text_embeds": add_text_embeds_input, "time_ids": add_time_ids_input}
                                noise_pred = self.unet(
                                    latent_model_input,
                                    t,
                                    encoder_hidden_states=prompt_embeds_input,
                                    # cross_attention_kwargs=cross_attention_kwargs,
                                    added_cond_kwargs=added_cond_kwargs,
                                    return_dict=False,
                                )[0]
                            else:
                                md_prompt_embeds_input = torch.cat(views_prompt_embeds_input[j])
                                md_add_text_embeds_input = torch.cat(views_add_text_embeds_input[j])
                                md_added_cond_kwargs = {"text_embeds": md_add_text_embeds_input, "time_ids": add_time_ids_input}

                                if use_controlnet:
                                    control_model_input = latent_model_input
                                    controlnet_prompt_embeds = md_prompt_embeds_input
                                    controlnet_added_cond_kwargs = md_added_cond_kwargs

                                    if isinstance(controlnet_keep[i], list):
                                        cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                                    else:
                                        controlnet_cond_scale = controlnet_conditioning_scale
                                        if isinstance(controlnet_cond_scale, list):
                                            controlnet_cond_scale = controlnet_cond_scale[0]
                                        cond_scale = controlnet_cond_scale * controlnet_keep[i]

                                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                                        control_model_input,
                                        t,
                                        encoder_hidden_states=controlnet_prompt_embeds,
                                        controlnet_cond=condition_image_input,
                                        conditioning_scale=cond_scale,
                                        guess_mode=False,
                                        added_cond_kwargs=controlnet_added_cond_kwargs,
                                        return_dict=False,
                                    )
                                    noise_pred = self.unet(
                                        latent_model_input,
                                        t,
                                        encoder_hidden_states=md_prompt_embeds_input,
                                        # cross_attention_kwargs=cross_attention_kwargs,
                                        down_block_additional_residuals=down_block_res_samples,
                                        mid_block_additional_residual=mid_block_res_sample,
                                        added_cond_kwargs=md_added_cond_kwargs,
                                        return_dict=False,
                                    )[0]
                                else:
                                    noise_pred = self.unet(
                                        latent_model_input,
                                        t,
                                        encoder_hidden_states=md_prompt_embeds_input,
                                        # cross_attention_kwargs=cross_attention_kwargs,
                                        added_cond_kwargs=md_added_cond_kwargs,
                                        return_dict=False,
                                    )[0]

                            if do_classifier_free_guidance:
                                noise_pred_uncond, noise_pred_text = noise_pred[::2], noise_pred[1::2]
                                noise_pred = noise_pred_uncond + multi_guidance_scale * (noise_pred_text - noise_pred_uncond)

                            if do_classifier_free_guidance and guidance_rescale > 0.0:
                                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                            # compute the previous noisy sample x_t -> x_t-1
                            self.scheduler._init_step_index(t)
                            latents_denoised_batch = self.scheduler.step(
                                noise_pred, t, latents_for_view, **extra_step_kwargs, return_dict=False)[0]

                            # extract value from batch
                            for latents_view_denoised, (h_start, h_end, w_start, w_end) in zip(
                                latents_denoised_batch.chunk(vb_size), batch_view
                            ):
                                value_local[:, :, h_start:h_end, w_start:w_end] += latents_view_denoised
                                count_local[:, :, h_start:h_end, w_start:w_end] += 1

                        if random_jitter:
                            value_local = value_local[: ,:, jitter_range: jitter_range + current_height // self.vae_scale_factor, jitter_range: jitter_range + current_width // self.vae_scale_factor]
                            count_local = count_local[: ,:, jitter_range: jitter_range + current_height // self.vae_scale_factor, jitter_range: jitter_range + current_width // self.vae_scale_factor]

                        if i != (len(timesteps) - 1):
                            noise_index = i + 1
                        else:
                            noise_index = i

                        value_local = torch.where(count_local == 0, noise_latents[noise_index], value_local)
                        count_local = torch.where(count_local == 0, torch.ones_like(count_local), count_local)
                        if use_dilated_sampling:
                            c2 = cosine_factor ** cosine_scale_2
                            value += value_local / count_local * (1 - c2)
                            count += torch.ones_like(value_local) * (1 - c2)
                        else:
                            value += value_local / count_local
                            count += torch.ones_like(value_local) 

                    if use_dilated_sampling:
                        ############################################# Dilated Sampling #############################################
                        views = [[h, w] for h in range(current_scale_num) for w in range(current_scale_num)]
                        views_batch = [views[i : i + view_batch_size] for i in range(0, len(views), view_batch_size)]
                        
                        h_pad = (current_scale_num - (latents.size(2) % current_scale_num)) % current_scale_num
                        w_pad = (current_scale_num - (latents.size(3) % current_scale_num)) % current_scale_num
                        latents_ = F.pad(latents, (w_pad, 0, h_pad, 0), 'constant', 0)
                        
                        count_global = torch.zeros_like(latents_)
                        value_global = torch.zeros_like(latents_)

                        if use_guassian:
                            c3 = 0.99 * cosine_factor ** cosine_scale_3 + 1e-2
                            std_, mean_ = latents_.std(), latents_.mean()
                            latents_gaussian = gaussian_filter(latents_, kernel_size=(2*current_scale_num-1), sigma=sigma*c3)
                            latents_gaussian = (latents_gaussian - latents_gaussian.mean()) / latents_gaussian.std() * std_ + mean_
                        else:
                            latents_gaussian = latents_

                        for j, batch_view in enumerate(views_batch):
                            
                            latents_for_view = torch.cat(
                                [
                                    latents_[:, :, h::current_scale_num, w::current_scale_num]
                                    for h, w in batch_view
                                ]
                            )

                            latents_for_view_gaussian = torch.cat(
                                [
                                    latents_gaussian[:, :, h::current_scale_num, w::current_scale_num]
                                    for h, w in batch_view
                                ]
                            )

                            if shuffle:
                                ######## window interaction ########
                                shape = latents_for_view.shape
                                shuffle_index = torch.stack([torch.randperm(shape[0]) for _ in range(latents_for_view.reshape(-1).shape[0]//shape[0])])

                                shuffle_index = shuffle_index.view(shape[1],shape[2],shape[3],shape[0])
                                original_index = torch.zeros_like(shuffle_index).scatter_(3, shuffle_index, torch.arange(shape[0]).repeat(shape[1], shape[2], shape[3], 1))

                                shuffle_index = shuffle_index.permute(3,0,1,2).to(device)
                                original_index = original_index.permute(3,0,1,2).to(device)
                                latents_for_view_gaussian = latents_for_view_gaussian.gather(0, shuffle_index)
                        
                            vb_size = latents_for_view.size(0)

                            # expand the latents if we are doing classifier free guidance
                            latent_model_input = latents_for_view_gaussian
                            latent_model_input = (
                                latent_model_input.repeat_interleave(2, dim=0)
                                if do_classifier_free_guidance
                                else latent_model_input
                            )
                            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                            prompt_embeds_input = torch.cat([prompt_embeds] * vb_size)
                            add_text_embeds_input = torch.cat([add_text_embeds] * vb_size)
                            add_time_ids_input = torch.cat([add_time_ids] * vb_size)

                            # predict the noise residual
                            added_cond_kwargs = {"text_embeds": add_text_embeds_input, "time_ids": add_time_ids_input}
                            noise_pred = self.unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=prompt_embeds_input,
                                # cross_attention_kwargs=cross_attention_kwargs,
                                added_cond_kwargs=added_cond_kwargs,
                                return_dict=False,
                            )[0]

                            if do_classifier_free_guidance:
                                noise_pred_uncond, noise_pred_text = noise_pred[::2], noise_pred[1::2]
                                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                            if do_classifier_free_guidance and guidance_rescale > 0.0:
                                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                            if shuffle:
                                ## recover
                                noise_pred = noise_pred.gather(0, original_index)

                            # compute the previous noisy sample x_t -> x_t-1
                            self.scheduler._init_step_index(t)
                            latents_denoised_batch = self.scheduler.step(noise_pred, t, latents_for_view, **extra_step_kwargs, return_dict=False)[0]

                            # extract value from batch
                            for latents_view_denoised, (h, w) in zip(
                                latents_denoised_batch.chunk(vb_size), batch_view
                            ):
                                value_global[:, :, h::current_scale_num, w::current_scale_num] += latents_view_denoised
                                count_global[:, :, h::current_scale_num, w::current_scale_num] += 1

                        value_global = value_global[: ,:, h_pad:, w_pad:]

                        if use_multidiffusion:
                            c2 = cosine_factor ** cosine_scale_2
                            value += value_global * c2
                            count += torch.ones_like(value_global) * c2
                        else:
                            value += value_global
                            count += torch.ones_like(value_global)
                                
                    latents = torch.where(count > 0, value / count, value)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)

    #########################################################################################################################################

                latents = (latents - latents.mean()) / latents.std() * anchor_std + anchor_mean
                if self.lowvram:
                    latents = latents.cpu()
                    torch.cuda.empty_cache()
                if not output_type == "latent":
                    # make sure the VAE is in float32 mode, as it overflows in float16
                    needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
                    if self.lowvram:
                        needs_upcasting = False # use madebyollin/sdxl-vae-fp16-fix in lowvram mode!
                        self.unet.cpu()
                        self.vae.to(device)
                    
                    if needs_upcasting:
                        self.upcast_vae()
                        latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

                    print("### Phase {} Decoding ###".format(current_scale_num))
                    if current_height > 2048 or current_width > 2048:
                        self.enable_vae_tiling()
                        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
                    else:
                        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

                    image = self.image_processor.postprocess(image, output_type=output_type)
                    image[0].save(f'{result_path}/AccDiffusion_{current_scale_num}.png')

                    if use_controlnet:
                        canny_image = np.array(image[0])
                        canny_image = auto_canny(canny_image, sigma=canny_sigma)
                        canny_image = canny_image[:, :, None]
                        canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
                        canny_image = Image.fromarray(canny_image)
                        canny_image.save(f'{result_path}/AccDiffusion_{current_scale_num}_cany.png')
                        condition_image = self.prepare_image(
                            image=canny_image,
                            width=current_width,
                            height=current_height,
                            batch_size=batch_size * num_images_per_prompt,
                            num_images_per_prompt=num_images_per_prompt,
                            device=device,
                            dtype=controlnet.dtype,
                            do_classifier_free_guidance=do_classifier_free_guidance,
                            guess_mode=False,
                        )

                    output_images.append(image[0])

                    # cast back to fp16 if needed
                    if needs_upcasting:
                        self.vae.to(dtype=torch.float16)
                else:
                    image = latents
        
        # Offload all models
        self.maybe_free_model_hooks()

        return output_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ### AccDiffusion v2 PARAMETERS ###
    parser.add_argument('--model_ckpt',default='stabilityai/stable-diffusion-xl-base-1.0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prompt', default="Astronaut on Mars During sunset.")
    parser.add_argument('--negative_prompt', default="blurry, ugly, duplicate, poorly drawn, deformed, mosaic")
    parser.add_argument('--cosine_scale_1', default=3, type=float, help="cosine scale 1")
    parser.add_argument('--cosine_scale_2', default=1, type=float, help="cosine scale 2")
    parser.add_argument('--cosine_scale_3', default=1, type=float, help="cosine scale 3")
    parser.add_argument('--sigma', default=0.8, type=float, help="sigma")
    parser.add_argument('--multi_decoder', default=True, type=bool, help="multi decoder or not")
    parser.add_argument('--num_inference_steps', default=50, type=int, help="num inference steps")
    parser.add_argument('--resolution', default='1024,1024', help="target resolution")
    parser.add_argument('--use_multidiffusion', default=False, action='store_true', help="use multidiffusion or not")
    parser.add_argument('--use_guassian', default=False, action='store_true', help="use guassian or not")
    parser.add_argument('--use_dilated_sampling', default=False, action='store_true', help="use dilated sampling or not")
    parser.add_argument('--use_progressive_upscaling', default=False, action='store_true', help="use progressive upscaling or not")
    parser.add_argument('--shuffle', default=False, action='store_true', help="shuffle or not")
    parser.add_argument('--use_skip_residual', default=False, action='store_true', help="use skip_residual or not")
    parser.add_argument('--save_attention_map', default=False, action='store_true', help="save attention map or not")
    parser.add_argument('--multi_guidance_scale', default=7.5, type=float, help="multi guidance scale")
    parser.add_argument('--upscale_mode', default="bicubic_latent", help="bicubic_image or bicubic_latent ")
    parser.add_argument('--use_md_prompt', default=False, action='store_true', help="use md prompt or not")
    parser.add_argument('--view_batch_size', default=16, type=int, help="view_batch_size")
    parser.add_argument('--stride', default=64, type=int, help="stride")
    parser.add_argument('--c', default=0.3, type=float, help="threshold")
    parser.add_argument('--canny_sigma', default=0.3, type=float, help="canny hypper parameter")
    parser.add_argument('--controlnet_conditioning_scale', default=0.3, type=float, help="controlnet conditioning scale")
    parser.add_argument('--control_guidance_end', default=1, type=float, help="control guidance end")
    parser.add_argument('--use_controlnet', default=False, action='store_true', help="use controlnet or not")
    ## others ##
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--experiment_name', default="AccDiffusion_v2")

    args = parser.parse_args()


    set_seed(args.seed)
    width,height = list(map(int, args.resolution.split(',')))
    controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = AccDiffusionv2SDXLPipeline.from_pretrained(args.model_ckpt, controlnet=controlnet, vae=vae, torch_dtype=torch.float16).to("cuda")
    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(args.seed)
    cross_attention_kwargs = {"edit_type": "visualize",
                              "n_self_replace": 0.4,
                              "n_cross_replace": {"default_": 1.0, "confetti": 0.8},
                              }
    
    if os.path.isfile(args.prompt):
        with open(args.prompt, "r") as file:
            prompts = file.read().strip()
            prompts =  prompts.split("\n")
    else:
        prompts = [args.prompt]

    seed = args.seed
    generator = generator.manual_seed(seed)

    for prompt in tqdm(prompts):
        print(f"Prompt: {prompt}")

        images = pipe(prompt, negative_prompt=args.negative_prompt, generator=generator,
                    width=width, height=height, view_batch_size=args.view_batch_size, stride=args.stride,
                    controlnet_conditioning_scale=args.controlnet_conditioning_scale, 
                    control_guidance_end=args.control_guidance_end, 
                    cross_attention_kwargs=cross_attention_kwargs,
                    use_controlnet=args.use_controlnet,
                    canny_sigma=args.canny_sigma,
                    num_inference_steps=args.num_inference_steps, 
                    guidance_scale = 7.5, multi_guidance_scale = args.multi_guidance_scale,
                    cosine_scale_1=args.cosine_scale_1, cosine_scale_2=args.cosine_scale_2, cosine_scale_3=args.cosine_scale_3,
                    sigma=args.sigma, use_guassian=args.use_guassian,
                    multi_decoder=args.multi_decoder, 
                    upscale_mode=args.upscale_mode, use_multidiffusion=args.use_multidiffusion, 
                    use_skip_residual=args.use_skip_residual, use_progressive_upscaling=args.use_progressive_upscaling,
                    use_dilated_sampling=args.use_dilated_sampling,
                    shuffle=args.shuffle, result_path=f"./output/{args.experiment_name}/{prompt[:200]}/{width}_{height}_{seed}/", 
                    debug=args.debug, save_attention_map=args.save_attention_map, use_md_prompt=args.use_md_prompt, c=args.c
                    )


