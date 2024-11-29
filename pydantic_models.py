from typing import Optional

from pydantic import BaseModel


class CreateImageRequest(BaseModel):
    prompt: str
    model: Optional[str] = "flux.1-dev"
    n: Optional[int] = 1
    quality: Optional[str] = "standard"
    response_format: Optional[str] = "url"
    size: Optional[str] = "1024x1024"
    style: Optional[str] = "vivid"
    user: Optional[str] = None  # Ignored


class CreateImageEditRequest(BaseModel):
    prompt: str
    image: str  # Assuming binary as a string
    mask: Optional[str] = None
    model: Optional[str] = "flux.1-dev"
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "url"
    user: Optional[str] = None  # Ignored
    guidance_scale: Optional[float] = 7.0  # Addon over openAI
    num_inference_steps: Optional[int] = 50  # Addon over openAI


class CreateImageVariationRequest(BaseModel):
    image: str  # Assuming binary as a string
    model: Optional[str] = "flux.1-dev"
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "url"
    user: Optional[str] = None  # Ignored
    prompt: str = None  # Requirement over openAI
    num_inference_steps: Optional[int] = 50  # Addon over openAI
    strength: Optional[float] = 0.75  # Addon over openAI
    guidance_scale: Optional[float] = 0.0  # Addon over openAI

class VariationsConfig(BaseModel):
    enable_images_variations: bool
    use_flux_redux: bool = False
    vision_model: str = "gpt-4o"
    vision_model_host: str = "https://api.openai.com"
    vision_model_api_key_variable: Optional[str] = ""
    vision_model_api_key: Optional[str] = ""

class PipelineConfig(BaseModel):
    hf_model_id: str
    max_n: int
    torch_dtype_init: str
    torch_dtype_run: str
    torch_device: str = "cuda"
    enable_cpu_offload: bool
    global_guidance_scale: float
    global_num_inference_steps: int
    enable_images_generations: bool
    enable_images_edits: bool
    variations_config: VariationsConfig
    enable_vae_slicing: bool
    enable_vae_tiling: bool


class GenerationPreset(BaseModel):
    guidance_scale: float
    num_inference_steps: int


class FastFusionConfig(BaseModel):
    name: str
    pipeline: PipelineConfig
    generation_presets: dict[str, GenerationPreset]
