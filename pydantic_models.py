from typing import Optional

from pydantic import BaseModel


class LoRASetting(BaseModel):
    name: str
    weight: Optional[float] = 1.0

class CreateImageRequest(BaseModel):
    prompt: str
    model: Optional[str] = "flux.1-dev"
    n: Optional[int] = 1
    quality: Optional[str] = None
    guidance_scale: Optional[float] = None
    num_inference_steps: Optional[int] = None
    response_format: Optional[str] = "url"
    size: Optional[str] = "1024x1024"
    style: Optional[str] = "vivid"
    user: Optional[str] = None  # Ignored
    lora_settings: Optional[list[LoRASetting]] = None


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
    prompt: Optional[str] = None  # Requirement over openAI
    num_inference_steps: Optional[int] = 50  # Addon over openAI
    strength: Optional[float] = 0.75  # Addon over openAI
    guidance_scale: Optional[float] = 0.0  # Addon over openAI

class GenerationsConfig(BaseModel):
    enabled: bool = False
    pipeline: Optional[str] = "AutoPipelineForText2Image"

class EditsConfig(BaseModel):
    enabled: bool = False
    pipeline: Optional[str] = "AutoPipelineForInpainting"

class VariationsConfig(BaseModel):
    enabled: bool = False
    pipeline: Optional[str] = "AutoPipelineForImage2Image"
    enable_flux_redux: bool = False
    vision_model: str = "gpt-4o"
    vision_model_host: str = "https://api.openai.com"
    vision_model_api_key_variable: Optional[str] = ""
    vision_model_api_key: Optional[str] = ""

class LoRAConfig(BaseModel):
    type: str # one of hub, local, or url
    address: str # address of the LoRA model, either a hub model name, a local path, or a URL
    weight_name: Optional[str] = None # name of the weight file to load
    adapter_name: str # name the adapter should have in the model

class PipelineConfig(BaseModel):
    hf_model_id: str
    max_n: int
    torch_dtype_init: str
    torch_dtype_run: str
    torch_device: str = "cuda"
    enable_cpu_offload: bool
    global_guidance_scale: float
    global_num_inference_steps: int
    generations_config: GenerationsConfig
    edits_config: EditsConfig
    variations_config: VariationsConfig
    enable_vae_slicing: bool
    enable_vae_tiling: bool
    loras: Optional[list[LoRAConfig]] = None


class GenerationPreset(BaseModel):
    guidance_scale: float
    num_inference_steps: int


class FastFusionConfig(BaseModel):
    name: str
    pipeline: PipelineConfig
    generation_presets: dict[str, GenerationPreset]
