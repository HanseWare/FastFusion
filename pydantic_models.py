from typing import Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class LoRASetting(BaseModel):
    # Accept both "name" and the legacy "adapter_name" key from clients
    model_config = ConfigDict(populate_by_name=True)
    name: str = Field(validation_alias=AliasChoices("name", "adapter_name"))
    weight: Optional[float] = 1.0


class CreateImageRequest(BaseModel):
    # Unknown/extra OpenAI parameters are ignored instead of rejected
    model_config = ConfigDict(extra="ignore")
    prompt: str
    model: Optional[str] = None  # Ignored, single-model server
    n: Optional[int] = 1
    quality: Optional[str] = None
    response_format: Optional[str] = None  # url | b64_json, default url
    size: Optional[str] = None  # "WIDTHxHEIGHT" or "auto"
    style: Optional[str] = None  # Ignored
    user: Optional[str] = None  # Ignored
    background: Optional[str] = None  # Accepted, echoed in stream events
    moderation: Optional[str] = None  # Ignored
    output_format: Optional[str] = None  # png | webp | jpeg
    output_compression: Optional[int] = None  # 0-100 for webp/jpeg
    stream: Optional[bool] = False
    partial_images: Optional[int] = 0  # 0-3 partial images while streaming
    guidance_scale: Optional[float] = None  # Addon over openAI
    num_inference_steps: Optional[int] = None  # Addon over openAI
    lora_settings: Optional[list[LoRASetting]] = None  # Addon over openAI


class GenerationsConfig(BaseModel):
    enabled: bool = False
    pipeline: Optional[str] = "AutoPipelineForText2Image"


class EditsConfig(BaseModel):
    enabled: bool = False
    pipeline: Optional[str] = "AutoPipelineForInpainting"
    inpaint_pipeline: Optional[str] = None


class VariationsConfig(BaseModel):
    enabled: bool = False
    pipeline: Optional[str] = "AutoPipelineForImage2Image"
    enable_flux_redux: bool = False
    vision_model: str = "gpt-4o"
    vision_model_host: str = "https://api.openai.com"
    vision_model_api_key_variable: Optional[str] = ""
    vision_model_api_key: Optional[str] = ""


class LoRAConfig(BaseModel):
    type: str  # one of hub, local, or url
    address: str  # hub repo id, local path, or URL of the LoRA weights
    weight_name: Optional[str] = None  # optional weight file name inside a hub repo
    adapter_name: str  # name the adapter should have in the model


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
    generations_pipe_class: Optional[str] = None
    edits_pipe_class: Optional[str] = None
    variations_pipe_class: Optional[str] = None
