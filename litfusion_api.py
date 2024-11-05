import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, AutoPipelineForInpainting
import base64
import io
import logging
import os
import json
from pydantic import BaseModel
from PIL import Image
from litserve import LitServer, LitAPI

from openai_image_spec import OpenAIImageSpec

from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast

def get_torch_dtype(dtype_name):
    return getattr(torch, dtype_name, None)

def convert_to_pil_image(image_data: str) -> Image.Image:
    if image_data.startswith("data:image"):
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
    else:
        image_bytes = image_data.encode('latin1')

    image = Image.open(io.BytesIO(image_bytes))
    return image

class PipelineConfig(BaseModel):
    hf_model_id: str
    max_n: int
    torch_dtype_init: str
    torch_dtype_run: str
    enable_cpu_offload: bool
    enable_images_generations: bool
    enable_images_edits: bool
    enable_images_variations: bool
    enable_vae_slicing: bool
    enable_vae_tiling: bool


class GenerationPreset(BaseModel):
    guidance_scale: float
    num_inference_steps: int


class LitFusionConfig(BaseModel):
    name: str
    pipeline: PipelineConfig
    generation_presets: dict[str, GenerationPreset]


class LitFusion(LitAPI):
    def __init__(self):
        self.config = None
        self.base_pipe = None

    def initialize_model(self, device):
        print(f"Setting up model with device '{device}'...")
        try:
            # Load configuration JSON
            config_path = os.getenv("CONFIG_PATH", "model_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as config_file:
                    config_data = json.load(config_file)
                    self.config = LitFusionConfig(**config_data)
            else:
                raise ValueError("Configuration file not found")

            # Load the model pipeline using AutoPipelineForText2Image
            init_dtype = get_torch_dtype(self.config.pipeline.torch_dtype_init)
            print(f"Loading model pipeline with init dtype {init_dtype}...")
            if self.config.pipeline.enable_images_generations:
                print("Start loading base pipeline as image generation")
                self.base_pipe = AutoPipelineForText2Image.from_pretrained(self.config.pipeline.hf_model_id,
                                                                           torch_dtype=init_dtype)
                print("Finished loading base pipeline as image generation")
            elif self.config.pipeline.enable_images_edits:
                print("Start loading base pipeline as image edit")
                self.base_pipe = AutoPipelineForInpainting.from_pretrained(self.config.pipeline.hf_model_id,
                                                                           torch_dtype=init_dtype)
                print("Finished loading base pipeline as image edit")
            elif self.config.pipeline.enable_images_variations:
                print("Start loading base pipeline as image variation")
                self.base_pipe = AutoPipelineForImage2Image.from_pretrained(self.config.pipeline.hf_model_id,
                                                                            torch_dtype=init_dtype)
                print("Finished loading base pipeline as image variation")
            else:
                raise ValueError(
                    "No pipeline enabled. Please enable at least one of the following: images generation, image edits, image variations")

            # Apply settings before moving to GPU if necessary
            if self.config.pipeline.enable_cpu_offload:
                self.base_pipe.enable_sequential_cpu_offload()
                print("Enabled CPU offload...")

            # Move the pipeline to GPU and convert to operation dtype
            print("Moving pipeline to runtime dtype")
            print("Pipeline runtime dtype:", self.base_pipe.dtype)
            self.base_pipe.to(get_torch_dtype(self.config.pipeline.torch_dtype_run))
            print("Move to GPU")
            self.base_pipe.to("cuda")
            print("Finished moving pipeline to GPU")
            # Apply settings that have to be applied after moving to GPU
            if self.config.pipeline.enable_vae_slicing:
                self.base_pipe.vae.enable_slicing()
                print("Enabled VAE slicing...")
            if self.config.pipeline.enable_vae_tiling:
                self.base_pipe.vae.enable_tiling()
                print("Enabled VAE tiling...")

            print("Model setup complete with:")
            print(f"Model: {self.config.pipeline.hf_model_id}")
            print(f"Max value for n: {self.config.pipeline.max_n}")
            print(f"CPU Offload Enabled: {self.config.pipeline.enable_cpu_offload}")
            print(f"VAE Slicing Enabled: {self.config.pipeline.enable_vae_slicing}")
            print(f"VAE Tiling Enabled: {self.config.pipeline.enable_vae_tiling}")
            print(f"Images Generation Enabled: {self.config.pipeline.enable_images_generations}")
            print(f"Image Edits Enabled: {self.config.pipeline.enable_images_edits}")
            print(f"Image Variations Enabled: {self.config.pipeline.enable_images_variations}")
        except Exception as e:
            logging.error("Error during setup: %s", e)
            import traceback
            traceback.print_exc()
            raise e

    def setup(self, device):
        # Initialize the model
        print("Initializing model...")
        config_path = os.getenv("CONFIG_PATH", "model_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as config_file:
                config_data = json.load(config_file)
                self.config = LitFusionConfig(**config_data)
        else:
            raise ValueError("Configuration file not found")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained("ChuckMcSneed/FLUX.1-dev",
                                                                    subfolder="scheduler")
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.bfloat16)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.bfloat16)
        text_encoder_2 = T5EncoderModel.from_pretrained("ChuckMcSneed/FLUX.1-dev", subfolder="text_encoder_2",
                                                        torch_dtype=torch.bfloat16)
        tokenizer_2 = T5TokenizerFast.from_pretrained("ChuckMcSneed/FLUX.1-dev", subfolder="tokenizer_2",
                                                      torch_dtype=torch.bfloat16)
        vae = AutoencoderKL.from_pretrained("ChuckMcSneed/FLUX.1-dev", subfolder="vae",
                                            torch_dtype=torch.bfloat16)
        transformer = FluxTransformer2DModel.from_pretrained("ChuckMcSneed/FLUX.1-dev",
                                                             subfolder="transformer", torch_dtype=torch.bfloat16)
        self.base_pipe = FluxPipeline(
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=None,
            tokenizer_2=tokenizer_2,
            vae=vae,
            transformer=None,
        )
        self.base_pipe.text_encoder_2 = text_encoder_2
        self.base_pipe.transformer = transformer

        # Apply settings before moving to GPU if necessary
        if self.config.pipeline.enable_cpu_offload:
            self.base_pipe.enable_sequential_cpu_offload()
            print("Enabled CPU offload...")

        # Move the pipeline to GPU and convert to operation dtype
        print("Moving pipeline to runtime dtype")
        print("Pipeline runtime dtype:", self.base_pipe.dtype)
        self.base_pipe.to(get_torch_dtype(self.config.pipeline.torch_dtype_run))
        print("Move to GPU")
        self.base_pipe.to("cuda")
        print("Finished moving pipeline to GPU")
        # Apply settings that have to be applied after moving to GPU
        if self.config.pipeline.enable_vae_slicing:
            self.base_pipe.vae.enable_slicing()
            print("Enabled VAE slicing...")
        if self.config.pipeline.enable_vae_tiling:
            self.base_pipe.vae.enable_tiling()
            print("Enabled VAE tiling...")

        print("Model setup complete with:")
        print(f"Model: {self.config.pipeline.hf_model_id}")
        print(f"Max value for n: {self.config.pipeline.max_n}")
        print(f"CPU Offload Enabled: {self.config.pipeline.enable_cpu_offload}")
        print(f"VAE Slicing Enabled: {self.config.pipeline.enable_vae_slicing}")
        print(f"VAE Tiling Enabled: {self.config.pipeline.enable_vae_tiling}")
        print(f"Images Generation Enabled: {self.config.pipeline.enable_images_generations}")
        print(f"Image Edits Enabled: {self.config.pipeline.enable_images_edits}")
        print(f"Image Variations Enabled: {self.config.pipeline.enable_images_variations}")
        print("Model initialization running as background task")

    def predict(self, request):
        # Logic to determine which type of request it is
        request_type = request.get('request_type')
        if request_type == "generation" and self.config.pipeline.enable_images_generations:
            print("Predict called with generation request")
            yield from self.generate_images(request)
        elif request_type == "edit" and self.config.pipeline.enable_images_edits:
            print("Predict called with edit request")
            yield from self.edit_images(request)
        elif request_type == "variation" and self.config.pipeline.enable_images_variations:
            print("Predict called with variation request")
            yield from self.generate_variations(request)
        else:
            yield "Unknown or disabled request type"

    def generate_images(self, request):
        gen_pipe = AutoPipelineForText2Image.from_pipe(self.base_pipe)
        images_to_generate = min(request.get('n', 1), self.config.pipeline.max_n)
        prompt = request.get('prompt', 'A beautiful landscape')
        width, height = map(int, request.get('size', '1024x1024').split('x'))
        quality = request.get('quality', 'standard')
        preset = self.config.generation_presets.get(quality)
        guidance_scale = preset.guidance_scale
        num_inference_steps = preset.num_inference_steps
        print(f"Generating {images_to_generate} images with prompt '{prompt}'")
        images = gen_pipe(
            prompt=prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=images_to_generate
        ).images
        for img in images:
            # Serialize image to base64 string
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            # Yield a dictionary containing the serialized image
            yield {
                "image": img_str,
                "response_format": request.get('response_format', 'url')
            }

    def edit_images(self, request):
        edit_pipe = AutoPipelineForInpainting.from_pipe(self.base_pipe)
        images_to_generate = min(request.get('n', 1), self.config.pipeline.max_n)
        prompt = request.get('prompt', 'Edit the image to look more vibrant')
        init_image_data = request.get('image')
        init_image = convert_to_pil_image(init_image_data)
        mask_image_data = request.get('mask')
        mask_image = convert_to_pil_image(mask_image_data) if mask_image_data else None
        images = edit_pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            num_images_per_prompt=images_to_generate
        ).images
        for img in images:
            # Serialize image to base64 string
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            # Yield a dictionary containing the serialized image
            yield {
                "image": img_str,
                "response_format": request.get('response_format', 'url')
            }

    def generate_variations(self, request):
        var_pipe = AutoPipelineForImage2Image.from_pipe(self.base_pipe)
        images_to_generate = min(request.get('n', 1), self.config.pipeline.max_n)
        prompt = request.get('prompt', 'Generate variations')
        init_image_data = request.get('image')
        init_image = convert_to_pil_image(init_image_data)
        images = var_pipe(
            prompt=prompt,
            image=init_image,
            num_images_per_prompt=images_to_generate
        ).images
        for img in images:
            # Serialize image to base64 string
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            # Yield a dictionary containing the serialized image
            yield {
                "image": img_str,
                "response_format": request.get('response_format', 'url')
            }


if __name__ == "__main__":
    api = LitFusion()
    # get loglevel from env
    loglevel = os.getenv("LITFUSION_LOGLEVEL", "info")
    print("Setting log level from env to", loglevel)
    if loglevel == "debug":
        from transformers import logging as transformers_logging
        from diffusers.utils import logging as diffusers_logging
        transformers_logging.set_verbosity_debug()
        diffusers_logging.set_verbosity_debug()
        logging.basicConfig(level=logging.DEBUG)
    server = LitServer(api, spec=OpenAIImageSpec(), accelerator="cuda")
    server.run(port=8000, log_level=loglevel)
