import os
import json
from pydantic import BaseModel
from PIL import Image
from litserve import LitServer, LitAPI
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, AutoPipelineForInpainting
import torch

from openai_image_spec import OpenAIImageSpec

def get_torch_dtype(dtype_name):
    return getattr(torch, dtype_name, None)

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

    def setup(self, device):
        # Load configuration JSON
        config_path = os.getenv("CONFIG_PATH", "model_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as config_file:
                config_data = json.load(config_file)
                self.config = LitFusionConfig(**config_data)
        else:
            raise ValueError("Configuration file not found")

        # Load the model pipeline using AutoPipelineForText2Image
        print("Loading model pipeline...")
        init_dtype = get_torch_dtype(self.config.pipeline.torch_dtype_init)
        if self.config.pipeline.enable_images_generations:
            self.base_pipe = AutoPipelineForText2Image.from_pretrained(self.config.pipeline.hf_model_id,
                                                                       torch_dtype=init_dtype)
        elif self.config.pipeline.enable_images_edits:
            self.base_pipe = AutoPipelineForInpainting.from_pretrained(self.config.pipeline.hf_model_id,
                                                                       torch_dtype=init_dtype)
        elif self.config.pipeline.enable_images_variations:
            self.base_pipe = AutoPipelineForImage2Image.from_pretrained(self.config.pipeline.hf_model_id,
                                                                        torch_dtype=init_dtype)
        else:
            raise ValueError(
                "No pipeline enabled. Please enable at least one of the following: images generation, image edits, image variations")

        # Apply settings before moving to GPU if necessary
        if self.config.pipeline.enable_cpu_offload:
            self.base_pipe.enable_sequential_cpu_offload()
        if self.config.pipeline.enable_vae_slicing:
            self.base_pipe.vae.enable_slicing()
        if self.config.pipeline.enable_vae_tiling:
            self.base_pipe.vae.enable_tiling()

        # Move the pipeline to GPU and convert to operation dtype
        self.base_pipe.to(get_torch_dtype(self.config.pipeline.torch_dtype_run)).to("cuda")

        print("Model setup complete with:")
        print(f"Model: {self.config.pipeline.hf_model_id}")
        print(f"Max value for n: {self.config.pipeline.max_n}")
        print(f"CPU Offload Enabled: {self.config.pipeline.enable_cpu_offload}")
        print(f"VAE Slicing Enabled: {self.config.pipeline.enable_vae_slicing}")
        print(f"VAE Tiling Enabled: {self.config.pipeline.enable_vae_tiling}")
        print(f"Images Generation Enabled: {self.config.pipeline.enable_images_generation}")
        print(f"Image Edits Enabled: {self.config.pipeline.enable_images_edits}")
        print(f"Image Variations Enabled: {self.config.pipeline.enable_images_variations}")

    def predict(self, request):
        # Logic to determine which type of request it is
        request_type = request.get('request_type')
        if request_type == "generation" and self.config.pipeline.enable_images_generations:
            yield self.generate_images(request)
        elif request_type == "edit" and self.config.pipeline.enable_images_edits:
            yield self.edit_images(request)
        elif request_type == "variation" and self.config.pipeline.enable_images_variations:
            yield self.generate_variations(request)
        else:
            yield "Unknown or disabled request type"

    def generate_images(self, request):
        gen_pipe = AutoPipelineForText2Image.from_pipe(self.base_pipe)
        images_to_generate = min(request.get('n', 1), self.config.pipeline.max_n)
        prompt = request.get('prompt', 'A beautiful landscape')
        width, height = request.get('size', '1024x1024').split('x')
        guidance_scale, num_inference_steps = self.config.generation_presets.get(request.get('quality', 'standard'))
        images = gen_pipe(prompt=prompt, width=width, height=height, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, num_images_per_prompt=images_to_generate).images
        for img in images:
            yield img

    def edit_images(self, request):
        edit_pipe = AutoPipelineForInpainting.from_pipe(self.base_pipe)
        for _ in range(min(request.get('n', 1), self.config.pipeline.max_n)):
            prompt = request.get('prompt', 'Edit the image to look more vibrant')
            init_image = request.get('image')
            mask_image = request.get('mask')
            images = edit_pipe(prompt=prompt, image=init_image, mask_image=mask_image).images
            for img in images:
                yield img

    def generate_variations(self, request):
        var_pipe = AutoPipelineForImage2Image.from_pipe(self.base_pipe)
        for _ in range(min(request.get('n', 1), self.config.pipeline.max_n)):
            prompt = request.get('prompt', 'Generate variations')
            init_image = request.get('image')
            images = var_pipe(prompt=prompt, image=init_image).images
            for img in images:
                yield img

if __name__ == "__main__":
    api = LitFusion()
    server = LitServer(api, spec=OpenAIImageSpec())
    server.run(port=8000)
