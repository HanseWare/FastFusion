import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import threading
from typing import Dict, List

import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, AutoPipelineForInpainting
import base64
import io
import logging
import os
import json

from fastapi import FastAPI, HTTPException, Request, Response
from PIL import Image

from pydantic_models import *

logger = logging.getLogger(__name__)


class FastFusionApp(FastAPI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipe_config = None
        self.base_pipe = None
        self.data_path = os.getenv("IMAGE_DATA_PATH", "/data")


fastfusion_app = FastFusionApp()


def shortuuid():
    return uuid.uuid4().hex[:6]


@fastfusion_app.middleware("http")
async def ensure_model_ready(request: Request, call_next):
    if not hasattr(fastfusion_app, "base_pipe"):
        raise HTTPException(status_code=503, detail="Model not ready")
    response = await call_next(request)
    return response


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


def clean_old_images(data_path: str):
    """
    Cleans up image files in the data directory that are older than 1 hour.
    """
    while True:
        now = datetime.now()
        for filename in os.listdir(data_path):
            filepath = os.path.join(data_path, filename)
            if os.path.isfile(filepath):
                file_creation_time = datetime.fromtimestamp(os.path.getctime(filepath))
                if now - file_creation_time > timedelta(hours=1):
                    os.remove(filepath)
                    logger.info(f"Deleted old image file: {filepath}")
        time.sleep(60)


def setup(device):
    # Set up data path
    data_path = os.getenv("IMAGE_DATA_PATH", "/data")
    os.makedirs(data_path, exist_ok=True)
    # Launch cleanup thread
    cleanup_thread = threading.Thread(target=clean_old_images, args=(data_path,), daemon=True)
    cleanup_thread.start()
    print(f"Setting up model with device '{device}'...")
    try:
        # Load configuration JSON
        config_path = os.getenv("CONFIG_PATH", "model_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as config_file:
                config_data = json.load(config_file)
                config = FastFusionConfig(**config_data)
        else:
            raise ValueError("Configuration file not found")

        # Load the model pipeline using AutoPipelineForText2Image
        init_dtype = get_torch_dtype(config.pipeline.torch_dtype_init)
        print(f"Loading model pipeline with init dtype {init_dtype}...")
        if config.pipeline.enable_images_generations:
            print("Start loading base pipeline as image generation")
            base_pipe = AutoPipelineForText2Image.from_pretrained(config.pipeline.hf_model_id,
                                                                  torch_dtype=init_dtype)
            print("Finished loading base pipeline as image generation")
        elif config.pipeline.enable_images_edits:
            print("Start loading base pipeline as image edit")
            base_pipe = AutoPipelineForInpainting.from_pretrained(config.pipeline.hf_model_id,
                                                                  torch_dtype=init_dtype)
            print("Finished loading base pipeline as image edit")
        elif config.pipeline.enable_images_variations:
            print("Start loading base pipeline as image variation")
            base_pipe = AutoPipelineForImage2Image.from_pretrained(config.pipeline.hf_model_id,
                                                                   torch_dtype=init_dtype)
            print("Finished loading base pipeline as image variation")
        else:
            raise ValueError(
                "No pipeline enabled. Please enable at least one of the following: images generation, image edits, image variations")

        # Apply settings before moving to GPU if necessary
        if config.pipeline.enable_cpu_offload:
            base_pipe.enable_sequential_cpu_offload()
            print("Enabled CPU offload...")

        # Move the pipeline to GPU and convert to operation dtype
        print("Moving pipeline to runtime dtype")
        print("Pipeline runtime dtype:", base_pipe.dtype)
        base_pipe.to(get_torch_dtype(config.pipeline.torch_dtype_run))
        print("Move to GPU")
        base_pipe.to("cuda")
        print("Finished moving pipeline to GPU")
        # Apply settings that have to be applied after moving to GPU
        if config.pipeline.enable_vae_slicing:
            base_pipe.vae.enable_slicing()
            print("Enabled VAE slicing...")
        if config.pipeline.enable_vae_tiling:
            base_pipe.vae.enable_tiling()
            print("Enabled VAE tiling...")

        fastfusion_app.pipe_config = config
        fastfusion_app.base_pipe = base_pipe
        print("Model setup complete with:")
        print(f"Model: {config.pipeline.hf_model_id}")
        print(f"Max value for n: {config.pipeline.max_n}")
        print(f"CPU Offload Enabled: {config.pipeline.enable_cpu_offload}")
        print(f"VAE Slicing Enabled: {config.pipeline.enable_vae_slicing}")
        print(f"VAE Tiling Enabled: {config.pipeline.enable_vae_tiling}")
        print(f"Images Generation Enabled: {config.pipeline.enable_images_generations}")
        print(f"Image Edits Enabled: {config.pipeline.enable_images_edits}")
        print(f"Image Variations Enabled: {config.pipeline.enable_images_variations}")
    except Exception as e:
        logging.error("Error during setup: %s", e)
        import traceback
        traceback.print_exc()
        raise e


# Run above setup() function in seperate thread on FastAPI app startup
@asynccontextmanager
async def lifespan(app: FastFusionApp):
    # Load the ML model
    setup("cuda")
    yield
    # Clean up the ML models and release the resources
    app.base_pipe = None
    app.pipe_config = None


@fastfusion_app.post("/v1/images/generations")
async def generate_images(self, request: CreateImageRequest):
    gen_pipe = AutoPipelineForText2Image.from_pipe(fastfusion_app.base_pipe)
    images_to_generate = min(request.get('n', 1), fastfusion_app.pipe_config.pipeline.max_n)
    prompt = request.get('prompt', 'A beautiful landscape')
    width, height = map(int, request.get('size', '1024x1024').split('x'))
    quality = request.get('quality', 'standard')
    preset = fastfusion_app.pipe_config.generation_presets.get(quality)
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
    return self.encode_response(images, request.get('response_format', 'url'))


@fastfusion_app.post("/v1/images/edits")
async def edit_images(self, request: CreateImageEditRequest):
    edit_pipe = AutoPipelineForInpainting.from_pipe(fastfusion_app.base_pipe)
    images_to_generate = min(request.get('n', 1), fastfusion_app.pipe_config.pipeline.max_n)
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
    return self.encode_response(images, request.get('response_format', 'url'))


@fastfusion_app.post("/v1/images/variations")
async def generate_variations(self, request: CreateImageVariationRequest):
    var_pipe = AutoPipelineForImage2Image.from_pipe(fastfusion_app.base_pipe)
    images_to_generate = min(request.get('n', 1), fastfusion_app.pipe_config.pipeline.max_n)
    prompt = request.get('prompt', 'Generate variations')
    init_image_data = request.get('image')
    init_image = convert_to_pil_image(init_image_data)
    images = var_pipe(
        prompt=prompt,
        image=init_image,
        num_images_per_prompt=images_to_generate
    ).images
    return self.encode_response(images, request.get('response_format', 'url'))


def encode_response(self, images, response_format) -> Response:
    logger.debug("Encoding image response")
    image_responses = []
    for image in images:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        logger.debug("Output: %s", image)
        if response_format == "b64_json":
            image_responses.append({"b64_json": img_str})
        elif response_format == "url":
            # Save image and return URL
            img_data = base64.b64decode(img_str)
            img = Image.open(io.BytesIO(img_data))
            file_id = f"{shortuuid()}.png"
            file_path = os.path.join(self.data_path, file_id)
            img.save(file_path, format="PNG")
            image_responses.append({"url": f"/v1/images/data/{file_id}"})
        else:
            raise HTTPException(status_code=500, detail="Unexpected output format")

    final_response = {
        "created": int(time.time()),
        "data": image_responses
    }
    return Response(content=json.dumps(final_response), media_type="application/json")


@fastfusion_app.get("/v1/images/data/{file_id}")
async def get_image_data(self, file_id: str):
    file_path = os.path.join(self.data_path, file_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(content=open(file_path, "rb").read(), media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    # get loglevel from env
    loglevel = os.getenv("FASTFUSION_LOGLEVEL", "info")
    print("Setting log level from env to", loglevel)
    if loglevel == "debug":
        from transformers import logging as transformers_logging
        from diffusers.utils import logging as diffusers_logging

        transformers_logging.set_verbosity_debug()
        diffusers_logging.set_verbosity_debug()
        logging.basicConfig(level=logging.DEBUG)
    uvicorn.run(fastfusion_app, host="0.0.0.0", port=9999)
