import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import threading
from typing import Dict, List

import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, AutoPipelineForInpainting, ConfigMixin
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
    pipe_config: FastFusionConfig | None
    base_pipe: ConfigMixin | None
    data_path: str
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipe_config = None
        self.base_pipe = None
        self.data_path = os.getenv("IMAGE_DATA_PATH", "/data")



def shortuuid():
    return uuid.uuid4().hex[:6]


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
    logger.info(f"Setting up model with device '{device}'...")
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
        logger.info(f"Loading model pipeline with init dtype {init_dtype}...")
        if config.pipeline.enable_images_generations:
            logger.info("Start loading base pipeline as image generation")
            pipe = AutoPipelineForText2Image.from_pretrained(config.pipeline.hf_model_id,
                                                                  torch_dtype=init_dtype)
            logger.info("Finished loading base pipeline as image generation")
        elif config.pipeline.enable_images_edits:
            logger.info("Start loading base pipeline as image edit")
            pipe = AutoPipelineForInpainting.from_pretrained(config.pipeline.hf_model_id,
                                                                  torch_dtype=init_dtype)
            logger.info("Finished loading base pipeline as image edit")
        elif config.pipeline.enable_images_variations:
            logger.info("Start loading base pipeline as image variation")
            pipe = AutoPipelineForImage2Image.from_pretrained(config.pipeline.hf_model_id,
                                                                   torch_dtype=init_dtype)
            logger.info("Finished loading base pipeline as image variation")
        else:
            raise ValueError(
                "No pipeline enabled. Please enable at least one of the following: images generation, image edits, image variations")

        # Apply settings before moving to GPU if necessary
        if config.pipeline.enable_cpu_offload:
            pipe.enable_sequential_cpu_offload()
            logger.info("Enabled CPU offload...")

        # Move the pipeline to GPU and convert to operation dtype
        logger.info("Moving pipeline to runtime dtype")
        logger.info("Pipeline runtime dtype:", pipe.dtype)
        pipe.to(get_torch_dtype(config.pipeline.torch_dtype_run))
        logger.info("Move to GPU")
        pipe.to("cuda")
        logger.info("Finished moving pipeline to GPU")
        # Apply settings that have to be applied after moving to GPU
        if config.pipeline.enable_vae_slicing:
            pipe.vae.enable_slicing()
            logger.info("Enabled VAE slicing...")
        if config.pipeline.enable_vae_tiling:
            pipe.vae.enable_tiling()
            logger.info("Enabled VAE tiling...")

        logger.info("Model setup complete with:")
        logger.info(f"Model: {config.pipeline.hf_model_id}")
        logger.info(f"Max value for n: {config.pipeline.max_n}")
        logger.info(f"CPU Offload Enabled: {config.pipeline.enable_cpu_offload}")
        logger.info(f"VAE Slicing Enabled: {config.pipeline.enable_vae_slicing}")
        logger.info(f"VAE Tiling Enabled: {config.pipeline.enable_vae_tiling}")
        logger.info(f"Images Generation Enabled: {config.pipeline.enable_images_generations}")
        logger.info(f"Image Edits Enabled: {config.pipeline.enable_images_edits}")
        logger.info(f"Image Variations Enabled: {config.pipeline.enable_images_variations}")
        return pipe, config, data_path
    except Exception as e:
        logging.error("Error during setup: %s", e)
        import traceback
        traceback.print_exc()
        raise e


# Run above setup() function in seperate thread on FastAPI app startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    pipe, config, data_path = setup("cuda")
    app.base_pipe = pipe
    app.pipe_config = config
    app.data_path = data_path
    yield
    # Clean up the ML models and release the resources
    app.base_pipe = None
    app.pipe_config = None

fastfusion_app = FastFusionApp(lifespan=lifespan)

@fastfusion_app.middleware("http")
async def ensure_model_ready(request: Request, call_next):
    if request.app.base_pipe is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    response = await call_next(request)
    return response

@fastfusion_app.post("/v1/images/generations")
async def generate_images(request: Request, body: CreateImageRequest):
    gen_pipe = AutoPipelineForText2Image.from_pipe(request.app.base_pipe)
    images_to_generate = min(body.n or 1, request.app.pipe_config.pipeline.max_n)
    prompt = body.prompt or 'A beautiful landscape'
    width, height = map(int, (body.size or '1024x1024').split('x'))
    quality = body.quality or 'standard'
    preset = request.app.pipe_config.generation_presets.get(quality)
    guidance_scale = preset.guidance_scale
    num_inference_steps = preset.num_inference_steps
    logger.info(f"Generating {images_to_generate} images with prompt '{prompt}'")
    images = gen_pipe(
        prompt=prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=images_to_generate
    ).images
    return encode_response(images, body.response_format or 'url', request)


@fastfusion_app.post("/v1/images/edits")
async def edit_images(request: Request, body: CreateImageEditRequest):
    edit_pipe = AutoPipelineForInpainting.from_pipe(request.app.base_pipe)
    images_to_generate = min(body.n or 1, request.app.pipe_config.pipeline.max_n)
    prompt = body.prompt or 'Edit the image to look more vibrant'
    init_image = convert_to_pil_image(body.image)
    mask_image = convert_to_pil_image(body.mask) if body.mask else None
    logger.info(f"Editing image with prompt '{prompt}'")
    images = edit_pipe(
        prompt=prompt,
        image=init_image,
        mask_image=mask_image,
        num_images_per_prompt=images_to_generate
    ).images
    return encode_response(images, body.response_format or 'url', request)



@fastfusion_app.post("/v1/images/variations")
async def generate_variations(request: Request, body: CreateImageVariationRequest):
    var_pipe = AutoPipelineForImage2Image.from_pipe(request.app.base_pipe)
    images_to_generate = min(body.n or 1, request.app.pipe_config.pipeline.max_n)
    prompt = body.prompt or 'Generate variations'
    init_image = convert_to_pil_image(body.image)
    images = var_pipe(
        prompt=prompt,
        image=init_image,
        num_images_per_prompt=images_to_generate
    ).images
    return encode_response(images, body.response_format or 'url', request)



def encode_response(images, response_format, request: Request) -> Response:
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
            file_path = os.path.join(request.app.data_path, file_id)
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
async def get_image_data(request: Request, file_id: str):
    file_path = os.path.join(request.app.data_path, file_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(content=open(file_path, "rb").read(), media_type="image/png")



if __name__ == "__main__":
    import uvicorn

    # get loglevel from env
    loglevel = os.getenv("FASTFUSION_LOGLEVEL", "info")
    logger.info("Setting log level from env to", loglevel)
    if loglevel == "debug":
        from transformers import logging as transformers_logging
        from diffusers.utils import logging as diffusers_logging

        transformers_logging.set_verbosity_debug()
        diffusers_logging.set_verbosity_debug()
        logging.basicConfig(level=logging.DEBUG)
    uvicorn.run(fastfusion_app, host="0.0.0.0", port=9999)
