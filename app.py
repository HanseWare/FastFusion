import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import threading
from typing import Dict, List

import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, AutoPipelineForInpainting, ConfigMixin
import base64
from io import BytesIO
import logging
import os
import json

from fastapi import FastAPI, HTTPException, Request, Response, Form, UploadFile, File
from PIL import Image
import openai
from pythonjsonlogger.json import JsonFormatter

from pydantic_models import *
__name__ = "hanseware.fastfusion"

RESERVED_ATTRS: List[str] = [
    "args",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
]

logger = logging.getLogger(__name__)

class FastFusionApp(FastAPI):
    pipe_config: FastFusionConfig | None
    base_pipe: ConfigMixin | None
    data_path: str
    base_url: str
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipe_config = None
        self.base_pipe = None
        self.data_path = os.getenv("IMAGE_DATA_PATH", "/data")
        self.base_url = os.getenv("BASE_URL")
        if self.base_url is None:
            raise ValueError("BASE_URL environment variable is required")



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

    image = Image.open(BytesIO(image_bytes))
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

        if (not config.pipeline.variations_config.vision_model_host) or (not config.pipeline.variations_config.vision_model):
            config.pipeline.variations_config.enable_images_variations = False
        elif config.pipeline.variations_config.enable_images_variations:
            if config.pipeline.variations_config.vision_model_api_key == "":
                config.pipeline.variations_config.vision_model_api_key = os.getenv(config.pipeline.variations_config.vision_model_api_key_variable, "ignored")

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
        elif config.pipeline.variations_config.enable_images_variations:
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
        logger.info(f"Pipeline runtime dtype: {config.pipeline.torch_dtype_run}")
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
        logger.info(f"Image Variations Enabled: {config.pipeline.variations_config.enable_images_variations}")
        return pipe, config, data_path
    except Exception as e:
        logging.error("Error during setup: %s", e)
        import traceback
        traceback.print_exc()
        raise e


def setup_logging():
    loglevel = os.getenv("FASTFUSION_LOGLEVEL", "INFO").upper()
    if loglevel == "DEBUG":
        from transformers import logging as transformers_logging
        from diffusers.utils import logging as diffusers_logging
        transformers_logging.set_verbosity_debug()
        diffusers_logging.set_verbosity_debug()
    logger.info("Setting log level from env to", loglevel)
    logging.basicConfig(level=logging.getLevelName(loglevel))
    logHandler = logging.StreamHandler()
    formatter = JsonFormatter(timestamp=True, reserved_attrs=RESERVED_ATTRS, datefmt='%Y-%m-%d %H:%M:%S')
    logHandler.setFormatter(formatter)
    logging.getLogger().handlers.clear()
    logging.getLogger().addHandler(logHandler)
    uvi_logger = logging.getLogger("uvicorn.access")
    uvi_logger.handlers.clear()
    uvi_logger.addHandler(logHandler)


# Run above setup() function in seperate thread on FastAPI app startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    setup_logging()
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
    if not request.app.pipe_config.pipeline.enable_images_generations:
        raise HTTPException(status_code=404, detail="Image generation not enabled")
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
async def edit_images(
    request: Request,
    prompt: str = Form(...),
    image: UploadFile = File(...),
    mask: UploadFile = File(None),
    n: int = Form(1),
    response_format: str = Form('url'),
    model: Optional[str] = Form("flux.1-dev"),
    size: Optional[str] = Form("1024x1024"),
    user: Optional[str] = Form(None),  # Ignored
    strength: Optional[float] = Form(None),  # Addon over openAI
    guidance_scale: Optional[float] = Form(None),  # Addon over openAI
    num_inference_steps: Optional[int] = Form(None) # Addon over openAI
):
    if not request.app.pipe_config.pipeline.enable_images_edits:
        raise HTTPException(status_code=404, detail="Image edits")
    try:
        if strength is None:
            strength = 1.0
        if guidance_scale is None:
            guidance_scale = request.app.pipe_config.pipeline.global_guidance_scale
        if num_inference_steps is None:
            num_inference_steps = request.app.pipe_config.pipeline.global_num_inference_steps
        edit_pipe = AutoPipelineForInpainting.from_pipe(request.app.base_pipe)
        images_to_generate = min(n, request.app.pipe_config.pipeline.max_n)

        # Read the uploaded files
        image_data = await image.read()
        mask_data = await mask.read() if mask is not None else None

        # Convert bytes to PIL Images
        init_image = Image.open(BytesIO(image_data)).convert("RGB")
        mask_image = Image.open(BytesIO(mask_data)).convert("RGB") if mask_data else None
        width, height = map(int, (size or '1024x1024').split('x'))

        images = edit_pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            num_images_per_prompt=images_to_generate,
            guidance_scale=guidance_scale,
            strength=strength,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height
        ).images
        return encode_response(images, response_format, request)
    except Exception as e:
        logging.error(f"Error during image editing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def get_image_description(image: Image.Image, request: Request) -> str:
    """
    Sends the image to OpenAI's GPT-4 model to get a description.
    """
    try:
        openai_client = openai.Client(api_key=request.app.pipe_config.pipeline.vision_model_api_key, base_url=request.app.pipe_config.pipeline.vision_model_host)
        # Convert the image to bytes
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()

        import base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        # Include the base64 image in the prompt
        # Alternatively, if you have access to GPT-4V, you can send the image directly.

        # For GPT-4V (assuming you have access and the API supports image inputs)
        response = openai_client.chat.completions.create(
            model=request.app.pipe_config.variations_config.vision_model,
            messages=[
                {
                    "role": "system", "content": "You are a helpful assistant to describe images."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Guess the prompt that generated this image! One should be able to recreate it from the description",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            },
                        },
                    ],
                }
            ],
            temperature=0.2,
            max_tokens=200
        )

        # Extract the description from the response
        description = response.choices[0].message.content.strip()
        openai_client.close()
        return description

    except Exception as e:
        logging.error(f"Error getting image description: {e}")
        # Return a default prompt if unable to get description
        return 'Create variations of the image at your best guess!'


@fastfusion_app.post("/v1/images/variations")
async def generate_variations(
    request: Request,
    image: UploadFile = File(...),
    n: int = Form(1),
    response_format: str = Form('url'),
    model: str = Form(...),
    size: Optional[str] = Form("1024x1024"),
    user: Optional[str] = Form(None),  # Ignored
    strength: Optional[float] = Form(None),  # Add-on over OpenAI
    guidance_scale: Optional[float] = Form(None),  # Add-on over OpenAI
    num_inference_steps: Optional[int] = Form(None)  # Add-on over OpenAI
):
    if not request.app.pipe_config.pipeline.variations_config.enable_images_variations:
        raise HTTPException(status_code=404, detail="Image variations not enabled")
    try:
        if strength is None:
            strength = 1.0
        if guidance_scale is None:
            guidance_scale = request.app.pipe_config.pipeline.global_guidance_scale
        if num_inference_steps is None:
            num_inference_steps = request.app.pipe_config.pipeline.global_num_inference_steps
        var_pipe = AutoPipelineForImage2Image.from_pipe(request.app.base_pipe)
        images_to_generate = min(n, request.app.pipe_config.pipeline.max_n)
        image_data = await image.read()
        init_image = Image.open(BytesIO(image_data)).convert("RGB")
        width, height = map(int, (size or '1024x1024').split('x'))
        prompt = await get_image_description(init_image, request)

        images = var_pipe(
            prompt=prompt,
            image=init_image,
            num_images_per_prompt=images_to_generate,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height
        ).images
        return encode_response(images, response_format, request)
    except Exception as e:
        logging.error(f"Error during image variation generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))



def encode_response(images, response_format, request: Request) -> Response:
    logger.debug("Encoding image response")
    image_responses = []
    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        logger.debug("Output: %s", image)
        if response_format == "b64_json":
            image_responses.append({"b64_json": img_str})
        elif response_format == "url":
            # Save image and return URL
            img_data = base64.b64decode(img_str)
            img = Image.open(BytesIO(img_data))
            file_id = f"{shortuuid()}.png"
            file_path = os.path.join(request.app.data_path, file_id)
            img.save(file_path, format="PNG")
            image_responses.append({"url": f"{fastfusion_app.base_url}/v1/images/data/{file_id}"})
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


@fastfusion_app.get("/health")
async def health():
    return {"status": "ok"}


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
    host = os.getenv("FASTFUSION_HOST", "0.0.0.0")
    port = int(os.getenv("FASTFUSION_PORT", 9999))
    uvicorn.run(fastfusion_app, host=host, port=port)
