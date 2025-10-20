import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import threading
from pathlib import Path
from typing import Dict, List

import requests
import torch
import diffusers
from diffusers import *
import base64
from io import BytesIO
import logging
import os
import json

import queue
from threading import Thread
import asyncio

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
    generations_pipe: ConfigMixin | None
    edits_pipe: ConfigMixin | None
    inpaint_edits_pipe: ConfigMixin | None
    variations_pipe: ConfigMixin | None
    redux_pipe: ConfigMixin | None
    base_pipe: ConfigMixin | None
    data_path: str
    base_url: str
    depth_processor: ConfigMixin | None
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

task_queue = queue.Queue()
worker_thread = Thread(target=lambda: None)
worker_active = False

def worker():
    global worker_active
    while True:
        try:
            # Retrieve a task and execute it
            task = task_queue.get(timeout=3)  # Timeout to check periodically if still active
            task()
            task_queue.task_done()
        except queue.Empty:
            # If no task, potentially stop the worker
            if task_queue.empty():
                worker_active = False
                break


def ensure_worker_running():
    global worker_thread, worker_active
    if not worker_active or not worker_thread.is_alive():
        worker_active = True
        worker_thread = Thread(target=worker, daemon=True)
        worker_thread.start()


def setup():
    # Set up data path
    data_path = os.getenv("IMAGE_DATA_PATH", "/data")
    os.makedirs(data_path, exist_ok=True)
    # Launch cleanup thread
    cleanup_thread = threading.Thread(target=clean_old_images, args=(data_path,), daemon=True)
    cleanup_thread.start()
    try:
        # Load configuration JSON
        config_path = os.getenv("CONFIG_PATH", "model_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as config_file:
                config_data = json.load(config_file)
                config = FastFusionConfig(**config_data)
        else:
            raise ValueError("Configuration file not found")

        # Set up the model pipeline
        device = config.pipeline.torch_device
        logger.info(f"Setting up model with device or device map: '{device}'...")

        # if (not config.pipeline.variations_config.enable_flux_redux) or (not config.pipeline.variations_config.vision_model_host) or (not config.pipeline.variations_config.vision_model):
        #     config.pipeline.variations_config.enabled = False
        # el
        if config.pipeline.variations_config.enabled:
            if config.pipeline.variations_config.vision_model_api_key == "":
                config.pipeline.variations_config.vision_model_api_key = os.getenv(config.pipeline.variations_config.vision_model_api_key_variable, "ignored")

        # Initialisieren Sie Container für die Pipelines
        initialized_pipelines = {
            "generation": None,
            "edit": None,
            "edit_inpaint": None,
            "variation": None,
            "redux": None
        }

        init_dtype = get_torch_dtype(
            config.pipeline.torch_dtype_init if config.pipeline.torch_device != "balanced" else config.pipeline.torch_dtype_run)
        run_dtype = get_torch_dtype(config.pipeline.torch_dtype_run)
        device_map = config.pipeline.torch_device

        # Laden Sie die Basis-Pipeline, die für alle anderen verwendet wird
        base_pipe = AutoPipelineForText2Image.from_pretrained(config.pipeline.hf_model_id, torch_dtype=init_dtype,
                                                 device_map=device_map)
        logger.info(f"Setting up model with base pipeline: '{base_pipe.__class__.__name__}'...")

        # Laden Sie jede benötigte Pipeline explizit
        if config.pipeline.generations_config.enabled:
            logger.info("Loading image generation pipeline...")
            if not config.pipeline.generations_config.pipeline is None:
                initialized_pipelines["generation"] = (getattr(diffusers, config.pipeline.generations_config.pipeline)).from_pipe(base_pipe, torch_dtype=run_dtype)
            else:
                initialized_pipelines["generation"] = AutoPipelineForText2Image.from_pipe(base_pipe, torch_dtype=run_dtype)
            logger.info("Finished loading image generation pipeline.")

        if config.pipeline.edits_config.enabled:
            logger.info("Loading image edit (inpainting) pipeline...")
            if not config.pipeline.edits_config.pipeline is None:
                initialized_pipelines["edit"] = (getattr(diffusers, config.pipeline.edits_config.pipeline)).from_pipe(base_pipe, torch_dtype=run_dtype)
            else:
                initialized_pipelines["edit"] = AutoPipelineForInpainting.from_pipe(base_pipe, torch_dtype=run_dtype)
            logger.info("Finished loading image edit pipeline.")
            if not config.pipeline.edits_config.inpaint_pipeline is None:
                initialized_pipelines["edit_inpaint"] = (getattr(diffusers, config.pipeline.edits_config.pipeline)).from_pipe(base_pipe, torch_dtype=run_dtype)
                logger.info("Finished loading image inpaint edit pipeline.")


        if config.pipeline.variations_config.enabled:
            logger.info("Loading image variation (image-to-image) pipeline...")
            if config.pipeline.variations_config.enable_flux_redux:
                # FLUX erfordert eine spezielle Handhabung
                initialized_pipelines["variation"] = base_pipe  # Die Basis-Flux-Pipeline wird direkt verwendet
                repo_redux = "black-forest-labs/FLUX.1-Redux-dev"
                initialized_pipelines["redux"] = FluxPriorReduxPipeline.from_pretrained(repo_redux, torch_dtype=run_dtype,
                                                                                        device_map=device_map)
            elif not config.pipeline.variations_config.pipeline is None:
                initialized_pipelines["variation"] = (getattr(diffusers, config.pipeline.variations_config.pipeline)).from_pipe(base_pipe, torch_dtype=run_dtype)
            else:
                initialized_pipelines["variation"] = AutoPipelineForImage2Image.from_pipe(base_pipe, torch_dtype=run_dtype)
            logger.info("Finished loading image variation pipeline.")

        # Löschen Sie die Basis-Pipeline, wenn sie nicht mehr benötigt wird, um Speicher freizugeben
        del base_pipe
        torch.cuda.empty_cache()

        # Laden Sie LoRA-Gewichte für jede geladene Pipeline
        if config.pipeline.loras:
            # Wenden Sie die Gewichte auf jede aktive Pipeline an
            for pipe in initialized_pipelines.values():
                if pipe is not None:
                    for lora_config in config.pipeline.loras:
                        # Stellen Sie sicher, dass die Gewichte bereits heruntergeladen wurden
                        weights_path = lora_config.address
                        if lora_config.type == "url":
                            weights_path = str(Path(
                                "weights") / f"{lora_config.weight_name or lora_config.adapter_name}.safetensors")

                        pipe.load_lora_weights(weights_path, adapter_name=lora_config.adapter_name)
                    pipe.disable_lora()
                    logger.info(f"Loaded LoRA weights for pipeline: {pipe.__class__.__name__}")

        # Wenden Sie weitere Einstellungen an
        for pipe in initialized_pipelines.values():
            if pipe is not None:
                if config.pipeline.enable_vae_slicing:
                    pipe.vae.enable_slicing()
                if config.pipeline.enable_vae_tiling:
                    pipe.vae.enable_tiling()

        logger.info("Model setup complete with:")
        logger.info(f"Model: {config.pipeline.hf_model_id}")
        logger.info(f"Max value for n: {config.pipeline.max_n}")
        logger.info(f"CPU Offload Enabled: {config.pipeline.enable_cpu_offload}")
        logger.info(f"VAE Slicing Enabled: {config.pipeline.enable_vae_slicing}")
        logger.info(f"VAE Tiling Enabled: {config.pipeline.enable_vae_tiling}")
        logger.info(f"Images Generation Enabled: {config.pipeline.generations_config.enabled}")
        logger.info(f"Image Edits Enabled: {config.pipeline.edits_config.enabled}")
        logger.info(f"Image Variations Enabled: {config.pipeline.variations_config.enabled}")
        return initialized_pipelines, config, data_path
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
    pipelines, config, data_path = setup()

    # Speichern Sie jede Pipeline in der App
    app.generations_pipe = pipelines.get("generation")
    app.edits_pipe = pipelines.get("edit")
    app.inpaint_edits_pipe = pipelines.get("edit_inpaint")
    app.variations_pipe = pipelines.get("variation")
    app.redux_pipe = pipelines.get("redux")

    app.pipe_config = config
    app.data_path = data_path

    app.base_pipe = app.generations_pipe or app.edits_pipe or app.variations_pipe

    yield
    # Clean up the ML models and release the resources
    app.generations_pipe = None
    app.edits_pipe = None
    app.variations_pipe = None
    app.redux_pipe = None
    app.pipe_config = None
    app.base_pipe = None
    torch.cuda.empty_cache()

fastfusion_app = FastFusionApp(lifespan=lifespan)

@fastfusion_app.middleware("http")
async def ensure_model_ready(request: Request, call_next):

    if request.app.base_pipe is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    response = await call_next(request)
    return response


@fastfusion_app.post("/v1/images/generations")
async def generate_images(request: Request, body: CreateImageRequest):
    event = asyncio.Event()
    result_container = {}

    def task():
        try:
            if not request.app.pipe_config.pipeline.generations_config.enabled:
                raise HTTPException(status_code=404, detail="Image generation not enabled")

            # Initialize the generation pipeline from the base pipeline
            gen_pipe = request.app.generations_pipe
            if gen_pipe is None:
                raise HTTPException(status_code=503, detail="Image generation pipeline not loaded")

            # Apply LoRA settings if provided
            if body.lora_settings:
                adapter_names = []
                adapter_weights = []
                for lora_setting in body.lora_settings:
                    adapter_names.append(lora_setting.name)
                    adapter_weights.append(lora_setting.weight)
                gen_pipe.set_adapters(adapter_names, adapter_weights)

            # Generate images based on the provided prompt and settings
            images_to_generate = min(body.n or 1, request.app.pipe_config.pipeline.max_n)
            prompt = body.prompt or 'A beautiful landscape'
            width, height = map(int, (body.size or '1024x1024').split('x'))
            if not body.guidance_scale is None and not body.num_inference_steps is None:
                guidance_scale = body.guidance_scale
                num_inference_steps = body.num_inference_steps
            elif not body.quality is None:
                quality = body.quality or 'standard'
                preset = request.app.pipe_config.generation_presets.get(quality)
                guidance_scale = preset.guidance_scale
                num_inference_steps = preset.num_inference_steps
            else:
                guidance_scale = request.app.pipe_config.pipeline.global_guidance_scale
                num_inference_steps = request.app.pipe_config.pipeline.global_num_inference_steps

            images = gen_pipe(
                prompt=prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=images_to_generate
            ).images

            gen_pipe.disable_lora()  # Disable LoRA after generation to reset the state

            result_container['result'] = encode_response(images, body.response_format or 'url', request)

        except Exception as e:
            result_container['error'] = e
        finally:
            event.set()

    # Enqueue the task and ensure the worker is running
    task_queue.put(task)
    ensure_worker_running()
    await event.wait()

    if 'error' in result_container:
        raise HTTPException(status_code=500, detail=str(result_container['error']))

    return result_container['result']


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
        num_inference_steps: Optional[int] = Form(None),  # Addon over openAI
        lora_settings: Optional[List[LoRASetting]] = None
        # Include this if you're using Pydantic models to parse LoRA settings
):
    event = asyncio.Event()
    result_container = {}

    def task():
        try:
            if not request.app.pipe_config.pipeline.edits_config.enabled:
                raise HTTPException(status_code=404, detail="Image edits")

            # Initialize the edit pipeline from the base pipeline
            if mask is not None and request.app.inpaint_edits_pipe is not None:
                edit_pipe = request.app.inpaint_edits_pipe
            else:
                edit_pipe = request.app.edits_pipe
            if edit_pipe is None:
                raise HTTPException(status_code=503, detail="Image edit pipeline not loaded")

            # Apply LoRA settings if provided
            if lora_settings:
                adapter_names = []
                adapter_weights = []
                for lora_setting in lora_settings:
                    adapter_names.append(lora_setting.name)
                    adapter_weights.append(lora_setting.weight)
                edit_pipe.set_adapters(adapter_names, adapter_weights)

            images_to_generate = min(n, request.app.pipe_config.pipeline.max_n)

            # Synchronously read and convert the uploaded images
            image_data = image.file.read()  # Use .file.read() in a synchronous context
            mask_data = mask.file.read() if mask is not None else None

            # Convert bytes to PIL Images
            init_image = Image.open(BytesIO(image_data)).convert("RGB")
            mask_image = Image.open(BytesIO(mask_data)).convert("RGB") if mask_data else None
            width, height = map(int, (size or '1024x1024').split('x'))

            logger.debug("Editing images with parameters: ")
            logger.debug(f"Prompt: {prompt}, Strength: {strength}, Guidance Scale: {guidance_scale}, Steps: {num_inference_steps}, Images to Generate: {images_to_generate}, Size: {width}x{height}")
            if mask_data is None:
                images = edit_pipe(
                    prompt=prompt,
                    image=init_image,
                    num_images_per_prompt=images_to_generate,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    width=width,
                    height=height
                ).images
            else:
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

            # Disable LoRA after editing to reset the model state
            edit_pipe.disable_lora()

            result_container['result'] = encode_response(images, response_format, request)

        except Exception as e:
            result_container['error'] = e
        finally:
            event.set()

    # Enqueue the task and ensure the worker is running
    task_queue.put(task)
    ensure_worker_running()
    await event.wait()

    if 'error' in result_container:
        logger.info(f"Image edits failed with error: {result_container['error']}")
        raise HTTPException(status_code=500, detail=str(result_container['error']))

    return result_container['result']


def get_image_description(image: Image.Image, request: Request) -> str:
    """
    Sends the image to Vision model to get a description.
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
        num_inference_steps: Optional[int] = Form(None),  # Add-on over OpenAI
        lora_settings: Optional[List[LoRASetting]] = None  # New parameter for LoRA settings
):
    event = asyncio.Event()
    result_container = {}

    def task():
        try:
            if not request.app.pipe_config.pipeline.variations_config.enabled:
                raise HTTPException(status_code=404, detail="Image variations not enabled")

            local_strength = strength if strength is not None else 1.0
            local_guidance = guidance_scale if guidance_scale is not None else request.app.pipe_config.pipeline.global_guidance_scale
            local_steps = num_inference_steps if num_inference_steps is not None else request.app.pipe_config.pipeline.global_num_inference_steps
            images_to_generate = min(n, request.app.pipe_config.pipeline.max_n)

            image_data = image.file.read()  # within the worker thread use synchronous I/O
            init_image = Image.open(BytesIO(image_data)).convert("RGB")
            width, height = map(int, (size or '1024x1024').split('x'))

            if request.app.redux_pipe is not None and request.app.pipe_config.pipeline.variations_config.enable_flux_redux:
                var_pipe = request.app.base_pipe

                # Apply LoRA settings if provided
                if lora_settings:
                    adapter_names = [ls.adapter_name for ls in lora_settings]
                    adapter_weights = [ls.weight for ls in lora_settings]
                    var_pipe.set_adapters(adapter_names, adapter_weights)

                # Use the redux pipe to produce extra kwargs from the image
                extra_kwargs = request.app.redux_pipe(init_image)
                # Use the prompt from the request as-is (even if None)
                images = var_pipe(
                    prompt=None,
                    num_images_per_prompt=images_to_generate,
                    guidance_scale=local_guidance,
                    num_inference_steps=local_steps,
                    **extra_kwargs
                ).images

            else:
                var_pipe = request.app.variations_pipe
                if var_pipe is None:
                    raise HTTPException(status_code=503, detail="Image variation pipeline not loaded")

                local_prompt = get_image_description(init_image, request)

                if lora_settings:
                    adapter_names = [ls.adapter_name for ls in lora_settings]
                    adapter_weights = [ls.weight for ls in lora_settings]
                    var_pipe.set_adapters(adapter_names, adapter_weights)

                logger.info("Generating variations with parameters: ")
                logger.info(f"Prompt: {local_prompt}, Strength: {local_strength}, Guidance Scale: {local_guidance}, Steps: {local_steps}, Images to Generate: {images_to_generate}, Size: {width}x{height}")
                images = var_pipe(
                    prompt=local_prompt,
                    image=init_image,
                    num_images_per_prompt=images_to_generate,
                    guidance_scale=local_guidance,
                    num_inference_steps=local_steps,
                    width=width,
                    height=height
                ).images


            var_pipe.disable_lora()
            result_container['result'] = encode_response(images, response_format, request)
        except Exception as e:
            logging.error(f"Error during image variation generation: {e}")
            result_container['error'] = e
        finally:
            event.set()

    # Enqueue the task and ensure the worker thread is running
    task_queue.put(task)
    ensure_worker_running()
    await event.wait()

    if 'error' in result_container:
        raise HTTPException(status_code=500, detail=str(result_container['error']))

    return result_container['result']


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
