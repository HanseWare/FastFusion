import asyncio
import base64
import inspect
import json
import os
import queue
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Dict, List, Optional

import diffusers
import openai
import requests
import torch
from PIL import Image
from diffusers import *
import logging  # re-import: the diffusers star import shadows stdlib logging
from fastapi import FastAPI, HTTPException, Request, Form, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pythonjsonlogger.json import JsonFormatter

from pydantic_models import *

LOGGER_NAME = "hanseware.fastfusion"

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

logger = logging.getLogger(LOGGER_NAME)

# Generous default: big presets (100 steps, n=2) can take many minutes on large models
REQUEST_TIMEOUT_SECONDS = float(os.getenv("FASTFUSION_REQUEST_TIMEOUT", "1800"))

# output_format -> (PIL format, media type)
IMAGE_FORMATS = {
    "png": ("PNG", "image/png"),
    "webp": ("WEBP", "image/webp"),
    "jpeg": ("JPEG", "image/jpeg"),
    "jpg": ("JPEG", "image/jpeg"),
}

FILE_ID_RE = re.compile(r"^[0-9a-f]{6}\.(png|webp|jpeg|jpg)$")

# We do not track tokens; emit a schema-complete usage object in stream events
ZERO_USAGE = {
    "total_tokens": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "input_tokens_details": {"text_tokens": 0, "image_tokens": 0},
}


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
        self.generations_pipe = None
        self.edits_pipe = None
        self.inpaint_edits_pipe = None
        self.variations_pipe = None
        self.redux_pipe = None
        self.base_pipe = None
        self.depth_processor = None
        self.data_path = os.getenv("IMAGE_DATA_PATH", "/data")
        self.base_url = os.getenv("BASE_URL")
        if self.base_url is None:
            raise ValueError("BASE_URL environment variable is required")


def shortuuid():
    return uuid.uuid4().hex[:6]


def get_torch_dtype(dtype_name):
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unknown torch dtype '{dtype_name}' in config")
    return dtype


def clean_old_images(data_path: str):
    """
    Cleans up image files in the data directory that are older than 1 hour.
    """
    while True:
        try:
            now = datetime.now()
            for filename in os.listdir(data_path):
                filepath = os.path.join(data_path, filename)
                try:
                    if os.path.isfile(filepath):
                        file_creation_time = datetime.fromtimestamp(os.path.getctime(filepath))
                        if now - file_creation_time > timedelta(hours=1):
                            os.remove(filepath)
                            logger.info(f"Deleted old image file: {filepath}")
                except OSError:
                    logger.warning("Could not clean up file %s", filepath, exc_info=True)
        except Exception:
            logger.exception("Image cleanup iteration failed")
        time.sleep(60)


# Single permanent worker thread serializes all GPU work. The pipelines share
# their model components (from_pipe), so adapter state is global mutable state
# and tasks must never run concurrently.
task_queue: "queue.Queue[Callable[[], None]]" = queue.Queue()


def worker():
    while True:
        task = task_queue.get()
        try:
            task()
        except Exception:
            logger.exception("Unhandled error in worker task")
        finally:
            task_queue.task_done()


# ---------------------------------------------------------------------------
# Generic pipeline helpers (model agnostic — no Flux-specific assumptions)
# ---------------------------------------------------------------------------

def filter_call_kwargs(pipe, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop None values and kwargs the pipeline's __call__ does not accept."""
    params = inspect.signature(pipe.__call__).parameters
    has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    filtered, dropped = {}, []
    for key, value in kwargs.items():
        if value is None:
            continue
        if has_var_kw or key in params:
            filtered[key] = value
        else:
            dropped.append(key)
    if dropped:
        logger.debug("Dropped kwargs %s not supported by %s", dropped, type(pipe).__name__)
    return filtered


def pipeline_accepts(pipe, param: str) -> bool:
    return param in inspect.signature(pipe.__call__).parameters


def snap_to_multiple(value: int, multiple: int = 16) -> int:
    return max(multiple, round(value / multiple) * multiple)


def parse_size(size: Optional[str], default=(1024, 1024)):
    """Parse OpenAI-style size. Accepts "WIDTHxHEIGHT" and "auto"."""
    if not size or size.lower() == "auto":
        width, height = default
    else:
        try:
            width, height = (int(v) for v in size.lower().split("x"))
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail=f"Invalid size '{size}', expected 'WIDTHxHEIGHT' or 'auto'")
    return snap_to_multiple(width), snap_to_multiple(height)


def resolve_generation_params(config: FastFusionConfig, quality: Optional[str],
                              guidance_scale: Optional[float], num_inference_steps: Optional[int]):
    """
    Resolve guidance/steps: explicit values win individually, then the quality
    preset, then the global defaults. Unknown quality values (including
    OpenAI's "auto") fall back to global defaults instead of erroring.
    """
    preset = None
    if quality and quality != "auto":
        preset = config.generation_presets.get(quality)
        if preset is None:
            logger.warning("Unknown quality preset '%s'; using global defaults", quality)
    guidance = guidance_scale if guidance_scale is not None else (
        preset.guidance_scale if preset else config.pipeline.global_guidance_scale)
    steps = num_inference_steps if num_inference_steps is not None else (
        preset.num_inference_steps if preset else config.pipeline.global_num_inference_steps)
    return guidance, steps


def normalize_output_format(output_format: Optional[str]) -> str:
    fmt = (output_format or "png").lower()
    if fmt not in IMAGE_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported output_format '{output_format}'")
    return fmt


def parse_lora_settings(raw: Optional[str]) -> Optional[List[LoRASetting]]:
    """Parse the lora_settings multipart form field (JSON string)."""
    if not raw:
        return None
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            data = [data]
        return [LoRASetting(**item) for item in data]
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid lora_settings: {e}")


def get_available_adapters(pipe) -> set:
    try:
        return {name for adapters in pipe.get_list_adapters().values() for name in adapters}
    except Exception:
        return set()


def validate_lora_settings(pipe, lora_settings: Optional[List[LoRASetting]]):
    """Fail fast with a 400 (instead of a late 500) on unknown adapter names."""
    if not lora_settings:
        return
    available = get_available_adapters(pipe)
    missing = {ls.name for ls in lora_settings} - available
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown LoRA adapter(s): {sorted(missing)}. Available: {sorted(available)}")


def apply_lora(pipe, lora_settings: Optional[List[LoRASetting]]) -> bool:
    if not lora_settings:
        return False
    names = [ls.name for ls in lora_settings]
    weights = [ls.weight if ls.weight is not None else 1.0 for ls in lora_settings]
    pipe.set_adapters(names, weights)
    pipe.enable_lora()
    logger.debug("Enabled LoRA adapters %s with weights %s", names, weights)
    return True


def clear_lora(pipe):
    """Disable LoRA; never raise — a raise here previously hung requests forever."""
    try:
        pipe.disable_lora()
        logger.debug("Disabled LoRA adapters")
    except Exception:
        logger.warning("Failed to disable LoRA adapters", exc_info=True)


def prepare_mask_image(mask_bytes: bytes) -> Image.Image:
    """
    OpenAI semantics: transparent areas of the mask mark the region to edit.
    Diffusers semantics: white pixels are repainted. If the uploaded mask has a
    real alpha channel we invert it; otherwise we keep the image as-is so
    existing white-means-edit grayscale masks keep working.
    """
    mask = Image.open(BytesIO(mask_bytes))
    if "A" in mask.getbands():
        alpha = mask.getchannel("A")
        lo, _hi = alpha.getextrema()
        if lo < 255:  # mask actually uses transparency
            return Image.eval(alpha, lambda a: 255 - a).convert("RGB")
    return mask.convert("RGB")


def _save_kwargs_for(pil_format: str, output_compression: Optional[int]) -> Dict[str, Any]:
    if pil_format in ("JPEG", "WEBP") and output_compression is not None:
        return {"quality": max(0, min(100, int(output_compression)))}
    return {}


def image_to_b64(image: Image.Image, output_format: str = "png",
                 output_compression: Optional[int] = None) -> str:
    pil_format, _ = IMAGE_FORMATS[output_format]
    if pil_format == "JPEG" and image.mode != "RGB":
        image = image.convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format=pil_format, **_save_kwargs_for(pil_format, output_compression))
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def encode_response(images, response_format, request: Request,
                    output_format: str = "png", output_compression: Optional[int] = None):
    logger.debug("Encoding image response")
    pil_format, _ = IMAGE_FORMATS[output_format]
    image_responses = []
    for image in images:
        if response_format == "b64_json":
            image_responses.append({"b64_json": image_to_b64(image, output_format, output_compression)})
        elif response_format == "url":
            file_id = f"{shortuuid()}.{output_format}"
            file_path = os.path.join(request.app.data_path, file_id)
            to_save = image.convert("RGB") if pil_format == "JPEG" and image.mode != "RGB" else image
            to_save.save(file_path, format=pil_format, **_save_kwargs_for(pil_format, output_compression))
            image_responses.append({"url": f"{request.app.base_url}/v1/images/data/{file_id}"})
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported response_format '{response_format}'")
    return {"created": int(time.time()), "data": image_responses}


# ---------------------------------------------------------------------------
# Streaming (OpenAI Images API SSE: *.partial_image / *.completed events)
# ---------------------------------------------------------------------------

def sse_format(payload: dict) -> str:
    return f"event: {payload['type']}\ndata: {json.dumps(payload)}\n\n"


def build_event_meta(width: int, height: int, quality: Optional[str],
                     background: Optional[str], output_format: str) -> Dict[str, Any]:
    return {
        "size": f"{width}x{height}",
        # Keep values inside the OpenAI enum so strict clients don't choke
        "quality": quality if quality in ("low", "medium", "high", "auto") else "high",
        "background": background if background in ("transparent", "opaque", "auto") else "opaque",
        "output_format": "jpeg" if output_format == "jpg" else output_format,
    }


def decode_preview(pipe, latents, width: int, height: int) -> Image.Image:
    """
    Best-effort decode of intermediate latents into a preview image using the
    pipeline's own VAE. Model agnostic: handles packed latents (Flux-style
    _unpack_latents) and the common scaling/shift and mean/std VAE configs.
    Raises on anything unexpected — the caller disables previews then.
    """
    if latents is None:
        raise ValueError("Pipeline callback provided no latents")
    vae = getattr(pipe, "vae", None)
    image_processor = getattr(pipe, "image_processor", None)
    if vae is None or image_processor is None:
        raise ValueError("Pipeline has no vae/image_processor to decode previews with")
    latents = latents.detach()
    if hasattr(pipe, "_unpack_latents"):
        latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents = latents[:1].to(dtype=vae.dtype)  # preview only the first image of the batch
    cfg = vae.config
    scaling = getattr(cfg, "scaling_factor", None) or 1.0
    shift = getattr(cfg, "shift_factor", None) or 0.0
    latents_mean = getattr(cfg, "latents_mean", None)
    latents_std = getattr(cfg, "latents_std", None)
    if latents_mean is not None and latents_std is not None:
        mean = torch.tensor(latents_mean).view(1, -1, 1, 1).to(latents.device, latents.dtype)
        std = torch.tensor(latents_std).view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents = latents * std / scaling + mean
    else:
        latents = latents / scaling + shift
    with torch.no_grad():
        image = vae.decode(latents, return_dict=False)[0]
    return image_processor.postprocess(image, output_type="pil")[0]


def build_stream_callback(pipe, emit, partial_images: int, num_steps: int,
                          width: int, height: int, event_prefix: str,
                          meta: Dict[str, Any], output_format: str,
                          output_compression: Optional[int], cancel: Dict[str, Any]):
    """
    Returns a callback_on_step_end for the pipeline that emits partial_image
    events at evenly spaced steps, or None if previews are off/unsupported.
    Also propagates client-side cancellation into the pipeline via _interrupt.
    """
    partial_images = max(0, min(3, partial_images or 0))
    supports_callback = pipeline_accepts(pipe, "callback_on_step_end")
    if partial_images > 0 and not supports_callback:
        logger.warning("Pipeline %s does not support step callbacks; no partial images will be streamed",
                       type(pipe).__name__)
    if not supports_callback:
        return None

    # Evenly spaced preview steps, excluding the final step (that's the completed event)
    marks = set()
    if partial_images > 0 and num_steps > 1:
        for i in range(partial_images):
            marks.add(min(num_steps - 2, max(0, round(num_steps * (i + 1) / (partial_images + 1)) - 1)))
    state = {"index": 0, "failed": False}

    def callback(p, step, timestep, callback_kwargs):
        if cancel.get("requested"):
            try:
                p._interrupt = True  # honored by pipelines with interrupt support
            except Exception:
                pass
        if not state["failed"] and step in marks:
            try:
                preview = decode_preview(p, callback_kwargs.get("latents"), width, height)
                emit({
                    "type": f"{event_prefix}.partial_image",
                    "b64_json": image_to_b64(preview, output_format, output_compression),
                    "created_at": int(time.time()),
                    "partial_image_index": state["index"],
                    **meta,
                })
                state["index"] += 1
            except Exception:
                state["failed"] = True
                logger.warning("Preview decode failed; disabling partial images for this request",
                               exc_info=True)
        return callback_kwargs

    return callback


async def stream_pipeline_task(event_prefix: str, run: Callable, meta: Dict[str, Any],
                               output_format: str, output_compression: Optional[int],
                               cancel: Dict[str, Any]):
    """
    Enqueue `run` on the worker and return an SSE StreamingResponse.
    `run(emit, cancel)` must return the list of final PIL images.
    """
    loop = asyncio.get_running_loop()
    events: asyncio.Queue = asyncio.Queue()

    def emit(payload: Optional[dict]):
        loop.call_soon_threadsafe(events.put_nowait, payload)

    def task():
        try:
            images = run(emit, cancel)
            for image in images:
                emit({
                    "type": f"{event_prefix}.completed",
                    "b64_json": image_to_b64(image, output_format, output_compression),
                    "created_at": int(time.time()),
                    "usage": ZERO_USAGE,
                    **meta,
                })
        except HTTPException as e:
            emit({"type": "error", "error": {"message": str(e.detail), "type": "server_error", "code": e.status_code}})
        except Exception as e:
            logger.exception("Streaming image task failed")
            emit({"type": "error", "error": {"message": str(e), "type": "server_error"}})
        finally:
            emit(None)  # sentinel

    task_queue.put(task)

    async def event_source():
        try:
            while True:
                payload = await events.get()
                if payload is None:
                    break
                yield sse_format(payload)
        finally:
            # Client disconnected (or stream finished): stop wasting GPU time
            cancel["requested"] = True
            pipe = cancel.get("pipe")
            if pipe is not None:
                try:
                    pipe._interrupt = True
                except Exception:
                    pass

    return StreamingResponse(event_source(), media_type="text/event-stream")


async def run_pipeline_task(run: Callable, request: Request, response_format: str,
                            output_format: str, output_compression: Optional[int],
                            cancel: Dict[str, Any]):
    """Enqueue `run` on the worker, wait for it, and encode the JSON response."""
    loop = asyncio.get_running_loop()
    done = asyncio.Event()
    result: Dict[str, Any] = {}

    def emit(_payload):  # non-streaming: partial events are discarded
        pass

    def task():
        try:
            images = run(emit, cancel)
            result["result"] = encode_response(images, response_format, request,
                                               output_format, output_compression)
        except Exception as e:
            result["error"] = e
        finally:
            # asyncio.Event is not thread-safe; must set it on the loop
            loop.call_soon_threadsafe(done.set)

    task_queue.put(task)
    try:
        await asyncio.wait_for(done.wait(), timeout=REQUEST_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        cancel["requested"] = True
        raise HTTPException(status_code=504, detail="Image request timed out")

    error = result.get("error")
    if error is not None:
        if isinstance(error, HTTPException):
            raise error
        logger.info(f"Image task failed with error: {error}")
        raise HTTPException(status_code=500, detail=str(error))
    return result["result"]


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def load_lora_weights_from_config(base_pipe, loras: List[LoRAConfig]):
    for lora_config in loras:
        load_kwargs = {"adapter_name": lora_config.adapter_name}
        if lora_config.weight_name:
            load_kwargs["weight_name"] = lora_config.weight_name
        if lora_config.type == "url":
            response = requests.get(lora_config.address)
            response.raise_for_status()
            weights_path = Path("weights") / f"{lora_config.adapter_name}.safetensors"
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            with open(weights_path, "wb") as f:
                f.write(response.content)
            source = str(weights_path)
            logger.info(f"Downloaded LoRA weights to {source}")
        elif lora_config.type in ("hub", "local"):
            # hub: repo id (weight_name optionally picks the file inside the repo)
            # local: path to a weights file or directory
            source = lora_config.address
        else:
            logger.warning("Skipping LoRA '%s' with unknown type '%s'",
                           lora_config.adapter_name, lora_config.type)
            continue
        logger.info(f"Setting up lora model {lora_config.adapter_name} with weights: {source}...")
        base_pipe.load_lora_weights(source, **load_kwargs)
    base_pipe.disable_lora()
    logger.info("Loaded LoRA weights for base pipeline")


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

        variations_config = config.pipeline.variations_config
        if variations_config.enabled and not variations_config.vision_model_api_key:
            env_var = variations_config.vision_model_api_key_variable
            variations_config.vision_model_api_key = (os.getenv(env_var) if env_var else None) or "ignored"

        initialized_pipelines = {
            "generation": None,
            "edit": None,
            "edit_inpaint": None,
            "variation": None,
            "redux": None,
        }

        init_dtype = get_torch_dtype(
            config.pipeline.torch_dtype_init if config.pipeline.torch_device != "balanced" else config.pipeline.torch_dtype_run)
        run_dtype = get_torch_dtype(config.pipeline.torch_dtype_run)
        device_map = config.pipeline.torch_device

        # Load the base pipeline all other pipelines are derived from
        base_pipe = AutoPipelineForText2Image.from_pretrained(config.pipeline.hf_model_id, torch_dtype=init_dtype,
                                                              device_map=device_map)
        logger.info(f"Setting up model with base pipeline: '{base_pipe.__class__.__name__}'...")

        # Load LoRA weights once on the base pipeline; all derived pipelines share them
        if config.pipeline.loras:
            load_lora_weights_from_config(base_pipe, config.pipeline.loras)

        if config.pipeline.generations_config.enabled:
            logger.info("Loading image generation pipeline...")
            if config.pipeline.generations_config.pipeline is not None:
                initialized_pipelines["generation"] = getattr(diffusers, config.pipeline.generations_config.pipeline).from_pipe(base_pipe, torch_dtype=run_dtype)
            else:
                initialized_pipelines["generation"] = AutoPipelineForText2Image.from_pipe(base_pipe, torch_dtype=run_dtype)
            logger.info("Finished loading image generation pipeline.")

        if config.pipeline.edits_config.enabled:
            logger.info("Loading image edit pipeline...")
            if config.pipeline.edits_config.pipeline is not None:
                initialized_pipelines["edit"] = getattr(diffusers, config.pipeline.edits_config.pipeline).from_pipe(base_pipe, torch_dtype=run_dtype)
            else:
                initialized_pipelines["edit"] = AutoPipelineForInpainting.from_pipe(base_pipe, torch_dtype=run_dtype)
            logger.info("Finished loading image edit pipeline.")
            if config.pipeline.edits_config.inpaint_pipeline is not None:
                initialized_pipelines["edit_inpaint"] = getattr(diffusers, config.pipeline.edits_config.inpaint_pipeline).from_pipe(base_pipe, torch_dtype=run_dtype)
                logger.info("Finished loading image inpaint edit pipeline.")

        if config.pipeline.variations_config.enabled:
            logger.info("Loading image variation (image-to-image) pipeline...")
            if config.pipeline.variations_config.enable_flux_redux:
                # FLUX Redux: the base text2image pipeline is driven by redux embeddings
                initialized_pipelines["variation"] = base_pipe
                repo_redux = "black-forest-labs/FLUX.1-Redux-dev"
                initialized_pipelines["redux"] = FluxPriorReduxPipeline.from_pretrained(repo_redux, torch_dtype=run_dtype,
                                                                                        device_map=device_map)
            elif config.pipeline.variations_config.pipeline is not None:
                initialized_pipelines["variation"] = getattr(diffusers, config.pipeline.variations_config.pipeline).from_pipe(base_pipe, torch_dtype=run_dtype)
            else:
                initialized_pipelines["variation"] = AutoPipelineForImage2Image.from_pipe(base_pipe, torch_dtype=run_dtype)
            logger.info("Finished loading image variation pipeline.")

        # Drop our own reference to the base pipeline (components stay shared)
        del base_pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Apply further settings
        for pipe_name, pipe in initialized_pipelines.items():
            if pipe is not None and pipe_name != "redux":
                vae = getattr(pipe, "vae", None)
                if vae is not None:
                    if config.pipeline.enable_vae_slicing and hasattr(vae, "enable_slicing"):
                        vae.enable_slicing()
                    if config.pipeline.enable_vae_tiling and hasattr(vae, "enable_tiling"):
                        vae.enable_tiling()
                if config.pipeline.enable_cpu_offload and config.pipeline.torch_device != "balanced":
                    try:
                        pipe.enable_model_cpu_offload()
                    except Exception:
                        logger.warning("Could not enable CPU offload for %s pipeline", pipe_name, exc_info=True)

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
        logger.error("Error during setup: %s", e)
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
    logger.info("Setting log level from env to %s", loglevel)
    logging.basicConfig(level=logging.getLevelName(loglevel))
    logHandler = logging.StreamHandler()
    formatter = JsonFormatter(timestamp=True, reserved_attrs=RESERVED_ATTRS, datefmt='%Y-%m-%d %H:%M:%S')
    logHandler.setFormatter(formatter)
    logging.getLogger().handlers.clear()
    logging.getLogger().addHandler(logHandler)
    uvi_logger = logging.getLogger("uvicorn.access")
    uvi_logger.handlers.clear()
    uvi_logger.addHandler(logHandler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    setup_logging()
    pipelines, config, data_path = setup()

    app.generations_pipe = pipelines.get("generation")
    app.edits_pipe = pipelines.get("edit")
    app.inpaint_edits_pipe = pipelines.get("edit_inpaint")
    app.variations_pipe = pipelines.get("variation")
    app.redux_pipe = pipelines.get("redux")

    app.pipe_config = config
    app.data_path = data_path

    app.base_pipe = app.generations_pipe or app.edits_pipe or app.variations_pipe

    # Start the single permanent GPU worker
    Thread(target=worker, daemon=True, name="fastfusion-worker").start()

    yield
    # Clean up the ML models and release the resources
    app.generations_pipe = None
    app.edits_pipe = None
    app.inpaint_edits_pipe = None
    app.variations_pipe = None
    app.redux_pipe = None
    app.pipe_config = None
    app.base_pipe = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


fastfusion_app = FastFusionApp(lifespan=lifespan)


@fastfusion_app.middleware("http")
async def ensure_model_ready(request: Request, call_next):
    # Raising HTTPException inside middleware would surface as a 500;
    # return the 503 response directly instead. /health stays reachable.
    if request.url.path != "/health" and request.app.base_pipe is None:
        return JSONResponse(status_code=503, content={"detail": "Model not ready"})
    return await call_next(request)


@fastfusion_app.post("/v1/images/generations")
async def generate_images(request: Request, body: CreateImageRequest):
    app: FastFusionApp = request.app
    if not app.pipe_config.pipeline.generations_config.enabled:
        raise HTTPException(status_code=404, detail="Image generation not enabled")
    gen_pipe = app.generations_pipe
    if gen_pipe is None:
        raise HTTPException(status_code=503, detail="Image generation pipeline not loaded")

    response_format = body.response_format or "url"
    if not body.stream and response_format not in ("url", "b64_json"):
        raise HTTPException(status_code=400, detail=f"Unsupported response_format '{response_format}'")
    output_format = normalize_output_format(body.output_format)
    width, height = parse_size(body.size)
    guidance_scale, num_inference_steps = resolve_generation_params(
        app.pipe_config, body.quality, body.guidance_scale, body.num_inference_steps)
    images_to_generate = max(1, min(body.n or 1, app.pipe_config.pipeline.max_n))
    validate_lora_settings(gen_pipe, body.lora_settings)

    meta = build_event_meta(width, height, body.quality, body.background, output_format)
    partial_images = body.partial_images if body.stream else 0
    cancel: Dict[str, Any] = {"requested": False, "pipe": None}
    prompt = body.prompt
    lora_settings = body.lora_settings
    output_compression = body.output_compression

    logger.debug("Generating images with parameters: ")
    logger.debug(f"Prompt: {prompt}, Guidance Scale: {guidance_scale}, Steps: {num_inference_steps}, "
                 f"Images to Generate: {images_to_generate}, Size: {width}x{height}, Stream: {body.stream}")

    def run(emit, cancel_state):
        lora_active = apply_lora(gen_pipe, lora_settings)
        try:
            cancel_state["pipe"] = gen_pipe
            call_kwargs = dict(
                prompt=prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=images_to_generate,
            )
            callback = build_stream_callback(gen_pipe, emit, partial_images, num_inference_steps,
                                             width, height, "image_generation", meta,
                                             output_format, output_compression, cancel_state)
            if callback is not None:
                call_kwargs["callback_on_step_end"] = callback
            return gen_pipe(**filter_call_kwargs(gen_pipe, call_kwargs)).images
        finally:
            if lora_active:
                clear_lora(gen_pipe)

    if body.stream:
        return await stream_pipeline_task("image_generation", run, meta,
                                          output_format, output_compression, cancel)
    return await run_pipeline_task(run, request, response_format, output_format,
                                   output_compression, cancel)


@fastfusion_app.post("/v1/images/edits")
async def edit_images(
        request: Request,
        prompt: str = Form(...),
        image: Optional[List[UploadFile]] = File(None),
        image_array: Optional[List[UploadFile]] = File(None, alias="image[]"),
        mask: Optional[UploadFile] = File(None),
        n: int = Form(1),
        response_format: str = Form("url"),
        model: Optional[str] = Form(None),  # Ignored
        size: Optional[str] = Form(None),
        quality: Optional[str] = Form(None),
        background: Optional[str] = Form(None),  # Accepted, echoed in stream events
        output_format: Optional[str] = Form(None),
        output_compression: Optional[int] = Form(None),
        user: Optional[str] = Form(None),  # Ignored
        stream: bool = Form(False),
        partial_images: int = Form(0),
        strength: Optional[float] = Form(None),  # Addon over openAI
        guidance_scale: Optional[float] = Form(None),  # Addon over openAI
        num_inference_steps: Optional[int] = Form(None),  # Addon over openAI
        lora_settings: Optional[str] = Form(None),  # Addon over openAI; JSON list
):
    app: FastFusionApp = request.app
    if not app.pipe_config.pipeline.edits_config.enabled:
        raise HTTPException(status_code=404, detail="Image edits not enabled")

    # OpenAI clients send a single "image" field or repeated "image[]" fields
    uploads = [f for f in ((image or []) + (image_array or [])) if f is not None]
    if not uploads:
        raise HTTPException(status_code=400, detail="At least one image file is required")

    if not stream and response_format not in ("url", "b64_json"):
        raise HTTPException(status_code=400, detail=f"Unsupported response_format '{response_format}'")
    output_format_normalized = normalize_output_format(output_format)
    lora_parsed = parse_lora_settings(lora_settings)

    # Read uploads on the event loop; PIL work happens in the worker
    image_bytes = [await f.read() for f in uploads]
    mask_bytes = await mask.read() if mask is not None else None

    init_images = [Image.open(BytesIO(b)).convert("RGB") for b in image_bytes]
    mask_image = prepare_mask_image(mask_bytes) if mask_bytes else None
    width, height = parse_size(size, default=init_images[0].size)
    guidance_scale, num_inference_steps = resolve_generation_params(
        app.pipe_config, quality, guidance_scale, num_inference_steps)
    images_to_generate = max(1, min(n, app.pipe_config.pipeline.max_n))

    # Choose pipeline: dedicated inpaint pipeline when a mask is provided
    if mask_image is not None and app.inpaint_edits_pipe is not None:
        edit_pipe = app.inpaint_edits_pipe
    else:
        edit_pipe = app.edits_pipe
    if edit_pipe is None:
        raise HTTPException(status_code=503, detail="Image edit pipeline not loaded")
    validate_lora_settings(edit_pipe, lora_parsed)

    if mask_image is None and pipeline_accepts(edit_pipe, "mask_image"):
        # Mask-requiring pipelines (e.g. inpainting) cannot edit without one;
        # a full-white mask means "repaint everything guided by the prompt"
        logger.info("No mask provided for a mask-based edit pipeline; using a full edit mask")
        mask_image = Image.new("RGB", init_images[0].size, (255, 255, 255))
    if mask_image is not None and not pipeline_accepts(edit_pipe, "mask_image"):
        logger.warning("Pipeline %s does not accept masks; the provided mask is ignored",
                       type(edit_pipe).__name__)
        mask_image = None

    meta = build_event_meta(width, height, quality, background, output_format_normalized)
    stream_partials = partial_images if stream else 0
    cancel: Dict[str, Any] = {"requested": False, "pipe": None}

    logger.debug("Editing images with parameters: ")
    logger.debug(f"Prompt: {prompt}, Strength: {strength}, Guidance Scale: {guidance_scale}, "
                 f"Steps: {num_inference_steps}, Images to Generate: {images_to_generate}, "
                 f"Size: {width}x{height}, Input Images: {len(init_images)}, Stream: {stream}")

    def run(emit, cancel_state):
        lora_active = apply_lora(edit_pipe, lora_parsed)
        try:
            cancel_state["pipe"] = edit_pipe
            call_kwargs = dict(
                prompt=prompt,
                image=init_images if len(init_images) > 1 else init_images[0],
                mask_image=mask_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=images_to_generate,
                width=width,
                height=height,
            )
            callback = build_stream_callback(edit_pipe, emit, stream_partials, num_inference_steps,
                                             width, height, "image_edit", meta,
                                             output_format_normalized, output_compression, cancel_state)
            if callback is not None:
                call_kwargs["callback_on_step_end"] = callback
            return edit_pipe(**filter_call_kwargs(edit_pipe, call_kwargs)).images
        finally:
            if lora_active:
                clear_lora(edit_pipe)

    if stream:
        return await stream_pipeline_task("image_edit", run, meta,
                                          output_format_normalized, output_compression, cancel)
    return await run_pipeline_task(run, request, response_format, output_format_normalized,
                                   output_compression, cancel)


def get_image_description(image: Image.Image, config: FastFusionConfig) -> str:
    """
    Sends the image to the configured vision model to get a description.
    """
    variations_config = config.pipeline.variations_config
    host = (variations_config.vision_model_host or "").rstrip("/")
    # OpenAI-compatible servers expect the /v1 suffix on the base URL
    if host and not host.endswith("/v1"):
        host = f"{host}/v1"
    openai_client = openai.Client(api_key=variations_config.vision_model_api_key or "ignored",
                                  base_url=host or None)
    try:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        response = openai_client.chat.completions.create(
            model=variations_config.vision_model,
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
        return response.choices[0].message.content.strip()
    except Exception:
        logger.exception("Error getting image description; using fallback prompt")
        return "Create variations of the image at your best guess!"
    finally:
        openai_client.close()


@fastfusion_app.post("/v1/images/variations")
async def generate_variations(
        request: Request,
        image: UploadFile = File(...),
        n: int = Form(1),
        response_format: str = Form("url"),
        model: Optional[str] = Form(None),  # Ignored
        size: Optional[str] = Form(None),
        user: Optional[str] = Form(None),  # Ignored
        strength: Optional[float] = Form(None),  # Addon over openAI
        guidance_scale: Optional[float] = Form(None),  # Addon over openAI
        num_inference_steps: Optional[int] = Form(None),  # Addon over openAI
        lora_settings: Optional[str] = Form(None),  # Addon over openAI; JSON list
):
    app: FastFusionApp = request.app
    if not app.pipe_config.pipeline.variations_config.enabled:
        raise HTTPException(status_code=404, detail="Image variations not enabled")

    if response_format not in ("url", "b64_json"):
        raise HTTPException(status_code=400, detail=f"Unsupported response_format '{response_format}'")
    lora_parsed = parse_lora_settings(lora_settings)

    use_redux = app.redux_pipe is not None and app.pipe_config.pipeline.variations_config.enable_flux_redux
    var_pipe = app.base_pipe if use_redux else app.variations_pipe
    if var_pipe is None:
        raise HTTPException(status_code=503, detail="Image variation pipeline not loaded")
    validate_lora_settings(var_pipe, lora_parsed)

    image_bytes = await image.read()
    init_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    width, height = parse_size(size, default=init_image.size)
    guidance_scale, num_inference_steps = resolve_generation_params(
        app.pipe_config, None, guidance_scale, num_inference_steps)
    images_to_generate = max(1, min(n, app.pipe_config.pipeline.max_n))

    local_prompt = None
    if not use_redux:
        # Network call: keep it off the GPU worker and off the event loop
        local_prompt = await asyncio.to_thread(get_image_description, init_image, app.pipe_config)
        logger.info("Generating variations with parameters: ")
        logger.info(f"Prompt: {local_prompt}, Strength: {strength}, Guidance Scale: {guidance_scale}, "
                    f"Steps: {num_inference_steps}, Images to Generate: {images_to_generate}, Size: {width}x{height}")

    cancel: Dict[str, Any] = {"requested": False, "pipe": None}

    def run(emit, cancel_state):
        lora_active = apply_lora(var_pipe, lora_parsed)
        try:
            cancel_state["pipe"] = var_pipe
            if use_redux:
                # The redux pipe produces prompt embeddings from the input image
                extra_kwargs = app.redux_pipe(init_image)
                call_kwargs = dict(
                    num_images_per_prompt=images_to_generate,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    **extra_kwargs,
                )
            else:
                call_kwargs = dict(
                    prompt=local_prompt,
                    image=init_image,
                    strength=strength,
                    num_images_per_prompt=images_to_generate,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    width=width,
                    height=height,
                )
            return var_pipe(**filter_call_kwargs(var_pipe, call_kwargs)).images
        finally:
            if lora_active:
                clear_lora(var_pipe)

    return await run_pipeline_task(run, request, response_format, "png", None, cancel)


@fastfusion_app.get("/v1/images/data/{file_id}")
async def get_image_data(request: Request, file_id: str):
    # Strict file id check also prevents path traversal (e.g. encoded backslashes)
    match = FILE_ID_RE.fullmatch(file_id)
    if not match:
        raise HTTPException(status_code=404, detail="Image not found")
    file_path = os.path.join(request.app.data_path, file_id)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    _, media_type = IMAGE_FORMATS[match.group(1)]
    return FileResponse(file_path, media_type=media_type)


@fastfusion_app.get("/v1/models")
async def list_models(request: Request):
    # Some OpenAI clients probe this endpoint at startup
    name = request.app.pipe_config.name if request.app.pipe_config else "fastfusion"
    return {
        "object": "list",
        "data": [{"id": name, "object": "model", "created": int(time.time()), "owned_by": "fastfusion"}],
    }


@fastfusion_app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("FASTFUSION_HOST", "0.0.0.0")
    port = int(os.getenv("FASTFUSION_PORT", 9999))
    uvicorn.run(fastfusion_app, host=host, port=port)
