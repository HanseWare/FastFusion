import asyncio
import inspect
import logging
import time
import uuid
import os
import threading
from datetime import datetime, timedelta
from typing import Annotated, Dict, Iterator, List, Literal, Optional, Union

from fastapi import BackgroundTasks, HTTPException, Request, Response
from pydantic import BaseModel, Field
from PIL import Image
import io
import base64
import json
from collections import deque

from litserve.specs.base import LitSpec
from litserve.utils import LitAPIStatus, azip

logger = logging.getLogger(__name__)

def shortuuid():
    return uuid.uuid4().hex[:6]


def convert_to_pil_image(image_data: str) -> Image.Image:
    """
    Converts various image formats (base64, binary PNG/JPEG) to a PIL Image object.
    """
    if image_data.startswith("data:image"):
        # Handling base64 encoded image data
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
    else:
        # Assuming binary image data
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


class CreateImageRequest(BaseModel):
    prompt: str
    model: Optional[str] = "flux.1-dev"
    n: Optional[int] = 1
    quality: Optional[str] = "standard"
    response_format: Optional[str] = "url"
    size: Optional[str] = "1024x1024"
    style: Optional[str] = "vivid"
    user: Optional[str] = None # Ignored


class CreateImageEditRequest(BaseModel):
    prompt: str
    image: str  # Assuming binary as a string
    mask: Optional[str] = None
    model: Optional[str] = "flux.1-dev"
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "url"
    user: Optional[str] = None # Ignored
    guidance_scale: Optional[float] = 7.0 # Addon over openAI
    num_inference_steps: Optional[int] = 50 # Addon over openAI

class CreateImageVariationRequest(BaseModel):
    image: str  # Assuming binary as a string
    model: Optional[str] = "flux.1-dev"
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "url"
    user: Optional[str] = None # Ignored
    prompt: str = None # Requirement over openAI
    num_inference_steps: Optional[int] = 50 # Addon over openAI
    strength: Optional[float] = 0.75 # Addon over openAI
    guidance_scale: Optional[float] = 0.0 # Addon over openAI


class OpenAIImageSpec(LitSpec):
    def __init__(
        self,
    ):
        super().__init__()
        # Register the endpoints
        self.events = None
        self.queues = None
        self.response_queue_id = None
        self.data_path = None
        self.add_endpoint("/v1/images/generations", self.handle_images_generations_request, ["POST"])
        self.add_endpoint("/v1/images/edits", self.handle_images_edits_request, ["POST"])
        self.add_endpoint("/v1/images/variations", self.handle_images_variations_request, ["POST"])
        self.add_endpoint("/v1/images/data/{file_id}", self.get_image_data, ["GET"])

    def setup(self, server: "LitServer"):
        super().setup(server)
        self.data_path = os.getenv("IMAGE_DATA_PATH", "/data")
        os.makedirs(self.data_path, exist_ok=True)
        # Launch cleanup thread
        cleanup_thread = threading.Thread(target=clean_old_images, args=(self.data_path,), daemon=True)
        cleanup_thread.start()
        print("OpenAI Image spec setup complete")

    def decode_request(self, request: Dict) -> Dict:
        request_dict = request.copy()
        if request["request_type"] is "edit":
            # Convert image and mask (if available) to PIL format
            request_dict['image'] = convert_to_pil_image(request["image"])
            if request["mask"]:
                request_dict['mask'] = convert_to_pil_image(request["mask"])
            request_dict['request_type'] = "edit"
        elif request["request_type"] is "variation":
            # Convert image to PIL format
            request_dict['image'] = convert_to_pil_image(request["image"])
        return request_dict

    def encode_response(self, output_generator: Iterator) -> Iterator[Dict]:
        image_responses = []
        for output in output_generator:
            if isinstance(output, str):
                image_responses.append({"result": output})
            elif isinstance(output, Image.Image):
                # Get the response format from the output or default to b64_json
                response_format = getattr(output, 'response_format', 'b64_json')
                if response_format == "b64_json":
                    # Convert PIL Image to base64 string
                    buffered = io.BytesIO()
                    output.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    image_responses.append({"b64_json": img_str})
                elif response_format == "url":
                    # Save image as a file and return the URL
                    file_id = f"{shortuuid()}.png"
                    file_path = os.path.join(self.data_path, file_id)
                    output.save(file_path, format="PNG")
                    image_responses.append({"url": f"/v1/images/data/{file_id}"})
            else:
                image_responses.append({"error": "Unexpected output format"})
        yield {"data": image_responses}

    async def handle_images_generations_request(self, request: CreateImageRequest, background_tasks: BackgroundTasks):
        request_dict = request.model_dump()
        request_dict['request_type'] = "generation"
        return await self.handle_image_request(request_dict, background_tasks)

    async def handle_images_edits_request(self, request: CreateImageEditRequest, background_tasks: BackgroundTasks):
        request_dict = request.model_dump()
        request_dict['request_type'] = "edit"
        return await self.handle_image_request(request_dict, background_tasks)

    async def handle_images_variations_request(self, request: CreateImageVariationRequest, background_tasks: BackgroundTasks):
        request_dict = request.model_dump()
        request_dict['request_type'] = "variation"
        return await self.handle_image_request(request_dict, background_tasks)

    async def handle_image_request(self, request: Dict, background_tasks: BackgroundTasks):
        response_queue_id = self.response_queue_id
        logger.debug("Received image request %s", request)
        uids = [uuid.uuid4() for _ in range(request["n"])]
        self.queues = []
        self.events = []
        for uid in uids:
            request_el = request.copy()
            request_el['n'] = 1
            q = deque()
            event = asyncio.Event()
            self._server.response_buffer[uid] = (q, event)
            self._server.request_queue.put((response_queue_id, uid, time.monotonic(), request_el))
            self.queues.append(q)
            self.events.append(event)

        responses = await self.get_from_queues(uids)

        response_task = asyncio.create_task(self.collect_image_responses(request, responses))
        response_list = await response_task
        return await response_list

    async def get_from_queues(self, uids) -> List[asyncio.Queue]:
        choice_pipes = []
        for uid, q, event in zip(uids, self.queues, self.events):
            data = self._server.data_streamer(q, event, send_status=True)
            choice_pipes.append(data)
        return choice_pipes

    async def collect_image_responses(self, request: Dict, generator_list: List[asyncio.Queue]):
        """
        Collects image responses from the server queues.
        """
        model = request["model"]
        image_responses = []
        # iterate over n responses
        for i, streaming_response in enumerate(generator_list):
            msgs = []
            async for response, status in streaming_response:
                if status == LitAPIStatus.ERROR:
                    raise response
                encoded_response = json.loads(response)
                msgs.append(encoded_response)

            content = msgs
            image_responses.extend(content)

        return Response(content=json.dumps({"data": image_responses}), media_type="application/json")

    async def get_image_data(self, file_id: str):
        file_path = os.path.join(self.data_path, file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Image not found")
        return Response(content=open(file_path, "rb").read(), media_type="image/png")
