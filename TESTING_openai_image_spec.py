import asyncio
import inspect
import logging
import time
import uuid
import os
import threading
from datetime import datetime, timedelta
from typing import Annotated, Dict, Iterator, List, Literal, Optional, Union, AsyncGenerator

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
    user: Optional[str] = None  # Ignored


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
    prompt: str = None  # Requirement over openAI
    num_inference_steps: Optional[int] = 50  # Addon over openAI
    strength: Optional[float] = 0.75  # Addon over openAI
    guidance_scale: Optional[float] = 0.0  # Addon over openAI


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
        return request.copy()
        # if request["request_type"] == "edit":
        #     # Convert image and mask (if available) to PIL format
        #     request_dict['image'] = convert_to_pil_image(request["image"])
        #     if request["mask"]:
        #         request_dict['mask'] = convert_to_pil_image(request["mask"])
        #     request_dict['request_type'] = "edit"
        # elif request["request_type"] == "variation":
        #     # Convert image to PIL format
        #     request_dict['image'] = convert_to_pil_image(request["image"])
        # return request_dict

    def encode_response(self, output_generator: list[Dict]) -> list[Dict]:
        logger.debug("Encoding image response")
        encoded_responses = []
        for output in output_generator:
            logger.debug("Output: %s", output)
            if isinstance(output, dict) and "image" in output:
                img_str = output["image"]
                response_format = output.get('response_format', 'url')
                if response_format == "b64_json":
                    encoded_responses.append({"b64_json": img_str})
                elif response_format == "url":
                    # Save image and return URL
                    img_data = base64.b64decode(img_str)
                    img = Image.open(io.BytesIO(img_data))
                    file_id = f"{shortuuid()}.png"
                    file_path = os.path.join(self.data_path, file_id)
                    img.save(file_path, format="PNG")
                    encoded_responses.append({"url": f"/v1/images/data/{file_id}"})
            else:
                return {"error": "Unexpected output format"}
        return encoded_responses

    async def handle_images_generations_request(self, request: CreateImageRequest, background_tasks: BackgroundTasks):
        request_dict = request.model_dump()
        request_dict['request_type'] = "generation"
        return await self.handle_image_request(request_dict, background_tasks)

    async def handle_images_edits_request(self, request: CreateImageEditRequest, background_tasks: BackgroundTasks):
        request_dict = request.model_dump()
        request_dict['request_type'] = "edit"
        return await self.handle_image_request(request_dict, background_tasks)

    async def handle_images_variations_request(self, request: CreateImageVariationRequest,
                                               background_tasks: BackgroundTasks):
        request_dict = request.model_dump()
        request_dict['request_type'] = "variation"
        return await self.handle_image_request(request_dict, background_tasks)

    async def handle_image_request(self, request: Dict, background_tasks: BackgroundTasks):
        logger.debug("Received image request %s", request)
        responses = await self._server.lit_api.predict(request)
        logger.debug(f"Got {len(responses)} responses")
        return await self.collect_image_responses(request, responses)

    async def collect_image_responses(self, request: Dict, generator_list: List[Dict]):
        logger.debug("Collecting image responses")
        image_responses = self.encode_response(generator_list)
        final_response = {
            "created": int(time.time()),
            "data": image_responses
        }
        return Response(content=json.dumps(final_response), media_type="application/json")

    async def get_image_data(self, file_id: str):
        file_path = os.path.join(self.data_path, file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Image not found")
        return Response(content=open(file_path, "rb").read(), media_type="image/png")