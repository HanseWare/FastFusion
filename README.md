![GitHub Tag](https://img.shields.io/github/v/tag/HanseWare/FastFusion?&label=Latest)
[![Static Badge](https://img.shields.io/badge/dockerhub-images-important.svg?&logo=Docker)](https://hub.docker.com/repository/docker/hanseware/fastfusion)

# FastFusion

FastFusion provides an OpenAI-compatible API for a wide variety of Huggingface Diffusers models. This allows users to leverage the power and flexibility of Huggingface's diffusion models through a familiar API interface.

The project is built using the following key technologies:
- **FastAPI:** For building the high-performance API.
- **Huggingface Diffusers:** To access and run various diffusion models.
- **Pydantic:** For data validation and settings management.

FastFusion is designed for easy deployment, for example, on a Kubernetes cluster.

## Features

- **OpenAI API Compatibility:** Offers a familiar API structure, including endpoints like `/v1/images/generations`, `/v1/images/edits`, and `/v1/images/variations`.
- **Broad Model Support:** Leverages Huggingface Diffusers Autopipelines to support a wide array of diffusion models, including specific pipelines like `FluxPipeline` for advanced capabilities.
- **Dynamic LoRA Loading:** Allows for loading and applying LoRA (Low-Rank Adaptation) weights on a per-request basis, enabling flexible model customization.
- **Flexible Configuration:** Utilizes a `model_config.json` file for detailed configuration of models, precision (e.g., `torch_dtype_init`, `torch_dtype_run`), target device (CPU/GPU), and other operational parameters.
- **Performance Optimizations:**
    - **CPU Offloading:** Reduces VRAM usage by offloading parts of the model to the CPU.
    - **VAE Slicing and Tiling:** Enables processing of large images by breaking them into smaller, manageable parts.
    - **Configurable Inference Datatypes:** Allows specifying different `torch` dtypes for model initialization and runtime for fine-tuned performance.
- **Asynchronous Task Queuing:** Efficiently manages multiple image generation requests concurrently using a task queue, improving throughput and responsiveness.
- **Automatic Image Cleanup:** Periodically removes old image files from storage to manage disk space.
- **Health Check Endpoint:** Provides a `/health` endpoint for easy monitoring of the application's status.
- **Easy Deployment with Docker:** Comes with Docker support for straightforward deployment and scaling.
- **Generation Presets:** Supports predefined presets for generation parameters (e.g., guidance scale, number of inference steps) for consistent results.
- **Vision Model Integration (Optional):** Can integrate with vision models (like GPT-4o) to automatically generate descriptive prompts for image variations when a prompt is not provided.
- **FLUX-Redux Pipeline (Optional):** Supports the FLUX-Redux pipeline for producing advanced and detailed image variations.

## Usage

This section provides instructions on how to run FastFusion using Docker and how to interact with its API.

### Docker Usage

The easiest way to run FastFusion is using Docker.

**Basic Command:**

To run FastFusion with default settings and expose it on port 8000:
```bash
docker run -d -p 8000:8000 hanseware/fastfusion
```

**Customized Docker Run:**

You can customize the Docker container by mounting your own configuration file and image data directory.

*   **Mounting Custom `model_config.json`:**
    To use a custom configuration, mount your `model_config.json` file to `/app/model_config.json` inside the container.
    ```bash
    -v /path/to/your/custom_model_config.json:/app/model_config.json
    ```
*   **Mounting Local Image Data Directory:**
    To persist generated images or use local images for operations like edits/variations, mount a local directory to `/data` inside the container. This directory is also where the application stores images when `response_format: url` is used.
    ```bash
    -v /path/to/your/image_data:/data
    ```
*   **Exposing the Port:**
    The application runs on port 8000 inside the container. You can map it to any host port using:
    ```bash
    -p <host_port>:8000
    ```
*   **Environment Variables:**
    You'll need to set the `BASE_URL` environment variable so the application can correctly form image URLs. For local testing, this would typically be `http://localhost:<host_port>`.
    ```bash
    -e BASE_URL="http://localhost:<host_port>"
    ```
    If your `model_config.json` uses API keys for services like a vision model for image variations, provide them as environment variables (e.g., `OPENAI_API_KEY`).
    ```bash
    -e OPENAI_API_KEY="your_openai_api_key_here"
    ```

**Example Customized `docker run` Command:**

This example runs FastFusion, mapping host port 8080 to the container's port 8000, using a custom `model_config.json`, a custom data directory, and setting the necessary `BASE_URL`.

```bash
docker run -d \
    -p 8080:8000 \
    -v /path/to/your/custom_model_config.json:/app/model_config.json \
    -v /path/to/your/image_data:/data \
    -e BASE_URL="http://localhost:8080" \
    -e OPENAI_API_KEY="your_openai_api_key_if_needed" \
    hanseware/fastfusion
```
Remember to replace `/path/to/your/custom_model_config.json` and `/path/to/your/image_data` with the actual paths on your host machine, and adjust `<host_port>` if you use a port other than 8080.

### Configuration Customization

The primary way to configure FastFusion, including the Huggingface model to use, LoRAs, performance settings (like precision and CPU offloading), and which API features (generations, edits, variations) are enabled, is through the `model_config.json` file.

For detailed information on the structure and available options in `model_config.json`, please refer to the [Configuration](#configuration) section below.

### API Examples (using `curl`)

The following examples demonstrate how to interact with the FastFusion API using `curl`. Ensure FastFusion is running and accessible at `http://localhost:8000` (or your configured host and port).

**1. Image Generation**

*   **Basic Prompt:**
    Generates a single image with the specified prompt.

    ```bash
    curl -X POST http://localhost:8000/v1/images/generations \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "A futuristic cityscape at sunset"
    }'
    ```

*   **Prompt with Size, Number of Images (n), and Quality Preset:**
    Generates 2 images of size 512x512 using the "hd" quality preset.

    ```bash
    curl -X POST http://localhost:8000/v1/images/generations \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "A majestic dragon soaring through a stormy sky",
        "n": 2,
        "size": "512x512",
        "quality": "hd"
    }'
    ```
    *(Note: Quality presets are defined in `model_config.json` and map to specific `guidance_scale` and `num_inference_steps`.)*

**2. Image Generation with LoRA**

*   Generates an image applying the "flux-canny" LoRA with a weight of 0.8. Assumes "flux-canny" is defined in your `model_config.json`.

    ```bash
    curl -X POST http://localhost:8000/v1/images/generations \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "A cat wearing a superhero costume, edge detected style",
        "lora_settings": [
            {
                "name": "flux-canny",
                "weight": 0.8
            }
        ]
    }'
    ```

**3. Image Editing**

*   Edits an image based on a prompt and an optional mask. The `image` and `mask` files are sent as form data.

    ```bash
    curl -X POST http://localhost:8000/v1/images/edits \
    -F "prompt=A pirate ship sailing on a calm sea" \
    -F "image=@/path/to/your/image.png" \
    -F "mask=@/path/to/your/mask.png" \
    -F "n=1" \
    -F "size=1024x1024" \
    -F "response_format=url"
    ```
    *(Replace `/path/to/your/image.png` and `/path/to/your/mask.png` with actual file paths. The mask is optional.)*

**4. Image Variation**

*   **Basic Variation:**
    Generates variations of an uploaded image.

    ```bash
    curl -X POST http://localhost:8000/v1/images/variations \
    -F "image=@/path/to/your/input_image.png" \
    -F "n=1" \
    -F "size=1024x1024" \
    -F "response_format=url" \
    -F "model=flux.1-dev" # Model parameter is often required
    ```
    *(Replace `/path/to/your/input_image.png` with your image file path. If the `variations_config.enable_flux_redux` is false and no prompt is provided, the system might try to use a vision model to generate a prompt if configured.)*

*   **Variation with a Custom Prompt:**
    Generates variations of an image, guided by a specific prompt.

    ```bash
    curl -X POST http://localhost:8000/v1/images/variations \
    -F "image=@/path/to/your/input_image.png" \
    -F "prompt=Make it look like a watercolor painting" \
    -F "n=1" \
    -F "size=1024x1024" \
    -F "response_format=url" \
    -F "model=flux.1-dev" # Model parameter is often required
    ```
    *(Replace `/path/to/your/input_image.png` with your image file path.)*

## Configuration

FastFusion is configured primarily through a `model_config.json` file and environment variables.

### 1. `model_config.json` Structure

This JSON file is the main configuration for the model pipeline, features, and presets. It typically resides at the root of the application or can be specified via the `CONFIG_PATH` environment variable.

The top-level keys are:

*   `name` (string): An arbitrary name for this configuration (e.g., "FLUX.1-dev").
*   `pipeline` (object): Defines the core model loading and operational settings. See details below.
*   `generation_presets` (object): A dictionary of predefined quality settings for image generation. See details below.

**Example `model_config.json`:**
```json
{
  "name": "FLUX.1-dev",
  "pipeline": {
    // ... pipeline object details ...
  },
  "generation_presets": {
    // ... generation_presets object details ...
  }
}
```

### 2. `pipeline` Object

This object contains all settings related to the diffusion model, its performance, and enabled API features.

*   `hf_model_id` (string): The Huggingface Hub model identifier (e.g., `"stabilityai/stable-diffusion-xl-base-1.0"`).
*   `max_n` (integer): The maximum number of images (`n`) that can be requested in a single API call.
*   `torch_dtype_init` (string): The PyTorch data type used when initially loading the model (e.g., `"bfloat16"`, `"float32"`). Helps manage memory during setup.
*   `torch_dtype_run` (string): The PyTorch data type used during model inference (e.g., `"float16"`, `"bfloat16"`). Can affect performance and VRAM usage.
*   `torch_device` (string): The primary device to run the model on (e.g., `"cuda"` for NVIDIA GPUs, `"cpu"`).
*   `enable_cpu_offload` (boolean): If `true`, enables model offloading to CPU, which can significantly reduce VRAM usage at the cost of slower inference.
*   `enable_vae_slicing` (boolean): If `true`, enables VAE slicing, allowing the generation of larger images without running out of VRAM by processing the VAE step in slices.
*   `enable_vae_tiling` (boolean): If `true`, enables VAE tiling, another technique to handle large images by breaking them into tiles. Useful for very high resolutions.
*   `global_guidance_scale` (float): The default `guidance_scale` (CFG scale) to be used for image generation if not specified in the request or a preset.
*   `global_num_inference_steps` (integer): The default number of inference steps for image generation if not specified in the request or a preset.

#### `generations_config` Object
Settings for the `/v1/images/generations` endpoint.
*   `enabled` (boolean): Set to `true` to enable this endpoint, `false` to disable.
*   `pipeline` (string): The specific Huggingface Diffusers pipeline class to use for text-to-image generation (e.g., `"AutoPipelineForText2Image"`, `"FluxPipeline"`).

#### `edits_config` Object
Settings for the `/v1/images/edits` endpoint.
*   `enabled` (boolean): Set to `true` to enable this endpoint, `false` to disable.
*   `pipeline` (string): The specific Huggingface Diffusers pipeline class for image editing/inpainting (e.g., `"AutoPipelineForInpainting"`, `"FluxFillPipeline"`).

#### `variations_config` Object
Settings for the `/v1/images/variations` endpoint.
*   `enabled` (boolean): Set to `true` to enable this endpoint, `false` to disable.
*   `pipeline` (string): The specific Huggingface Diffusers pipeline class for image-to-image tasks or variations (e.g., `"AutoPipelineForImage2Image"`).
*   `enable_flux_redux` (boolean): If `true`, and the base model is compatible (like FLUX), this enables the FLUX.1-Redux pipeline for generating variations, which can produce more detailed and creative results. The `FLUX.1-Redux-dev` model will be loaded automatically.
*   `vision_model` (string, optional): The identifier of a vision model (e.g., `"gpt-4o"`, `"llava-hf/llava-1.5-7b-hf"`) used to automatically generate a descriptive prompt if no prompt is provided in a variation request.
*   `vision_model_host` (string, optional): The API endpoint host for the vision model (e.g., `"https://api.openai.com/v1"`).
*   `vision_model_api_key_variable` (string, optional): The name of the environment variable that holds the API key for the vision model (e.g., `"OPENAI_API_KEY"`).

#### `loras` Array
A list of LoRA (Low-Rank Adaptation) configurations to be loaded at startup. These LoRAs can then be applied dynamically per API request.
Each object in the array has the following properties:
*   `type` (string): The source of the LoRA. Can be:
    *   `"hub"`: Download from Huggingface Hub. `address` should be the model ID.
    *   `"local"`: Load from a local file path. `address` should be the path to the `.safetensors` file.
    *   `"url"`: Download from a direct URL. `address` should be the URL to the `.safetensors` file.
*   `address` (string): The identifier, path, or URL for the LoRA model.
*   `weight_name` (string, optional): Specific name of the weight file. Often not needed if `address` directly points to the file or for Hub models where it's inferred. For `url` type, if the URL doesn't end with a clear filename, this can be used to name the downloaded file (e.g., `"my_lora_weights"` which becomes `my_lora_weights.safetensors`).
*   `adapter_name` (string): A unique name you assign to this LoRA. This name is used in the `lora_settings` of API requests to apply this LoRA (e.g., `"flux-canny"`, `"style_xyz"`).

**Example LoRA Configuration:**
```json
"loras": [
  {
    "type": "hub",
    "address": "black-forest-labs/FLUX.1-Canny-dev-lora",
    "adapter_name": "flux-canny"
  },
  {
    "type": "local",
    "address": "/opt/loras/my_style.safetensors",
    "adapter_name": "my-art-style"
  },
  {
    "type": "url",
    "address": "https://example.com/path/to/another_lora.safetensors",
    "weight_name": "another_lora_downloaded",
    "adapter_name": "another-lora"
  }
]
```

### 3. `generation_presets` Object

This object is a dictionary where each key is a preset name (e.g., `"standard"`, `"hd"`, `"fast"`) and the value is an object defining parameters for that preset. These presets can be requested via the `quality` parameter in image generation API calls.

Each preset object has:
*   `guidance_scale` (float): The Classifier-Free Guidance (CFG) scale for this preset.
*   `num_inference_steps` (integer): The number of diffusion steps for this preset.

**Example `generation_presets`:**
```json
"generation_presets": {
  "standard": {
    "guidance_scale": 7.0,
    "num_inference_steps": 25
  },
  "hd": {
    "guidance_scale": 5.0,
    "num_inference_steps": 50
  }
}
```

### 4. Environment Variables

Several environment variables are used to configure the application's runtime behavior:

*   `IMAGE_DATA_PATH`: Path to the directory where generated images are stored if `response_format: "url"` is used.
    *   Default: `/data`
*   `BASE_URL` (Required): The base URL of the FastFusion service. This is crucial for constructing the correct image URLs in API responses when `response_format: "url"` is used.
    *   Example: `http://localhost:8000` or `https://yourdomain.com/fastfusion`
*   `CONFIG_PATH`: Filesystem path to the `model_config.json` file.
    *   Default: `model_config.json` (expects it in the current working directory of the app)
*   `FASTFUSION_LOGLEVEL`: Sets the logging verbosity for the application.
    *   Examples: `"INFO"`, `"DEBUG"`, `"WARNING"`, `"ERROR"`
    *   Default: `"INFO"`
*   Value of `vision_model_api_key_variable` (e.g., `OPENAI_API_KEY`): If you're using a vision model that requires an API key (as specified in `variations_config.vision_model_api_key_variable`), this environment variable must be set to that key.
    *   Example: If `vision_model_api_key_variable` is `"OPENAI_API_KEY"`, then you set `OPENAI_API_KEY="your_actual_key_here"`.
*   `FASTFUSION_HOST`: The host address the Uvicorn server binds to when running `app.py` directly.
    *   Default: `"0.0.0.0"` (listens on all available network interfaces)
*   `FASTFUSION_PORT`: The port the Uvicorn server binds to when running `app.py` directly.
    *   Default: `9999`

## API Documentation

This section details the API endpoints provided by FastFusion.

### Common Response Object for Image Endpoints

Endpoints that generate images (`/generations`, `/edits`, `/variations`) return a JSON object with the following structure:

```json
{
  "created": 1678886400, // Unix timestamp of creation
  "data": [
    {
      "url": "http://localhost:8000/v1/images/data/your_image_id.png" // If response_format is "url"
    },
    {
      "b64_json": "iVBORw0KGgoAAAANSUhEUg..." // If response_format is "b64_json"
    }
    // ... more image objects if n > 1
  ]
}
```

### 1. `POST /v1/images/generations`

*   **Purpose:** Generates images from text prompts.
*   **Request Body (Content-Type: `application/json`):**
    Corresponds to the `CreateImageRequest` model.
    *   `prompt` (string, required): The main text prompt describing the image to generate.
    *   `model` (string, optional): Specifies the model to use. Typically fixed by the server-side `model_config.json`.
    *   `n` (integer, optional, default: 1): The number of images to generate. Limited by `max_n` in `model_config.json`.
    *   `quality` (string, optional): A predefined quality preset name (e.g., `"standard"`, `"hd"`) from `generation_presets` in `model_config.json`. This sets `guidance_scale` and `num_inference_steps`.
    *   `guidance_scale` (float, optional): Overrides the `guidance_scale` from the `quality` preset or the global default. Controls how much the image generation adheres to the prompt.
    *   `num_inference_steps` (integer, optional): Overrides the `num_inference_steps` from the `quality` preset or the global default. Controls the number of diffusion steps.
    *   `response_format` (string, optional, default: `"url"`): Specifies the format of the image data in the response.
        *   `"url"`: Returns a URL to the generated image.
        *   `"b64_json"`: Returns the image as a Base64 encoded string.
    *   `size` (string, optional, default: `"1024x1024"`): The desired dimensions of the image in `"WidthxHeight"` format (e.g., `"512x512"`, `"1024x768"`).
    *   `lora_settings` (array of objects, optional): A list of LoRA settings to apply during generation. Each object should have:
        *   `name` (string): The `adapter_name` of a LoRA defined in `model_config.json`.
        *   `weight` (float, optional, default: 1.0): The weight to apply to this LoRA.
    *   `style` (string, optional, default: `"vivid"`): Hint for desired image style (OpenAI compatible, may not be used by all models).
    *   `user` (string, optional): User identifier (OpenAI compatible, ignored by FastFusion).
*   **Response (JSON):** Common image response object (see above).

### 2. `POST /v1/images/edits`

*   **Purpose:** Edits an existing image based on a text prompt, using an optional mask for inpainting.
*   **Request Body (Content-Type: `multipart/form-data`):**
    *   `prompt` (string, required): Text prompt describing the desired edits.
    *   `image` (file, required): The initial image file to be edited.
    *   `mask` (file, optional): An optional mask image file. In an image editing task, the mask isolates the area of the image that should be modified. White areas in the mask are typically inpainted, black areas are preserved.
    *   `model` (string, optional): Model name (usually fixed by config).
    *   `n` (integer, optional, default: 1): Number of edited images to generate.
    *   `size` (string, optional, default: `"1024x1024"`): Desired output dimensions (`"WidthxHeight"`).
    *   `response_format` (string, optional, default: `"url"`): `"url"` or `"b64_json"`.
    *   `strength` (float, optional): How much the original image content should be transformed by the diffusion process (typically 0.0 to 1.0). A higher value means more change.
    *   `guidance_scale` (float, optional): CFG scale.
    *   `num_inference_steps` (integer, optional): Number of diffusion steps.
    *   `lora_settings` (string, optional): A JSON string representing an array of LoRA settings (e.g., `'[{"name": "my-lora", "weight": 0.7}]'`). Each object in the array should have `name` (string) and `weight` (float, optional).
    *   `user` (string, optional): User identifier (ignored).
*   **Response (JSON):** Common image response object.

### 3. `POST /v1/images/variations`

*   **Purpose:** Creates variations of an input image, optionally guided by a text prompt.
*   **Request Body (Content-Type: `multipart/form-data`):**
    *   `image` (file, required): The initial image file to generate variations from.
    *   `prompt` (string, optional): A text prompt to guide the variations. If not provided and a vision model is configured (see `variations_config` in `model_config.json`), a prompt will be automatically generated from the input image.
    *   `model` (string, optional): Model name (usually fixed by config, e.g., `"flux.1-dev"`).
    *   `n` (integer, optional, default: 1): Number of variations to generate.
    *   `size` (string, optional, default: `"1024x1024"`): Desired output dimensions (`"WidthxHeight"`).
    *   `response_format` (string, optional, default: `"url"`): `"url"` or `"b64_json"`.
    *   `strength` (float, optional): How much the original image should be altered (typically 0.0 to 1.0). Effective range can depend on the model.
    *   `guidance_scale` (float, optional): CFG scale.
    *   `num_inference_steps` (integer, optional): Number of diffusion steps.
    *   `lora_settings` (string, optional): A JSON string representing an array of LoRA settings (e.g., `'[{"name": "my-lora", "weight": 0.7}]'`).
    *   `user` (string, optional): User identifier (ignored).
*   **Response (JSON):** Common image response object.

### 4. `GET /v1/images/data/{file_id}`

*   **Purpose:** Retrieves a previously generated image by its file ID. This endpoint is used when `response_format: "url"` was specified in a generation/edit/variation request.
*   **Path Parameters:**
    *   `file_id` (string, required): The unique identifier of the image file (e.g., `"abcdef.png"` as returned in the `url` field of the image response object).
*   **Response:** The raw image file (e.g., `image/png`).

### 5. `GET /health`

*   **Purpose:** Provides a simple health check for the FastFusion service.
*   **Request Body:** None.
*   **Response (Content-Type: `application/json`):**
    ```json
    {
      "status": "ok"
    }
    ```

## Models Supported

FastFusion is designed to be flexible, supporting a wide range of diffusion models available through the Huggingface Hub. The primary factor for compatibility is a model's support within the Huggingface Diffusers library.

The application uses specific pipeline classes from the Diffusers library for its core functionalities. You can configure the `hf_model_id` in `model_config.json` to point to your desired Huggingface Hub model.

### General Compatibility

Generally, models compatible with the following base AutoPipelines (or their more specific counterparts as configured in `model_config.json`) should work:

*   **For Image Generations (`/v1/images/generations`):**
    Models compatible with `AutoPipelineForText2Image`. The `model_config.json` can specify a more concrete pipeline like `FluxPipeline` if the chosen model requires it (e.g., "ChuckMcSneed/FLUX.1-dev").
*   **For Image Edits (`/v1/images/edits`):**
    Models compatible with `AutoPipelineForInpainting`. Similarly, a specific pipeline like `FluxFillPipeline` can be configured.
*   **For Image Variations (`/v1/images/variations`):**
    Models compatible with `AutoPipelineForImage2Image`. If `enable_flux_redux` is true and a FLUX model is used, the `FluxPipeline` (via the base pipe) and `FluxPriorReduxPipeline` are utilized.

### Finding Compatible Models on Huggingface Hub

To find models on the [Huggingface Hub](https://huggingface.co/models) that are likely compatible with FastFusion:

1.  **Search with Relevant Tags:** Look for models tagged with keywords like:
    *   `diffusers` (essential)
    *   `text-to-image` (for generations)
    *   `image-to-image` (for variations)
    *   `inpainting` (for edits)
    *   Specific model architectures like `stable-diffusion`, `sdxl`, `flux`, etc.
2.  **Check the Model Card:**
    *   **Diffusers Compatibility:** The model card usually explicitly states if it's compatible with the Diffusers library.
    *   **Pipeline Information:** Look for information about which Diffusers pipeline the model is intended to be used with (e.g., `StableDiffusionXLPipeline`, `FluxPipeline`). This will help you determine if it fits the pipelines FastFusion uses (or can be configured to use).
    *   **Instantiation Code:** Model cards often provide Python code snippets for loading the model with Diffusers. This can give you a clear indication of the correct pipeline and model ID.

### Configuring a New Model

Once you've found a suitable model:

1.  Update the `hf_model_id` in your `model_config.json` to the Huggingface Hub identifier of the chosen model (e.g., `"stabilityai/stable-diffusion-xl-base-1.0"`).
2.  Adjust the `pipeline` names in `generations_config`, `edits_config`, or `variations_config` within `model_config.json` if the new model requires a more specific pipeline class than the default `AutoPipelineFor...` types. For example, if using a FLUX model for generations, you would set `generations_config.pipeline` to `"FluxPipeline"`.

Always ensure that the chosen model's license is appropriate for your use case.

## Advanced Usage

This section covers some of the more advanced features of FastFusion that allow for greater control and customization over the image generation process.

### 1. Using LoRAs (Low-Rank Adaptations)

LoRAs allow you to modify or fine-tune the behavior of the base diffusion model without needing to retrain the entire model. This is useful for applying specific styles, character concepts, or other artistic modifications.

*   **Defining LoRAs:** Available LoRAs are defined in the `loras` array within the `pipeline` object in your `model_config.json` file. Each LoRA entry specifies its source (Huggingface Hub, local file, or URL) and an `adapter_name` that you'll use to reference it. For more details, see the `loras` Array documentation in the [Configuration](#configuration) section.

*   **Applying LoRAs at Inference:** To use a LoRA during image generation, editing, or variation, you include the `lora_settings` parameter in your API request.
    *   For `/v1/images/generations` (JSON body): `lora_settings` is an array of objects, where each object contains `name` (the `adapter_name` of the LoRA) and `weight` (float, optional, default: 1.0).
    *   For `/v1/images/edits` and `/v1/images/variations` (form-data): `lora_settings` is a JSON string representing an array of these LoRA setting objects (e.g., `'[{"name": "flux-canny", "weight": 0.8}]'`).

    This allows you to dynamically combine and apply multiple LoRAs with specific weights for each request.

### 2. FLUX.1-Redux for Enhanced Variations

If you are using a FLUX model (like "ChuckMcSneed/FLUX.1-dev") as your base model, FastFusion can leverage the FLUX.1-Redux pipeline for generating image variations.

*   **Enabling FLUX.1-Redux:** This feature is enabled by setting `enable_flux_redux: true` in the `variations_config` section of your `model_config.json`.
*   **How it Works:** When enabled, the system uses the `FluxPriorReduxPipeline` in conjunction with the main FLUX pipeline. The Redux pipeline processes the input image to extract its embeddings, which then guide the main FLUX pipeline to generate variations that are often more coherent, detailed, and faithful to the original image's core concepts compared to standard image-to-image variations.

### 3. Vision Model for Automatic Prompts (Variations)

For the `/v1/images/variations` endpoint, FastFusion can automatically generate a descriptive prompt if one is not provided by the user. This is particularly useful when you want to create variations of an image without having a specific textual idea of what those variations should be.

*   **Configuration:** This feature is configured via the `variations_config` section in `model_config.json`:
    *   `vision_model`: The identifier of the vision model to use (e.g., `"gpt-4o"`).
    *   `vision_model_host`: The API endpoint for the vision model.
    *   `vision_model_api_key_variable`: The name of the environment variable that stores the API key for the vision model (e.g., `"OPENAI_API_KEY"`). Ensure this environment variable is set when running FastFusion.
*   **Usage:** If a request to `/v1/images/variations` is made without a `prompt` field, and the vision model is configured, FastFusion will send the input image to the specified vision model. The vision model's textual description of the image will then be used as the prompt to guide the variation generation process.

## Limitations

*   **GPU Selection:** GPU selection is controlled by the `torch_device` setting in `model_config.json` (e.g., `"cuda"`, `"cuda:0"`, `"cpu"`) and the standard `CUDA_VISIBLE_DEVICES` environment variable. The application does not currently offer explicit control over allocating a specific number of multiple GPUs beyond what PyTorch provides based on these settings.
*   **Single Model Instance:** Only one primary diffusion model (defined by `hf_model_id` in `model_config.json`) can be loaded at a time per running instance of the FastFusion application. To serve multiple base models simultaneously, you would need to run multiple instances of FastFusion with different configurations.
*   **Error Reporting:** Error handling for issues during Huggingface model loading or pipeline operations could be more verbose in API responses. Detailed logs are available server-side.
*   Further testing on a wider range of models, LoRAs, and hardware configurations is ongoing. Community feedback and contributions in this area are welcome.

## Contributing

We welcome contributions from the community to help make FastFusion even better! Here are some ways you can contribute:

*   **Reporting Bugs:** If you encounter any bugs or unexpected behavior, please open an issue on GitHub. Before creating a new issue, please check existing issues to see if your problem has already been reported.
*   **Suggesting Enhancements:** If you have ideas for new features or improvements to existing ones, feel free to open an issue to discuss them. We appreciate detailed suggestions!
*   **Submitting Pull Requests:** We are happy to review pull requests for:
    *   Bug fixes
    *   New features or enhancements
    *   Documentation improvements (like updates to this README, API docs, or adding examples)
    *   Improvements to code clarity or performance

If you plan to make code changes, we generally recommend following a standard GitHub workflow:
1.  **Fork** the repository to your own GitHub account.
2.  Create a new **branch** in your fork for your changes.
3.  Make your changes in your branch.
4.  Test your changes thoroughly.
5.  Submit a **pull request** from your branch to the main FastFusion repository.

Please ensure your pull request descriptions are clear and explain the purpose and scope of your changes.

Thank you for considering contributing to FastFusion!

## License
[MIT](https://choosealicense.com/licenses/mit/)