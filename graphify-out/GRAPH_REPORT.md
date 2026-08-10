# Graph Report - f:\private-git\fastfusion  (2026-07-24)

## Corpus Check
- Corpus is ~6,997 words - fits in a single context window. You may not need a graph.

## Summary
- 93 nodes · 168 edges · 7 communities
- Extraction: 73% EXTRACTED · 26% INFERRED · 1% AMBIGUOUS · INFERRED: 43 edges (avg confidence: 0.88)
- Token cost: 24,000 input · 7,000 output

## Community Hubs (Navigation)
- App Shell and Response Serving
- Config Schema and Generation
- Model Setup and VRAM Tuning
- Task Queue and Edit Endpoints
- Vision Prompting and Image I/O
- Dependency Stack and LoRA Tooling
- Public API Surface and Deployment

## God Nodes (most connected - your core abstractions)
1. `setup()` - 14 edges
2. `Python Dependency Set` - 12 edges
3. `edit_images()` - 8 edges
4. `pipeline Config Object` - 8 edges
5. `get_image_description()` - 7 edges
6. `generate_variations()` - 7 edges
7. `ensure_worker_running()` - 6 edges
8. `generate_images()` - 6 edges
9. `encode_response()` - 6 edges
10. `variations_config` - 6 edges

## Surprising Connections (you probably didn't know these)
- `fastapi` --references--> `FastFusionApp`  [INFERRED]
  requirements.txt → app.py
- `Single Model Instance Limitation` --references--> `setup()`  [INFERRED]
  README.md → app.py
- `FastFusion` --references--> `FastFusionApp`  [INFERRED]
  README.md → app.py
- `Diffusers AutoPipelines` --references--> `setup()`  [INFERRED]
  README.md → app.py
- `Dynamic LoRA Loading` --references--> `setup()`  [INFERRED]
  README.md → app.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **OpenAI-Compatible Image API Surface** — readme_images_generations_endpoint, readme_images_edits_endpoint, readme_images_variations_endpoint, readme_images_data_endpoint, readme_health_endpoint [EXTRACTED 1.00]
- **LoRA Load-and-Apply Lifecycle** — readme_loras_array, pydantic_models_loraconfig, pydantic_models_lorasetting, app_setup, app_generate_images, readme_dynamic_lora_loading [INFERRED 0.85]
- **VRAM Reduction Strategy** — readme_cpu_offloading, readme_vae_slicing, readme_vae_tiling, readme_torch_dtype_config [INFERRED 0.85]

## Communities (7 total, 0 thin omitted)

### Community 0 - "App Shell and Response Serving"
Cohesion: 0.14
Nodes (19): encode_response(), ensure_model_ready(), FastFusionApp, get_image_data(), health(), lifespan(), setup_logging(), shortuuid() (+11 more)

### Community 1 - "Config Schema and Generation"
Cohesion: 0.19
Nodes (17): generate_images(), BaseModel, CreateImageRequest, EditsConfig, FastFusionConfig, GenerationPreset, GenerationsConfig, LoRAConfig (+9 more)

### Community 2 - "Model Setup and VRAM Tuning"
Cohesion: 0.18
Nodes (15): clean_old_images(), get_torch_dtype(), Cleans up image files in the data directory that are older than 1 hour., setup(), Automatic Image Cleanup, Diffusers AutoPipelines, CPU Offloading, CONFIG_PATH (+7 more)

### Community 3 - "Task Queue and Edit Endpoints"
Cohesion: 0.21
Nodes (12): edit_images(), ensure_worker_running(), generate_variations(), worker(), LoRASetting, Asynchronous Task Queuing, edits_config, FLUX.1-Redux Pipeline (+4 more)

### Community 4 - "Vision Prompting and Image I/O"
Cohesion: 0.24
Nodes (10): convert_to_pil_image(), get_image_description(), Sends the image to Vision model to get a description., Image, VariationsConfig, OPENAI_API_KEY, variations_config, Vision Model Automatic Prompting (+2 more)

### Community 5 - "Dependency Stack and LoRA Tooling"
Cohesion: 0.22
Nodes (9): Dynamic LoRA Loading, LoRA (Low-Rank Adaptation), controlnet-aux, Python Dependency Set, fastapi, image_gen_aux (git), peft, transformers (+1 more)

### Community 6 - "Public API Surface and Deployment"
Cohesion: 0.29
Nodes (8): CreateImageEditRequest, CreateImageVariationRequest, Docker Deployment, FastFusion, POST /v1/images/edits, POST /v1/images/variations, Kubernetes Deployment Target, OpenAI API Compatibility

## Ambiguous Edges - Review These
- `edit_images()` → `edits_config`  [AMBIGUOUS]
  README.md · relation: references
- `get_image_description()` → `variations_config`  [AMBIGUOUS]
  README.md · relation: shares_data_with

## Knowledge Gaps
- **10 isolated node(s):** `GET /health`, `lora_settings Request Parameter`, `hf_model_id`, `CONFIG_PATH`, `FASTFUSION_LOGLEVEL` (+5 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `edit_images()` and `edits_config`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **What is the exact relationship between `get_image_description()` and `variations_config`?**
  _Edge tagged AMBIGUOUS (relation: shares_data_with) - confidence is low._
- **Why does `setup()` connect `Model Setup and VRAM Tuning` to `App Shell and Response Serving`, `Config Schema and Generation`, `Task Queue and Edit Endpoints`, `Dependency Stack and LoRA Tooling`?**
  _High betweenness centrality (0.204) - this node is a cross-community bridge._
- **Why does `Python Dependency Set` connect `Dependency Stack and LoRA Tooling` to `App Shell and Response Serving`, `Model Setup and VRAM Tuning`, `Task Queue and Edit Endpoints`, `Vision Prompting and Image I/O`?**
  _High betweenness centrality (0.113) - this node is a cross-community bridge._
- **Why does `get_image_description()` connect `Vision Prompting and Image I/O` to `App Shell and Response Serving`?**
  _High betweenness centrality (0.099) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `setup()` (e.g. with `clean_old_images()` and `Diffusers AutoPipelines`) actually correct?**
  _`setup()` has 10 INFERRED edges - model-reasoned connections that need verification._
- **What connects `GET /health`, `lora_settings Request Parameter`, `hf_model_id` to the rest of the system?**
  _10 weakly-connected nodes found - possible documentation gaps or missing edges._