# ~~Lit~~FastFusion
openAI API compatible hosting for Huggingface Diffusers supported models
uses FastAPI and Huggingface Diffuser library

Direct successor to LitFusion that was given up due to LitServe turning out to be utter garbage.

## Features
- [x] openAI API compatible
- [x] all Huggingface Diffusers supported models

## Usage
```bash
docker run -d -p 8000:8000 hanseware/fastfusion
```