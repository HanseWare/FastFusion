# ~~Lit~~FastFusion
openAI API compatible hosting for Huggingface Diffusers Autopipeline supported models
uses FastAPI and Huggingface Diffuser library

Direct successor to LitFusion that was given up due to LitServe turning out to be utter garbage.

Ment to be deployed e.g. on a Kubernetes cluster.

## Features
- [x] openAI API compatible
- [x] all models supported via Huggingface Diffusers Autopipeline

## Usage
```bash
docker run -d -p 8000:8000 hanseware/fastfusion
```

## Limitations
- [ ] currently num gpus not manually adjustable, should take all available

## License
[MIT](https://choosealicense.com/licenses/mit/)