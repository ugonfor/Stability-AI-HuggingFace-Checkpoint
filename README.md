# Stability-HuggingFace-Checkpoint
checkpoint dictionary key change

You can use the research checkpoint of [stable-diffusion](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)(such as sd-v1-4.ckpt, ... ) at Huggingface StableDiffusionPipeline.

### commandline: 
```
python scripts/txt2img.py --prompt "a photograph of an astronaut riding a horse" --plms 
```

### output:

before.png:

![before](./contents/before.png)

after.png:

![after](./contents/after.png)
