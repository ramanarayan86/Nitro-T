import torch
from diffusers import DiffusionPipeline
from transformers import AutoModelForCausalLM

torch.set_grad_enabled(False)

device = torch.device('cuda:0')
dtype = torch.bfloat16
resolution = 512 #1024
MODEL_NAME = "models/Nitro-T-0.6B-Quant"
# MODEL_NAME = "models/Nitro-T-1.2B"

# from pathlib import Path
# MODEL_NAME = str(Path("models/Nitro-T-0.6B").resolve())
# print("Using MODEL_NAME:", MODEL_NAME, "exists:", Path(MODEL_NAME).exists())

text_encoder = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", torch_dtype=dtype)
pipe = DiffusionPipeline.from_pretrained(
    MODEL_NAME,
    text_encoder=text_encoder,
    torch_dtype=dtype, 
    trust_remote_code=True,
)
pipe.to(device)

image = pipe(
    prompt="The image is a close-up portrait of a scientist in a modern laboratory. He has short, neatly styled black hair and wears thin, stylish eyeglasses. The lighting is soft and warm, highlighting his facial features against a backdrop of lab equipment and glowing screens.",
    height=resolution, width=resolution,
    num_inference_steps=20,
    guidance_scale=4.0,
).images[0]

image.save("output_local_quant.png")

# Safety check: ensure it's a PIL image-like object
#print("Image type:", type(image))

#image.save(str(out_path))
#print("Saved OK:", out_path, "exists:", out_path.exists(), "size:", out_path.stat().st_size)
