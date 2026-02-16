import torch
from diffusers import DiffusionPipeline
from transformers import AutoModelForCausalLM

torch.set_grad_enabled(False)

device = torch.device('cuda:0')
dtype = torch.bfloat16
resolution = 512
MODEL_NAME = "amd/Nitro-T-0.6B"

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

image.save("output1.png")

# Safety check: ensure it's a PIL image-like object
#print("Image type:", type(image))

#image.save(str(out_path))
#print("Saved OK:", out_path, "exists:", out_path.exists(), "size:", out_path.stat().st_size)
