import torch
from diffusers import DiffusionPipeline
from transformers import AutoModelForCausalLM

from quant.int8_emu_diffusion import convert_linear_to_quant_linear_int8

torch.set_grad_enabled(False)

device = torch.device('cuda:0')
dtype = torch.bfloat16
resolution = 1024
#MODEL_NAME = "models/Nitro-T-0.6B"
MODEL_NAME = "models/Nitro-T-1.2B"

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

## Pipe base line --------------------------

# pipe.to(device)

# image = pipe(
#     prompt="The image is a close-up portrait of a scientist in a modern laboratory. He has short, neatly styled black hair and wears thin, stylish eyeglasses. The lighting is soft and warm, highlighting his facial features against a backdrop of lab equipment and glowing screens.",
#     height=resolution, width=resolution,
#     num_inference_steps=20,
#     guidance_scale=4.0,
# ).images[0]

##--------------------------------------------

# Find backbone
for attr in ["transformer", "unet", "dit", "diffusion_model", "model"]:
    if hasattr(pipe, attr):
        backbone_attr = attr
        break
else:
    print("pipe.components:", list(pipe.components.keys()))
    raise RuntimeError("No known diffusion backbone attr found")

backbone = getattr(pipe, backbone_attr).eval()

print(f"Print the Backbone Model {backbone_attr}: {backbone}")

# image.save("output_local_quant.png")

# Safety check: ensure it's a PIL image-like object
#print("Image type:", type(image))

#image.save(str(out_path))
#print("Saved OK:", out_path, "exists:", out_path.exists(), "size:", out_path.stat().st_size)

# --- quantize it ---
backbone_q = convert_linear_to_quant_linear_int8(
    backbone,
    # target_layer = args.target_layer,
    mode="weight_only", 
    # mode="no_quant",
    # quantize_now = False,
    
)

# --- put quantized backbone back into pipeline ---
setattr(pipe, backbone_attr, backbone_q)

# Move to device AFTER replacement (or do .to(device) again)
pipe.to(device)

image = pipe(
    prompt="The image is a close-up portrait of a scientist in a modern laboratory. He has short, neatly styled black hair and wears thin, stylish eyeglasses. The lighting is soft and warm, highlighting his facial features against a backdrop of lab equipment and glowing screens.",
    height=resolution,
    width=resolution,
    num_inference_steps=20,
    guidance_scale=4.0,
).images[0]

image.save("output_W_int8_channel_12B.png")

## Sanity check that the pipe is actually using the quantized backbone ---------------
print("Using backbone attr:", backbone_attr)
print("Backbone type:", type(getattr(pipe, backbone_attr)))

b = getattr(pipe, backbone_attr)

# show first few modules/types
for i, (n, m) in enumerate(b.named_modules()):
    if i >= 40: break
    if "Linear" in type(m).__name__ or "Quant" in type(m).__name__:
        print(n, "->", type(m))

