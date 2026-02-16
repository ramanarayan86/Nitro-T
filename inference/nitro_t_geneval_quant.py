"""
Nitro-T GenEval Integration Script
Generates images using AMD Nitro-T model for GenEval benchmark evaluation

Ex:
For DiT:
python nitro_t_geneval.py geneval/prompts/evaluation_metadata.jsonl --outdir outputs/nitro-t-0.6B --model amd/Nitro-T-0.6B |&tee log_Nitro-T-0.6B_reproduce.txt

For MMDiT:
python nitro_t_geneval.py geneval/prompts/evaluation_metadata.jsonl --resolution 1024 --outdir outputs/nitro-t-0.6B --model amd/Nitro-T-0.6B |&tee log_Nitro-T-0.6B_reproduce.txt
"""
from __future__ import annotations

import torch
from torch import nn
from diffusers import DiffusionPipeline
from transformers import AutoModelForCausalLM
import json
import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import argparse

from copy import deepcopy
from typing import Optional, Sequence


from quant.int8_emu_diffusion import convert_linear_to_quant_linear_int8

##=================================================================================================


def setup_model(device="cuda:0", dtype=torch.bfloat16, model_name="amd/Nitro-T-0.6B"):
    """
    Initialize the Nitro-T model pipeline
    
    Args:
        device: Device to run the model on
        dtype: Data type for model weights
        model_name: HuggingFace model name
    
    Returns:
        Configured pipeline
    """
    torch.set_grad_enabled(False)
    
    print(f"Loading text encoder...")
    text_encoder = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B", 
        torch_dtype=dtype
    )
    
    print(f"Loading Nitro-T pipeline from {model_name}...")
    pipe = DiffusionPipeline.from_pretrained(
        model_name,
        text_encoder=text_encoder,
        torch_dtype=dtype, 
        trust_remote_code=True,
    )
    pipe.to(device)
    
    print("Model loaded successfully!")
    return pipe


##---------------------------------------------------------------------------------------------



def quantize_pipe_int8(
    pipe,
    mode: str = "weight_only",
    backbone_attrs: Sequence[str] = ("transformer", "unet", "dit", "diffusion_model", "model"),
    inplace: bool = False,
    device: Optional[torch.device] = None,
    verbose: bool = True,
):
    """
    Wrap the diffusion backbone Linear layers with QuantLinear_INT8 (fake quant) and
    return a quantized pipeline.

    - pipe: DiffusionPipeline (or compatible)
    - mode: "no_quant" | "weight_only" | "input_only" | "input_weight"
    - inplace: if False, returns a deep-copied pipeline (safer). If True, modifies input pipe.
    - device: optional device to move the returned pipe to (e.g., torch.device("cuda:0"))
    - verbose: prints backbone attr and summary info

    Returns: pipe_q
    """

    pipe_q = pipe if inplace else deepcopy(pipe)

    # -----------------------------
    # 1) Locate diffusion backbone
    # -----------------------------
    backbone_attr = None
    for attr in backbone_attrs:
        if hasattr(pipe_q, attr):
            backbone_attr = attr
            break

    if backbone_attr is None:
        keys = list(getattr(pipe_q, "components", {}).keys())
        raise RuntimeError(
            f"No known diffusion backbone attr found on pipe. "
            f"Tried {list(backbone_attrs)}. pipe.components keys={keys}"
        )

    backbone: nn.Module = getattr(pipe_q, backbone_attr).eval()
    if verbose:
        print(f"[PIPE-INT8] Backbone attr: pipe.{backbone_attr} ({backbone.__class__.__name__})")

    # ----------------------------------------------------------
    # 2) Wrap Linear layers with QuantLinear_INT8 (fake quant)
    # ----------------------------------------------------------
    backbone_q = convert_linear_to_quant_linear_int8(
        backbone,
        target_layer=None,
        mode=mode,  # e.g., "weight_only" for fake-quant weights
    )

    # ----------------------------------------
    # 3) Put quantized backbone back into pipe
    # ----------------------------------------
    setattr(pipe_q, backbone_attr, backbone_q)

    # ----------------------------------------
    # 4) Ensure everything is on device/dtype
    # ----------------------------------------
    if device is not None:
        pipe_q.to(device)

    return pipe_q

##---------------------------------------------------------------------------------------------


def load_geneval_prompts(metadata_path):
    """
    Load GenEval prompts from metadata JSONL file
    
    Args:
        metadata_path: Path to evaluation_metadata.jsonl
    
    Returns:
        List of prompt dictionaries
    """
    prompts = []
    with open(metadata_path, 'r') as f:
        for line in f:
            prompts.append(json.loads(line.strip()))
    return prompts


def generate_images(pipe, prompts, output_dir, resolution=512, num_samples=4, 
                   num_inference_steps=20, guidance_scale=4.0, seed=42):
    """
    Generate images for GenEval prompts using Nitro-T
    
    Args:
        pipe: Diffusion pipeline
        prompts: List of prompt dictionaries from GenEval
        output_dir: Output directory for generated images
        resolution: Image resolution (default: 512)
        num_samples: Number of samples per prompt (default: 4)
        num_inference_steps: Number of inference steps (default: 20)
        guidance_scale: Guidance scale for generation (default: 4.0)
        seed: Random seed for reproducibility
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    device = pipe.device
    
    for idx, prompt_data in enumerate(tqdm(prompts, desc="Generating images")):
        # Create directory for this prompt
        prompt_dir = output_path / f"{idx:05d}"
        prompt_dir.mkdir(exist_ok=True)
        
        samples_dir = prompt_dir / "samples"
        samples_dir.mkdir(exist_ok=True)
        
        # Save metadata for this prompt
        with open(prompt_dir / "metadata.jsonl", 'w') as f:
            json.dump(prompt_data, f)
            f.write('\n')
        
        prompt_text = prompt_data['prompt']
        
        # Generate multiple samples
        sample_images = []
        for sample_idx in range(num_samples):
            # Use different seed for each sample
            current_seed = seed + idx * num_samples + sample_idx
            generator = torch.Generator(device=device).manual_seed(current_seed)
            
            # Generate image
            try:
                image = pipe(
                    prompt=prompt_text,
                    height=resolution,
                    width=resolution,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator
                ).images[0]
                
                # Save individual sample
                sample_path = samples_dir / f"{sample_idx:04d}.png"
                image.save(sample_path)
                sample_images.append(image)
                
            except Exception as e:
                print(f"Error generating sample {sample_idx} for prompt {idx}: {e}")
                continue
        
        # Create grid image (optional, for visualization)
        if sample_images:
            try:
                grid = create_image_grid(sample_images)
                grid.save(prompt_dir / "grid.png")
            except Exception as e:
                print(f"Error creating grid for prompt {idx}: {e}")
    
    print(f"Image generation complete! Images saved to {output_dir}")


def create_image_grid(images, rows=2, cols=2):
    """
    Create a grid of images
    
    Args:
        images: List of PIL Images
        rows: Number of rows in grid
        cols: Number of columns in grid
    
    Returns:
        PIL Image containing the grid
    """
    if not images:
        return None
    
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    
    for i, img in enumerate(images[:rows * cols]):
        grid.paste(img, box=((i % cols) * w, (i // cols) * h))
    
    return grid


def main():
    parser = argparse.ArgumentParser(description='Generate images with Nitro-T for GenEval benchmark')
    parser.add_argument('metadata_path', type=str, 
                       help='Path to GenEval evaluation_metadata.jsonl file')
    parser.add_argument('--outdir', type=str, required=True,
                       help='Output directory for generated images')
    parser.add_argument('--model', type=str, default='amd/Nitro-T-0.6B',
                       help='Model name (default: amd/Nitro-T-0.6B)')
    parser.add_argument('--resolution', type=int, default=512,
                       help='Image resolution (default: 512)')
    parser.add_argument('--num-samples', type=int, default=4,
                       help='Number of samples per prompt (default: 4)')
    parser.add_argument('--num-inference-steps', type=int, default=20,
                       help='Number of inference steps (default: 20)')
    parser.add_argument('--guidance-scale', type=float, default=4.0,
                       help='Guidance scale (default: 4.0)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use (default: cuda:0)')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                       choices=['float16', 'bfloat16', 'float32'],
                       help='Data type (default: bfloat16)')
    
    args = parser.parse_args()
    
    # Set dtype
    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32
    }
    dtype = dtype_map[args.dtype]

    # model_q = convert_linear_to_quant_linear_int8(
    #         args.model,
    #         mode="weight_only",
    # )

    # Load model
    pipe = setup_model(device=args.device, dtype=dtype, model_name=args.model)

    pipe_q = quantize_pipe_int8(
        pipe,
        mode="weight_only",
        inplace=False,         # keep original pipe unchanged
    )
         
    # Load prompts
    print(f"Loading prompts from {args.metadata_path}...")
    prompts = load_geneval_prompts(args.metadata_path)
    print(f"Loaded {len(prompts)} prompts")
    
    # Generate images
    generate_images(
        pipe=pipe_q,
        prompts=prompts,
        output_dir=args.outdir,
        resolution=args.resolution,
        num_samples=args.num_samples,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed
    )
    
    print("Done! You can now run GenEval evaluation on the generated images.")


if __name__ == "__main__":
    main()