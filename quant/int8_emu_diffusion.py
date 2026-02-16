from typing import Iterable, Callable, Any, List, Optional, Sequence
#from enum import StrEnum
import torch
from torch import nn, Tensor
from torch.nn import functional as F


import os
import numpy as np
import matplotlib.pyplot as plt


from functools import cache

#from lm_eval.utils import eval_logger
from typing import Literal, Optional

##====================================================================================================

def is_linear_like(m: nn.Module) -> bool:
    # Extend this tuple with project-specific classes (e.g., LoraLinear)
    return isinstance(m, nn.Linear) or type(m).__name__ in {"MyLinear"}


def _get_parent_by_path(root: nn.Module, path:str):
    """
    path like 'encoder.layers.3.mlp.fc1' -> (parent_module, 'fc1')
    """
    parts = path.split('.')
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

##====================================================================================================





       


##====================================================================================================

class QuantLinear_INT8(torch.nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias=False,
            weight_quant=True,
            act_quant=True,
            name=None,
    ):
        
        super().__init__(in_features, out_features, bias)
        self.name = name or "QuantLinear"
        self.weight_quant = bool(weight_quant)
        self.act_quant = bool(act_quant)

        self.no_quant     = (not self.weight_quant) and (not self.act_quant)
        self.weight_only  = (self.weight_quant) and (not self.act_quant)
        self.input_only   = (not self.weight_quant) and (self.act_quant)
        self.input_weight = (self.weight_quant) and (self.act_quant)

        # bookkeeping
        self._weights_already_quantized = False
        self._w_fp_snapshot: Optional[torch.Tensor] = None
        self._input_check_done = False
        self._weight_check_done = False

        self.weight_per_channel_dim = 0
        self.verify_weight_quant = True



        # global weight scale (default identity)
        self.register_buffer("weight_int8_scale", torch.tensor(1.0), persistent=True)

    #------- Printing Helpers ----------------
    def _mode_str(self):
        if self.no_quant: return "no_quant"
        if self.weight_only: return "weight_only"
        if self.input_only: return "input_only"
        return "input_weight"

    #---------------------------------------------

    def _print_input_diff(self, x_fp, x_q):
        # if not self.verify_input_quant:
        #     return
        if self._input_check_done and self.print_first_only:
            return
        with torch.no_grad():
            diff = (x_q - x_fp).abs()
            print(f"[INPUT-QCHECK] {self.name} (mode={self._mode_str()}): "
                  f"shape={tuple(x_fp.shape)} mean|Δ|={diff.mean().item():.4e} "
                  f"max|Δ|={diff.max().item():.4e}")
            print("  fp[:8]:", x_fp.flatten()[:8].detach().cpu())
            print("  q [:8]:", x_q.flatten()[:8].detach().cpu())
        self._input_check_done = True

    #--------------------------------------------------

    def _print_weight_diff(self, w_before, w_after):
        if not self.verify_weight_quant:
            return
        if self._weight_check_done and self.print_first_only:
            return
        if w_before is None or w_after is None:
            print(f"[WEIGHT-QCHECK] {self.name}: (missing snapshot/weight)")
            self._weight_check_done = True
            return
        with torch.no_grad():
            diff = (w_after - w_before).abs()
            print(f"[WEIGHT-QCHECK] {self.name} (mode={self._mode_str()}): "
                  f"shape={tuple(w_before.shape)} mean|Δ|={diff.mean().item():.4e} "
                  f"max|Δ|={diff.max().item():.4e}")
            print("  w_fp [:8]:", w_before.flatten()[:8].cpu())
            print("  w_q  [:8]:", w_after.flatten()[:8].cpu())
        self._weight_check_done = True

     #--------------------------------------------------

    def _linear_with_dtype_match(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        b = self.bias

        if w.dtype != x.dtype:
            w = w.to(dtype=x.dtype)
        if b is not None and b.dtype != x.dtype:
            b = b.to(dtype=x.dtype)

        return F.linear(x, w, b)


##===================================================================
##================== Weight Quantization Channel Wise ===============
##===================================================================

    def calibrate_weight_int8_scale(self, w: torch.Tensor):
        """
            Compute self.weight_int8_scale to match *your convention*:
            w_scaled = w * scale
            q = round(w_scaled) clipped to [-127,127]
            w_hat = q * (1/scale)

            For per-channel (dim=0): scale is [O] (one per output channel).
        """
        qmax = 127
        ch_dim = getattr(self, "weight_per_channel_dim", None)

        with torch.no_grad():
            if ch_dim is None:
                maxabs = w.detach().abs().max().clamp(min=1e-12) #scalar
                scale = (qmax / maxabs).to(torch.float32)
            else:
                # normalize ch_dim
                ch_dim = ch_dim if ch_dim >= 0 else w.ndim + ch_dim

                # For Linear [O, I] and per-output-channel => ch_dim=0
                # Reduce over all dims except ch_dim:
                reduce_dims = [d for d in range(w.ndim) if d != ch_dim]
                maxabs = w.detach().abs().amax(dim=reduce_dims).clamp(min=1e-12) #shape [C]
                scale = (qmax / maxabs).to(torch.float32) # shape [C]                
        
        # store to buffer device
        self.weight_int8_scale = scale.to(self.weight_int8_scale.device)

##===================================================================
    
    def _int8_quant_weight_channel(self, w: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        """
        INT8 fake quant: quantize to int8 then dequant back to float.
        Returns dequantized weights in out_dtype (bf16/fp16).
        """

        # print(f"Uses _int8_quant_weight_channel --------------------")

        int8_dtype = torch.int8
        qmax = 127
        qmin = -127

        scale = self.weight_int8_scale  # scalar or [C]
        ch_dim = getattr(self, "weight_per_channel_dim", None)

        # broadcast scale to w
        if ch_dim is None or scale.dim() == 0:
            scale_b = scale  # scalar
        else:
            ch_dim = ch_dim if ch_dim >= 0 else w.ndim + ch_dim
            if scale.numel() != w.size(ch_dim):
                raise ValueError(
                    f"weight_int8_scale has {scale.numel()} elems but w dim {ch_dim} has size {w.size(ch_dim)}"
                )
            shape = [1] * w.ndim
            shape[ch_dim] = scale.shape[0]
            scale_b = scale.view(*shape)

        scale_b = torch.clamp(scale_b, min=1e-8, max=1e8)

        # quant/dequant in fp32
        w_fp32 = w.to(torch.float32)
        w_scaled = w_fp32 * scale_b
        q = torch.round(w_scaled).clamp(qmin, qmax).to(int8_dtype)

        inv_scale_b = torch.nan_to_num(1.0 / scale_b, nan=0.0, posinf=0.0, neginf=0.0)
        w_deq = q.to(torch.float32) * inv_scale_b

        return w_deq.to(out_dtype)


##===================================================================
##================= Weight Quantization Global Scale ================
##===================================================================

    def _int8_quant_weight_global(self, w:torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        """
        Global INT8 quantization for weights using precomputed
        self.weight_int8_scale (scalar or per-channel).

        Interprets scale like: w_scaled = w * scale.
        Quant: q = round(w_scaled) clipped to int8 range
        Dequant: w_hat = q * (1/scale)
        """    
        # w_dtype = w.dtype
        int8_dtype = torch.int8
        iinfo = torch.iinfo(int8_dtype)
        int8_max = iinfo.max
        int8_min = iinfo.min

        ## Symmetric range for weights
        # qmin = int(int8_min)
        qmax = min (int(int8_max), 127) # 127
        qmin = -qmax                    # -127 

        # 1) get global / pre-calibrated scale 
        scale = self.weight_int8_scale  # scalar

        # 2) Broadcast to weight shape
        if self.weight_per_channel_dim is None or scale.dim() == 0:
            scale_b = scale

        else:
            ch_dim = (
            self.weight_per_channel_dim
            if self.weight_per_channel_dim >= 0 
            else w.ndim + self.weight_per_channel_dim
            )

            if scale.numel() != w.size(ch_dim):
                raise ValueError(
                    f"weight_int8_scale has {scale.numel()} elements but weight "
                    f"dimension {ch_dim} has size {w.size(ch_dim)}"         
                )
            
            shape = [1] * w.ndim
            shape[ch_dim] = scale.shape[0]
            scale_b = scale.view(*shape)

        scale_b = torch.clamp(scale_b, min=1e-8, max=1e8)

        # 3) scale, quantize to int8, dequant
        # Scale
        w_scaled = w.to(torch.float32) * scale_b

        # Quantize (round) + clip to symmetric int8 range
        q = torch.round(w_scaled).clamp(qmin, qmax).to(int8_dtype)

        # Dequant
        inv_scale_b = 1.0 / scale_b
        inv_scale_b = torch.nan_to_num(inv_scale_b, nan=0.0, posinf=0.0, neginf=0.0)

        w_deq = q.to(torch.float32) * inv_scale_b
        return w_deq.to(out_dtype)

##===================================================================
    
    def quantize_weight_int8(self):
        if not self.weight_quant or self.no_quant:
            #print(f"no_quant ----------------------")
            return

        if self._weights_already_quantized:
            #print(f"self._weights_already_quantized -------------")
            return

        #if self.no_quant:
        #    return

        with torch.no_grad():
            #print(f"use quant_int8_GSint8_GSint8_GS -------------")
            # snapshot FP weights for later diff
            # if self.verify_weight_quant and self._w_fp_snapshot is None:
            if self._w_fp_snapshot is None:
                self._w_fp_snapshot = self.weight.detach().clone().cpu()
            
            out_dtype = self.weight.dtype
            
            # q_w =self._int8_quant_weight_global(self.weight.data, out_dtype=out_dtype)

            # 1) calibrate scales (per-channel or global based on self.weight_per_channel_dim)
            self.calibrate_weight_int8_scale(self.weight)
            
            # 2) fake-quant dequantize back to the module's param dtype (bf16 ideally)
            q_w =self._int8_quant_weight_channel(self.weight.data, out_dtype=out_dtype)

            
            # 3) store dequantized weights (still float, fake-quant emulation)
            self.weight.copy_(q_w)

        self._weights_already_quantized = True

        # print diff immediately after quantizing
        if self.verify_weight_quant:
           self._print_weight_diff(self._w_fp_snapshot, self.weight.detach().cpu())  

##===================================================================

    def set_weight_bias(self, w, b=False):
        with torch.no_grad():
            assert w.shape == self.weight.shape, f"weight shape mismatch: {w.shape} vs {self.weight.shape}"
            self.weight.copy_(w)
            if self.bias is None and b is not None:
                self.bias = nn.Parameter(b.detach().clone())
            elif self.bias is not None and b is None:
                self.bias.zero_()
            elif self.bias is not None and b is not None:
                assert b.shape == self.bias.shape, f"bias shape mismatch: {b.shape} vs {self.bias.shape}"
                self.bias.copy_(b)

        self._w_fp_snapshot = self.weight.detach().clone().cpu()
        self._weights_already_quantized = False
        self._weight_check_done = False

##===================================================================
##===================================================================

    def forward(self, input):

        # ---------- Mode 1: no quant ----------
        if self.no_quant:
            # if self.verify_input_quant and (not self._input_check_done or not self.print_first_only):
            #     print(f"[MODE] {self.name}: mode=no_quant (no quantization performed)")
            #     self._input_check_done = True
            # return F.linear(input, self.weight, self.bias)
            return self._linear_with_dtype_match(input)

        # ---------- Mode 2: weight-only ----------
        if self.weight_only:

            self.quantize_weight_int8()

            # if self.verify_input_quant and (not self._input_check_done or not self.print_first_only):
            #     print(f"[MODE] {self.name}: mode=weight_only (inputs not quantized)")
            #     self._input_check_done = True

            return self._linear_with_dtype_match(input)

        # # PRE-INT8 DEBUG: check for NaN/Inf BEFORE any INT8 or custom quant -----------------
        # if self.verify_input_quant:
        #     if torch.isnan(input).any() or torch.isinf(input).any():
        #         print(f"[PRE-FP8] {self.name}: found NaN/Inf in input")

        #---------- Mode 3: input-only ----------
        if self.input_only:

            x_fp = input
            # x_q = self._int8_quant_act_global(x_fp, fp8_dtype=torch.float8_e5m2)
            # self._print_input_diff(x_fp, x_q)
            
            x_q = x_fp
            #print(f"Passing through input_only -----------------------------------------------")
            # return F.linear(x_q, self.weight, self.bias)
            return self._linear_with_dtype_match(x_q)

        # ---------- Mode 4: input + weight ----------

        ## For W=8bit (FP8)
        self.quantize_weight_int8()
        
        ## For A=8bit (INT8)
        x_fp = input
        # x_q = self._fp8_quant_act_global(x_fp, fp8_dtype=torch.float8_e5m2)
        # self._print_input_diff(x_fp, x_q)

        x_q=x_fp

        # return F.linear(x_q, self.weight, self.bias)   
        return self._linear_with_dtype_match(x_q)     

##====================================================================================================

def convert_linear_to_quant_linear_int8(
        model: nn.Module,
        target_layer: Optional[str] = None,
        mode: Literal["no_quant", "weight_only", "input_only", "input_weight"] = "weight_only",
        # quantize_now: bool = True,
) -> nn.Module:
    
    """
    mode:
      - "no_quant"      -> weight_quant=False, act_quant=False
      - "weight_only"   -> weight_quant=True,  act_quant=False
      - "input_only"    -> weight_quant=False, act_quant=True
      - "input_weight"  -> weight_quant=True,  act_quant=True
    """
    printed = False
    mode2flags = {
        "no_quant":     (False, False),
        "weight_only":  (True,  False),
        "input_only":   (False, True),
        "input_weight": (True,  True),
    }

    if mode not in mode2flags:
        raise ValueError(f"Unknown mode '{mode}'. Choose from {list(mode2flags.keys())}.")
    
    weight_quant_flag, act_quant_flag = mode2flags[mode]

    # Collect all candidate module paths first
    linear_paths: List[str] = []

    for name, mod in model.named_modules():
        if name == "":
            continue
        if is_linear_like(mod):
            if target_layer is None or name == target_layer:
                linear_paths.append(name)

    if not linear_paths:
        print(f"[INFO] No linear-like modules found. (Wrapped? Using F.linear?)")
        return model


    ##=============================== Main Pass ========================================

    for idx, full in enumerate(linear_paths):
        parent, child_name = _get_parent_by_path(model, full)
        old: nn.Module = getattr(parent, child_name)
        parent_cls_name = parent.__class__.__name__
        old_cls_name = old.__class__.__name__

        # Decide if this module *should* be quantized according to debug settings
        should_quantize = True

        # Capture pre-quant weights for debug print
        w_before = getattr(old, "weight", None)
        if isinstance(w_before, torch.Tensor):
            w_before = w_before.detach().clone().cpu()

        in_features = getattr(old, "in_features", None)
        out_features = getattr(old, "out_features", None)
        has_bias = getattr(old, "bias", None) is not None

        if in_features is None or out_features is None:
            print(f"[WARN] {full}: not a standard Linear spec; skipping quantization.")
            print(
                f"[LINEAR-SKIP:NOSTD] idx={idx} | path={full} | "
                f"parent={parent_cls_name} | child={child_name} | module={old_cls_name}"
            )
            continue

        # ---- Build QuantLinear with FP8 activation config ----
        new_module = QuantLinear_INT8(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
            weight_quant=weight_quant_flag,
            act_quant=act_quant_flag,
            name=full,
        )

        # Transfer parameters
        if hasattr(old, "weight") and isinstance(old.weight, torch.Tensor):
            new_module.set_weight_bias(w=old.weight, b=old.bias if has_bias else None)


        # Keep compatible buffers/extra fields (non-strict on purpose)
        try:
            new_module.load_state_dict(old.state_dict(), strict=False)
        except Exception as e:
            print(f"[WARN] {full}: load_state_dict(strict=False) failed: {e}")


        # Replace in parent
        new_module = new_module.to(device=old.weight.device, dtype=old.weight.dtype)
        setattr(parent, child_name, new_module)

        # LOG: this path is now QuantLinear
        print(
            f"[QUANT] idx={idx} | path={full} | "
            f"parent={parent_cls_name} | child={child_name} | module={new_module.__class__.__name__} "
            f"(mode={mode})"
        )

        # # Quantize weights now if requested and applicable
        # if quantize_now and weight_quant_flag and hasattr(new_module, "quantize_weight"):
        #     try:
        #         new_module.quantize_weight()
        #     except Exception as e:
        #         print(f"[WARN] {full}: quantize_weight() raised {e}")

        # # Summary print + quick weight delta check
        # if (target_layer is None or full == target_layer) and (not printed or not print_first_only):
        #     w_after = getattr(new_module, "weight", None)
        #     if isinstance(w_after, torch.Tensor):
        #         w_after = w_after.detach().clone().cpu()
        #     print(f"[INFO] Replaced: {full} -> QuantLinear(mode='{mode}', block_size={block_size}, "
        #           f"act_fmt={act_fp8_format}, act_pct={act_percentile}, per_ch={act_per_channel_dim})")
        #     if (w_before is not None) and (w_after is not None) and (w_before.shape == w_after.shape):
        #         diff = (w_after - w_before).abs()
        #         print(f" weight Δ (post-wrap): mean|Δ|={diff.mean():.6e} max|Δ|={diff.max():.6e} "
        #               f"shape={tuple(w_before.shape)}")
        #     printed = True        


    # ========================= SUMMARY PASS =========================
    print("\n[SUMMARY] Final status of linear-like paths:")
    for idx, full in enumerate(linear_paths):
        parent, child_name = _get_parent_by_path(model, full)
        mod = getattr(parent, child_name)
        parent_cls_name = parent.__class__.__name__
        mod_cls_name = mod.__class__.__name__

        kind = "QuantLinear" if mod_cls_name == "QuantLinear" else "Linear"
        print(
            f"[SUMMARY-{kind}] idx={idx} | path={full} | "
            f"parent={parent_cls_name} | child={child_name} | module={mod_cls_name}"
        )

    return model
