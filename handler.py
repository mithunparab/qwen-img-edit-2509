import runpod
import torch
import numpy as np
from PIL import Image
import base64
import io
import os
import math

from diffusers import FlowMatchEulerDiscreteScheduler
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from optimization import optimize_pipeline_
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

pipe = None
MODELS_DIR = "/home/user/app/models"

def load_model():
    """Loads the QwenImageEditPlusPipeline with optimizations."""
    global pipe
    if pipe is not None:
        return pipe
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for inference.")
    
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Available VRAM: {vram_gb:.1f}GB")
    
    enable_compile = os.environ.get("ENABLE_COMPILE", "false").lower() == "true"
    
    print(f"Loading model from cache directory: {MODELS_DIR}")
    print(f"Torch compile: {'enabled' if enable_compile else 'disabled (set ENABLE_COMPILE=true to enable)'}")
    
    scheduler_config = {
        "base_image_seq_len": 256, 
        "base_shift": math.log(3), 
        "invert_sigmas": False,
        "max_image_seq_len": 8192, 
        "max_shift": math.log(3), 
        "num_train_timesteps": 1000,
        "shift": 1.0, 
        "shift_terminal": None, 
        "stochastic_sampling": False,
        "time_shift_type": "exponential", 
        "use_beta_sigmas": False, 
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False, 
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    
    try:
        print("Loading QwenImageEditPlusPipeline...")
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2509",
            scheduler=scheduler,
            torch_dtype=dtype,
            cache_dir=MODELS_DIR
        ).to(device)
        
        pipe.transformer.__class__ = QwenImageTransformer2DModel
        pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
        
        print("Loading LoRA weights...")
        pipe.load_lora_weights(
            "lightx2v/Qwen-Image-Lightning",
            weight_name="Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors",
            cache_dir=MODELS_DIR
        )
        pipe.fuse_lora()
        print("LoRA weights fused successfully.")
        
        if enable_compile:
            print("Applying pipeline optimizations (torch.compile)...")
            try:
                dummy_images = [Image.new("RGB", (1024, 1024)), Image.new("RGB", (1024, 1024))]
                optimize_pipeline_(pipe, image=dummy_images, prompt="a cat")
                print("Pipeline optimized and ready.")
            except Exception as e:
                print(f"Warning: Could not compile model (non-fatal): {e}")
                print("Continuing without compilation...")
        else:
            print("Skipping torch.compile (disabled by default for stability)")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    return pipe

def base64_to_pil(base64_string):
    """Decode base64 string to PIL Image"""
    image_bytes = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def pil_to_base64(pil_image):
    """Encode PIL Image to base64 string"""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def handler(job):
    """
    RunPod serverless handler function.
    
    Expected input format:
    {
        "input": {
            "images": ["base64_string1", "base64_string2", ...],
            "prompt": "make it beautiful",
            "seed": 42,
            "true_guidance_scale": 1.0,
            "num_inference_steps": 8,
            "num_outputs": 1,
            "height": None,
            "width": None
        }
    }
    """
    global pipe
    
    if pipe is None:
        print("First request - loading model...")
        load_model()
    
    job_input = job.get('input', {})
    
    images_b64 = job_input.get('images', [])
    if not isinstance(images_b64, list):
        return {"error": "'images' must be a list of base64-encoded strings."}
    
    if not images_b64 and 'image' in job_input:
        images_b64 = [job_input['image']]
    
    prompt = job_input.get('prompt', 'make it beautiful')
    seed = job_input.get('seed', None)
    true_guidance_scale = float(job_input.get('true_guidance_scale', 1.0))
    num_inference_steps = int(job_input.get('num_inference_steps', 8))
    num_outputs = int(job_input.get('num_outputs', 1))
    height = job_input.get('height', None)
    width = job_input.get('width', None)
    
    if seed is None:
        seed = int(np.random.randint(0, np.iinfo(np.int32).max))
    
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    try:
        input_images = [base64_to_pil(img_b64) for img_b64 in images_b64]
    except Exception as e:
        return {"error": f"Failed to decode input images: {str(e)}"}
    
    print(f"Processing {len(input_images)} image(s) | Prompt: '{prompt}' | Seed: {seed} | Steps: {num_inference_steps}")
    
    try:
        output_images = pipe(
            image=input_images if input_images else None,
            prompt=prompt,
            negative_prompt=" ",
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            generator=generator,
            true_cfg_scale=true_guidance_scale,
            num_images_per_prompt=num_outputs,
        ).images
        
        output_b64_list = [pil_to_base64(img) for img in output_images]
        
        result = {
            "images": output_b64_list,
            "seed": seed,
            "version": "2.0"
        }
        
        if len(output_b64_list) == 1:
            result["image"] = output_b64_list[0]
        
        return result
        
    except Exception as e:
        print(f"Inference error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    print("Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})