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

CUSTOM_OPTION_VALUE = "__custom__"

CAMERA_OPTIONS = [
    {"cn": "镜头方向左回转45度", "en": "Rotate camera 45° left"},
    {"cn": "镜头向右回转45度", "en": "Rotate camera 45° right"},
    {"cn": "镜头方向左回转90度", "en": "Rotate camera 90° left"},
    {"cn": "镜头向右回转90度", "en": "Rotate camera 90° right"},
    {"cn": "将镜头转为俯视", "en": "Switch to top-down view"},
    {"cn": "将镜头转为仰视", "en": "Switch to low-angle view"},
    {"cn": "将镜头转为特写镜头", "en": "Switch to close-up lens"},
    {"cn": "将镜头转为中近景镜头", "en": "Switch to medium close-up lens"},
    {"cn": "将镜头转为拉远镜头", "en": "Switch to zoom out lens"},
]

CAMERA_CN_MAP = {item["en"]: item["cn"] for item in CAMERA_OPTIONS}

def load_model():
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
        
        print("Loading and fusing existing Lightning LoRA weights...")
        pipe.load_lora_weights(
            "lightx2v/Qwen-Image-Lightning",
            weight_name="Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors",
            cache_dir=MODELS_DIR
        )
        pipe.fuse_lora()

        print("Loading and fusing Multi-Angle LoRA weights...")
        pipe.load_lora_weights(
            "dx8152/Qwen-Edit-2509-Multiple-angles",
            weight_name="镜头转换.safetensors",
            cache_dir=MODELS_DIR
        )
        pipe.fuse_lora()
        print("All LoRA weights fused successfully.")
        
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
    image_bytes = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def pil_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def handler(job):
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
    
    extra_prompt_input = job_input.get('prompt', 'make it beautiful') 
    
    camera_work_option = job_input.get('camera_work_option', None)
    custom_camera_prompt = job_input.get('custom_camera_prompt', None)

    # Initialize negative_prompt to a space string as used by the original library's pipeline default
    negative_prompt = job_input.get('negative_prompt', ' ') 

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

    is_single_image = len(input_images) == 1
    final_prompt = extra_prompt_input

    if is_single_image and camera_work_option:
        base_cn = None
        
        if camera_work_option == CUSTOM_OPTION_VALUE:
            base_cn = (custom_camera_prompt or "").strip()
        elif camera_work_option in CAMERA_CN_MAP:
            base_cn = CAMERA_CN_MAP[camera_work_option]
        
        if base_cn:
            final_prompt = f"{base_cn} {extra_prompt_input}".strip()
            print(f"Using Camera Work Prompt (Translated): '{base_cn}' + Extra: '{extra_prompt_input}'")
        else:
            print(f"Camera option '{camera_work_option}' not recognized/valid. Using default prompt.")
    else:
        if not is_single_image and camera_work_option:
            print("Multi-image detected. Bypassing camera work prompt translation.")
        
    print(f"Final Prompt for pipe: '{final_prompt}' | Negative Prompt: '{negative_prompt}' | Seed: {seed} | Steps: {num_inference_steps}")
    
    try:
        output_images = pipe(
            image=input_images if input_images else None,
            prompt=final_prompt,
            negative_prompt=negative_prompt,
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
            "version": "2.1"
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