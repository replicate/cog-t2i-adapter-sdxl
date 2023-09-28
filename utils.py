from diffusers import ( 
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
import os
import subprocess
import tarfile

SDXL_URL_MAP= {
    "sdxl-1.0": "https://weights.replicate.delivery/default/sdxl/sdxl-1.0.tar",
    "scheduler": "https://weights.replicate.delivery/default/T2I-Adapter-SDXL/scheduler.tar",
    "vae-fp16-fix": "https://weights.replicate.delivery/default/T2I-Adapter-SDXL/sdxl-vae-fp16-fix.tar",
}

ADAPTER_URL_MAP = {
    "openpose": "https://weights.replicate.delivery/default/T2I-Adapter-SDXL/t2i-adapter-openpose-sdxl-1.0.tar",
    "lineart": "https://weights.replicate.delivery/default/T2I-Adapter-SDXL/t2i-adapter-lineart-sdxl-1.0.tar",
    "canny": "https://weights.replicate.delivery/default/T2I-Adapter-SDXL/t2i-adapter-canny-sdxl-1.0.tar",
    "sketch": "https://weights.replicate.delivery/default/T2I-Adapter-SDXL/t2i-adapter-sketch-sdxl-1.0.tar",
    "depth-midas": "https://weights.replicate.delivery/default/T2I-Adapter-SDXL/t2i-adapter-depth-midas-sdxl-1.0.tar",
}

ANNOTATOR_URL_MAP = {
    "openpose": "https://weights.replicate.delivery/default/T2I-Adapter-SDXL/t2i-openpose-annotator.tar",
    "lineart": "https://weights.replicate.delivery/default/T2I-Adapter-SDXL/t2i-lineart-annotator.tar",
    "sketch": "https://weights.replicate.delivery/default/T2I-Adapter-SDXL/t2i-sketch-annotator.tar",
    "depth-midas": "https://weights.replicate.delivery/default/T2I-Adapter-SDXL/t2i-depth-midas-annotator.tar",
}

class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)

SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KarrasDPM": KarrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
    "LMSDiscrete": LMSDiscreteScheduler,
}

def download_and_extract(url: str, dest: str):
    try:
        if os.path.exists("/src/tmp.tar"):
            subprocess.check_call(["rm", "/src/tmp.tar"])
        subprocess.check_call(["pget", url, "/src/tmp.tar"])
    except subprocess.CalledProcessError as e:
        print(e.output)
        raise e
    tar = tarfile.open("/src/tmp.tar")
    tar.extractall(dest)
    tar.close()
    os.remove("/src/tmp.tar")

def install_t2i_adapter_cache(
        model_type:str,
        model_base_cache:str,
        model_scheduler_cache:str,
        model_vae_cache:str,
        model_adapter_cache:str,
        model_annotator_cache:str
):
    # Base Model
    if not os.path.exists(model_base_cache):
        os.makedirs(model_base_cache)
        download_and_extract(SDXL_URL_MAP["sdxl-1.0"], model_base_cache)
    # Scheduler
    if not os.path.exists(model_scheduler_cache):
        os.makedirs(model_scheduler_cache)
        download_and_extract(SDXL_URL_MAP["scheduler"], model_scheduler_cache)
    # VAE
    if not os.path.exists(model_vae_cache):
        os.makedirs(model_vae_cache)
        download_and_extract(SDXL_URL_MAP["vae-fp16-fix"], model_vae_cache)
    # Adapter
    if not os.path.exists(model_adapter_cache):
        os.makedirs(model_adapter_cache)
        download_and_extract(ADAPTER_URL_MAP[model_type], model_adapter_cache)
    # Annotator
    if not os.path.exists(model_annotator_cache) and model_type != "canny":
        os.makedirs(model_annotator_cache)
        download_and_extract(ANNOTATOR_URL_MAP[model_type], model_annotator_cache)
