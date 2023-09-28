import os
from typing import List
from cog import BasePredictor, Input, Path
import torch
import numpy as np
from PIL import Image
from diffusers import (
    T2IAdapter, 
    AutoencoderKL, 
    StableDiffusionXLAdapterPipeline, 
    EulerAncestralDiscreteScheduler
)
from utils import SCHEDULERS, install_t2i_adapter_cache


os.environ["HF_HOME"] = os.environ["HUGGINGFACE_HUB_CACHE"] = "/src/hf-cache"

# Available options: "openpose", "lineart", "canny", "sketch", "depth-midas"
MODEL_TYPE = "depth-midas" 

MODEL_BASE_CACHE = "/src/hf-cache/sdxl-1.0"
MODEL_ADAPTER_CACHE = f"/src/hf-cache/t2-adapter-{MODEL_TYPE}-sdxl-1.0"
MODEL_VAE_CACHE = "/src/hf-cache/sdxl-vae-fp16-fix"
MODEL_SCHEDULER_CACHE = "/src/hf-cache/scheduler"
MODEL_ANNOTATOR_CACHE = f"/src/hf-cache/annotator/{MODEL_TYPE}"

install_t2i_adapter_cache(
    model_type=MODEL_TYPE,
    model_base_cache=MODEL_BASE_CACHE,
    model_scheduler_cache=MODEL_SCHEDULER_CACHE,
    model_vae_cache=MODEL_VAE_CACHE,
    model_adapter_cache=MODEL_ADAPTER_CACHE,
    model_annotator_cache=MODEL_ANNOTATOR_CACHE,
)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        
        # Load the annotator
        if MODEL_TYPE == "openpose":
            from controlnet_aux.open_pose import OpenposeDetector
            self.annotator = OpenposeDetector.from_pretrained(MODEL_ANNOTATOR_CACHE).to("cuda")
        
        elif MODEL_TYPE == "lineart":
            from controlnet_aux.lineart import LineartDetector
            self.annotator = LineartDetector.from_pretrained(MODEL_ANNOTATOR_CACHE).to("cuda")

        elif MODEL_TYPE == "canny":
            from controlnet_aux.canny import CannyDetector
            self.annotator = CannyDetector()

        elif MODEL_TYPE == "sketch":
            from controlnet_aux.pidi import PidiNetDetector
            self.annotator = PidiNetDetector.from_pretrained(MODEL_ANNOTATOR_CACHE).to("cuda")

        elif MODEL_TYPE == "depth-midas":
            from controlnet_aux.midas import MidasDetector
            self.annotator = MidasDetector.from_pretrained(
                MODEL_ANNOTATOR_CACHE, filename="dpt_large_384.pt", model_type="dpt_large"
            ).to("cuda")

        # Load the pipeline
        vae = AutoencoderKL.from_pretrained(
            MODEL_VAE_CACHE, torch_dtype=torch.float16, local_files_only=True
        )

        adapter = T2IAdapter.from_pretrained(
            MODEL_ADAPTER_CACHE, torch_dtype=torch.float16, varient="fp16", local_files_only=True
        ).to("cuda")

        euler_a = EulerAncestralDiscreteScheduler.from_pretrained(MODEL_SCHEDULER_CACHE)

        self.pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            MODEL_BASE_CACHE, 
            vae=vae, 
            adapter=adapter, 
            scheduler=euler_a,
            torch_dtype=torch.float16, 
            variant="fp16",
        ).to("cuda")
        
        self.pipe.enable_xformers_memory_efficient_attention()

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(
            description="Input prompt",
            default="A photo of a room, 4k photo, highly detailed",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default= "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
        ),
        num_inference_steps: int = Input(
            description="Number of diffusion steps", ge=0, le=100, default=30
        ),
        adapter_conditioning_scale: float = Input(
            description="Conditioning scale", ge=0, le=5.0, default=1.0
        ),
        adapter_conditioning_factor: float = Input(
            description="Factor to scale image by", ge=0, le=1.0, default=1.0
        ),
        guidance_scale: float = Input(
            description="Guidance scale to match the prompt", ge=0, le=10.0, default=7.5
        ),
        num_samples: int = Input(
            description="Number of outputs to generate", ge=1, le=4, default=1
        ),
        scheduler: str = Input(
            description="Which scheduler to use",
            choices=SCHEDULERS.keys(),
            default="K_EULER_ANCESTRAL",
        )
    ) -> List[Path]:
        
        """Run a single prediction on the model"""

        if scheduler != "K_EULER_ANCESTRAL":
            self.pipe.scheduler = SCHEDULERS["K_EULER_ANCESTRAL"].from_config(self.pipe.scheduler.config)

        image = Image.open(image).convert("RGB")

        if MODEL_TYPE == "lineart":
            image = self.annotator(image, detect_resolution=384, image_resolution=1024)
        elif MODEL_TYPE == "canny":
            image = self.annotator(image, detect_resolution=384, image_resolution=1024)
        elif MODEL_TYPE == "sketch":
            image = self.annotator(image, detect_resolution=1024, image_resolution=1024, apply_filter=True)
        elif MODEL_TYPE == "openpose":
            image = self.annotator(image, detect_resolution=512, image_resolution=1024)
            image = np.array(image)[:, :, ::-1]           
            image = Image.fromarray(np.uint8(image)) 
        elif MODEL_TYPE == "depth-midas":
            image = self.annotator(image, detect_resolution=512, image_resolution=1024)

        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, 
            adapter_conditioning_scale=adapter_conditioning_scale,
            adapter_conditioning_factor=adapter_conditioning_factor,
            num_images_per_prompt=num_samples
        )

        outputs_paths = []
        for i, output_image in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            output_image.save(output_path)
            outputs_paths.append(Path(output_path))

        return outputs_paths