import os
import torch
import numpy as np
from PIL import Image

from diffusers import T2IAdapter, AutoencoderKL, StableDiffusionXLAdapterPipeline
from diffusers import ( 
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from huggingface_hub import hf_hub_download, snapshot_download
from cog import BasePredictor, Input, Path
from utils import ADAPTER_MAP, ANNOTATOR_MAP, SCHEDULERS


MODEL_TYPE = "openpose"
MODEL_CACHE = "./t2i-adapter-cache"
MODEL_ADAPTER_CACHE = "./t2i-adapter-cache/adapter"
MODEL_VAE_CACHE = "./t2i-adapter-cache/sdxl-vae-fp16-fix"
MODEL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        if not os.path.exists(MODEL_VAE_CACHE):
            snapshot_download(repo_id="madebyollin/sdxl-vae-fp16-fix", local_dir=MODEL_VAE_CACHE)

        if not os.path.exists(MODEL_ADAPTER_CACHE):
            snapshot_download(repo_id=adapter_map[MODEL_TYPE], local_dir=MODEL_ADAPTER_CACHE)

        repo_id, filenames = ANNOTATOR_MAP[MODEL_TYPE]
        for filename in filenames:
            hf_hub_download(repo_id=repo_id, filename=filemame, local_dir=MODEL_ANNOTATOR_CACHE)

        # Load data annotator / preprocessor
        if MODEL_TYPE == "lineart":
            from controlnet_aux.lineart import LineartDetector
            self.annotator = LineartDetector.from_pretrained(MODEL_ANNOTATOR_CACHE).to("cuda")

        elif MODEL_TYPE == "canny":
            from controlnet_aux.canny import CannyDetector
            self.annotator = CannyDetector()

        elif MODEL_TYPE == "sketch":
            from controlnet_aux.pidi import PidiNetDetector
            self.annotator = PidiNetDetector.from_pretrained(MODEL_ANNOTATOR_CACHE).to("cuda")

        elif MODEL_TYPE == "pose":
            from controlnet_aux import OpenposeDetector
            self.annotator = OpenposeDetector.from_pretrained(MODEL_ANNOTATOR_CACHE)

        elif MODEL_TYPE == "midas_depth":
            from controlnet_aux.midas import MidasDetector
            self.annotator = MidasDetector.from_pretrained(
                MODEL_ANNOTATOR_CACHE, filename="dpt_large_384.pt", model_type="dpt_large"
            ).to("cuda")

        # Load pipeline
        vae = AutoencoderKL.from_pretrained(
            MODEL_VAE_CACHE, torch_dtype=torch.float16, local_files_only=True
        )

        adapter = T2IAdapter.from_pretrained(
            MODEL_ADAPTER_CACHE, torch_dtype=torch.float16, varient="fp16", local_files_only=True
        ).to("cuda")

        self.pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            MODEL_BASE, 
            vae=vae, 
            adapter=adapter, 
            torch_dtype=torch.float16, 
            variant="fp16", 
            use_safetensors=True,
            cache_dir=MODEL_CACHE
        ).to("cuda")

        self.pipe.enable_xformers_memory_efficient_attention()

    def predict(
        self,
        image: Path = Input(description="Input image"),
        adapter_name: str = Input(
            description="Type of adapter to apply to input image",
            choices=["canny", "sketch", "lineart", "pose", "midas-depth"],
            default="canny",
        ),
        prompt: str = Input(
            description="Input prompt",
            default="A car with flying wings",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face",
        ),
        num_inference_steps: int = Input(
            description="Number of diffusion steps", ge=0, le=100, default=30
        ),
        adapter_conditioning_scale: float = Input(description="Factor to scale image by", ge=0, le= , default=0.8),,
        adapter_conditioning_factor: float = Input(description="Factor to scale image by", ge=0, le=1, default=1),
        guidance_scale: float = Input(default=7.5),
        scheduler: str = Input(
            description="Which scheduler to use",
            choices=SCHEDULERS.keys(),
            default="K_EULER_ANCESTRAL"
        )
    ) -> List[Path]:
        """Run a single prediction on the model"""

        if scheduler != "K_EULER_ANCESTRAL":
            self.pipe.scheduler = SCHEDULERS["K_EULER_ANCESTRAL"].from(self.pipe.scheduler.config)

        if MODEL_TYPE == "lineart":
            image = self.annotator(image, detect_resolution=384, image_resolution=1024)
        elif MODEL_TYPE == "canny":
            image = self.annotator(image, detect_resolution=384, image_resolution=1024)
        elif MODEL_TYPE == "sketch":
            image = self.annotator(image, detect_resolution=1024, image_resolution=1024, apply_filter=True)
        elif MODEL_TYPE == "pose":
            image = self.annotator(image, detect_resolution=512, image_resolution=1024)
            image = np.array(image)[:, :, ::-1]           
            image = Image.fromarray(np.uint8(image)) 
        elif MODEL_TYPE == "midas_depth":
            image = self.annotator(image, detect_resolution=512, image_resolution=1024)

        outputs = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, 
            adapter_conditioning_scale=adapter_conditioning_scale,
            adapter_conditioning_factor=adapter_conditioning_factor
        )
        output_paths = []
        for i, sample in enumerate(outputs.images):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
