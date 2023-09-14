from diffusers import ( 
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)


ADAPTER_MAP = {
  "lineart": "TencentARC/t2i-adapter-lineart-sdxl-1.0",
  "canny": "TencentARC/t2i-adapter-canny-sdxl-1.0",
  "sketch": "TencentARC/t2i-adapter-sketch-sdxl-1.0",
  "openpose": "TencentARC/t2i-adapter-openpose-sdxl-1.0",
  "depth-midas": "TencentARC/t2i-adapter-depth-midas-sdxl-1.0",
}

ANNOTATOR_MAP = {
  "lineart": ("lllyasviel/Annotators", ["sk_model.pth", "sk_model2.pth"]),
  "sketch": ("lllyasviel/Annotators", ["table5_pidinet.pth"]),
  "openpose": ("lllyasviel/Annotators", ["body_pose_model.pth", "hand_pose_model.pth", "facenet.pth"]), 
  "depth-midas": ("valhalla/t2iadapter-aux-models", ["dpt_large_384.pt"]),
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