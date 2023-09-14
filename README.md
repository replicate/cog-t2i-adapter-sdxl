# Cog-T2I-Adapter-SDXL

[![Replicate demo and cloud API](https://replicate.com/stability-ai/sdxl/badge)](https://replicate.com/stability-ai/sdxl)

This is an implementation of TencentARC and the diffuser team's [T2I-Adapter-SDXL](https://github.com/TencentARC/T2I-Adapter) as a [Cog](https://github.com/replicate/cog) model.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own fork of T2I-Adapter-SDXL to [Replicate](https://replicate.com).

## Basic Usage

To run a prediction:

```bash
cog predict -i prompt="Ice dragon roar, 4k photo" -i adapter_name="lineart"
```

```bash
cog run -p 5000 python -m cog.server.http
```

## References
```
@article{mou2023t2i,
  title={T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models},
  author={Mou, Chong and Wang, Xintao and Xie, Liangbin and Wu, Yanze and Zhang, Jian and Qi, Zhongang and Shan, Ying and Qie, Xiaohu},
  journal={arXiv preprint arXiv:2302.08453},
  year={2023}
}
```