<div align="center">

# RingZip: Coherence-Guided 2D Recursive Token Compression for Ultra-High-Resolution Remote Sensing Image Understanding with MLLMs

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-RingZip-blue?logo=github&logoColor=white)](https://github.com/AIRS101/RingZip)
[![Model](https://img.shields.io/badge/Model-RingZip--PMA-blue?logo=huggingface&logoColor=FFD21E)](https://huggingface.co/VVVVH/RingZip-PMA)

</div>

---
## Introduction

RingZip is a coherence-guided 2D recursive token compression framework for efficient UHR RS image interpretation with MLLMs. It performs region-level token compression by introducing a region-wise weighted coherence metric to estimate the mergeability of spatially adjacent regions, jointly modeling local feature consistency and the global impact of merging. Guided by this metric, RingZip performs a coarse-to-fine split-and-merge search over the 2D token grid, compressing large homogeneous regions while preserving low coherence details and spatial topology. 

![Image](images/method.jpg)

## Installation

RingZip ships with a trimmed copy of LMMs-Eval.

```bash
pip install torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/AIRS101/RingZip.git
cd RingZip

pip install -e ".[qwen]"

cd lmms-eval && pip install -e . && cd ..

pip install flash-attn==2.8.3 --no-build-isolation
```

## Quick Start

Run RingZip on Qwen2.5-VL:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --num_processes 2 \
    --main_process_port 29505 \
    -m lmms_eval \
    --model qwen2_5_vl_ringzip \
    --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=flash_attention_2,init_stride=16,norm_temperature=0.5,max_pixels=12845056" \
    --tasks xlrs-lite \
    --batch_size 1 --log_samples \
    --output_path ./outputs/qwen2_5_vl_ringzip
```

For the trainable variant RingZip-PMA:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --num_processes 2 \
    --main_process_port 29505 \
    -m lmms_eval \
    --model qwen2_5_vl_ringzip_pma \
    --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,pma_checkpoint=./pma_checkpoint,attn_implementation=flash_attention_2,init_stride=16,norm_temperature=0.5,max_pixels=12845056" \
    --tasks xlrs-lite \
    --batch_size 1 --log_samples \
    --output_path ./outputs/qwen2_5_vl_ringzip_pma
```

## Citation

## Acknowledgement

RingZip is evaluated with the [LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)
framework and built on top of the host MLLMs
[Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL),
[InternVL](https://github.com/OpenGVLab/InternVL),
[LongVA](https://github.com/EvolvingLMMs-Lab/LongVA), and
[GeoLLaVA-8K](https://github.com/MiliLab/GeoLLaVA-8K).

## License

Released under the [Apache License 2.0](LICENSE).
