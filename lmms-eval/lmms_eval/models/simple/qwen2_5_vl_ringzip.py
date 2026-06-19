import base64
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
from ringzip import (
    CompressPlan,
    RingZipCompressor,
)

process_vision_info, _has_qwen_vl = optional_import("qwen_vl_utils", "process_vision_info")
if not _has_qwen_vl:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")

from transformers.cache_utils import DynamicCache


def _get_visual_model(model):
    return model.model.visual


def _patch_merger_compat(model):
    import inspect

    try:
        visual = _get_visual_model(model)
    except AttributeError:
        return

    if not hasattr(visual, 'merger'):
        return

    merger = visual.merger
    sig = inspect.signature(merger.forward)
    params = list(sig.parameters.keys())

    if 'grid_thw' not in params and 'kwargs' not in params:
        original_forward = merger.forward

        def _compat_forward(hidden_states, grid_thw=None, **kwargs):
            return original_forward(hidden_states, **kwargs)

        merger.forward = _compat_forward
        eval_logger.info(
            "🔧 Patched PatchMerger.forward for grid_thw compatibility "
            f"(original params: {params})"
        )


def _extract_visual_embeds(outputs):
    return outputs.pooler_output


class ViTFeatureExtractor:

    def _ensure_post_merger(
        self,
        features: torch.Tensor,
        h_tok: int,
        w_tok: int,
        visual_model=None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_vis = h_tok * w_tok
        D = features.shape[-1]
        n_tokens = features.shape[0] if features.ndim == 2 else features.numel() // D
        features = features.reshape(n_tokens, D)

        if n_tokens == num_vis * 4:
            if (
                visual_model is None
                or image_grid_thw is None
                or not hasattr(visual_model, "get_window_index")
            ):
                raise ValueError(
                    "_ensure_post_merger: pre-merger input "
                    f"(n_tokens={n_tokens} = num_vis*4={num_vis*4}) requires "
                    "visual_model + image_grid_thw + visual_model.get_window_index() "
                    "to reverse window-order to row-major. Missing: "
                    f"visual_model={visual_model is not None}, "
                    f"image_grid_thw={image_grid_thw is not None}, "
                    f"get_window_index={hasattr(visual_model, 'get_window_index') if visual_model else 'n/a'}. "
                    "Caller must pass these to avoid silent spatial misalignment."
                )

            features = features.reshape(num_vis, 4, D)
            window_index, _ = visual_model.get_window_index(image_grid_thw)
            reverse_indices = torch.argsort(window_index.to(features.device))
            features = features[reverse_indices]

            features = features.reshape(h_tok, w_tok, 2, 2, D)
            features = features.float().mean(dim=(2, 3)).to(features.dtype)
            features = features.reshape(num_vis, D)

        return features

    @torch.no_grad()
    def extract_for_compress(
        self,
        model,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        h_tok: int,
        w_tok: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        visual_model = _get_visual_model(model)
        raw_output = visual_model(pixel_values, grid_thw=image_grid_thw)
        features = _extract_visual_embeds(raw_output)
        features = self._ensure_post_merger(features, h_tok, w_tok)
        return features, features


def _find_get_rope_index(model):
    return model.model.get_rope_index


class RingZipViTSideMount:

    def __init__(self, compressor: RingZipCompressor):
        self.compressor = compressor

    @torch.no_grad()
    def prefill_with_compression(
        self, model, inputs, vis_s, vis_e, h_tok, w_tok,
        mrope_section, device,
        feature_extractor: ViTFeatureExtractor = None,
    ):
        original_seq_len = inputs["input_ids"].shape[1]
        num_vis = vis_e - vis_s

        if feature_extractor is None:
            feature_extractor = ViTFeatureExtractor()

        coherence_features, output_features = feature_extractor.extract_for_compress(
            model,
            inputs["pixel_values"],
            inputs["image_grid_thw"],
            h_tok, w_tok,
        )

        plan = self.compressor.plan_compression(
            coherence_features, h_tok, w_tok)
        compressed_vis_len = plan.num_output
        compressed_seq_len = vis_s + compressed_vis_len + (original_seq_len - vis_e)


        get_rope_index = _find_get_rope_index(model)
        if get_rope_index is None:
            raise RuntimeError("Cannot find get_rope_index on the model.")

        original_position_ids, original_rope_deltas = get_rope_index(
            input_ids=inputs["input_ids"],
            image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=inputs.get("video_grid_thw"),
            attention_mask=inputs.get("attention_mask"),
            **({
                "mm_token_type_ids": inputs.get("mm_token_type_ids")
            } if "mm_token_type_ids" in inputs else {}),
        )

        pos_ndim = original_position_ids.shape[0]
        vis_pos = original_position_ids[:, :, vis_s:vis_e]
        compressed_vis_pos = plan.apply_long(vis_pos, seq_dim=-1)

        old_max = vis_pos.amax(dim=-1, keepdim=True)
        new_max = compressed_vis_pos.amax(dim=-1, keepdim=True)
        shift = old_max - new_max
        text_after_shifted = original_position_ids[:, :, vis_e:] - shift

        compressed_position_ids = torch.cat([
            original_position_ids[:, :, :vis_s],
            compressed_vis_pos,
            text_after_shifted,
        ], dim=2)

        new_inputs = dict(inputs)

        input_ids = inputs["input_ids"]
        image_token_id = model.config.image_token_id
        compressed_pads = torch.full(
            (input_ids.shape[0], compressed_vis_len),
            image_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        new_inputs["input_ids"] = torch.cat([
            input_ids[:, :vis_s],
            compressed_pads,
            input_ids[:, vis_e:],
        ], dim=1)

        if "attention_mask" in inputs and inputs["attention_mask"] is not None:
            attn_mask = inputs["attention_mask"]
            new_inputs["attention_mask"] = torch.cat([
                attn_mask[:, :vis_s],
                torch.ones(
                    (attn_mask.shape[0], compressed_vis_len),
                    dtype=attn_mask.dtype, device=attn_mask.device),
                attn_mask[:, vis_e:],
            ], dim=1)

        new_inputs["position_ids"] = compressed_position_ids
        new_inputs["rope_deltas"] = original_rope_deltas


        embed_fn = model.get_input_embeddings()
        llm_hidden_dim = embed_fn.weight.shape[1]

        vis_3d = output_features.reshape(1, num_vis, -1)
        compressed_features = self.compressor.compress_hidden(vis_3d, plan)
        compressed_features = compressed_features.squeeze(0)

        if compressed_features.shape[-1] != llm_hidden_dim:
            eval_logger.warning(
                f"ViT output dim ({compressed_features.shape[-1]}) "
                f"!= LLM hidden dim ({llm_hidden_dim}), "
                f"falling back to get_image_features()")
            if hasattr(model, 'model') and hasattr(model.model, 'get_image_features'):
                feat_host = model.model
            elif hasattr(model, 'get_image_features'):
                feat_host = model
            else:
                raise RuntimeError("Cannot find get_image_features on the model.")

            vision_outputs = feat_host.get_image_features(
                inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True)
            pooler = vision_outputs.pooler_output
            if isinstance(pooler, (tuple, list)):
                vis_features_llm = torch.cat(pooler, dim=0)
            else:
                vis_features_llm = pooler
            vis_3d_llm = vis_features_llm.reshape(1, num_vis, -1)
            compressed_features = self.compressor.compress_hidden(
                vis_3d_llm, plan).squeeze(0)


        input_ids = new_inputs["input_ids"]
        inputs_embeds = embed_fn(input_ids)

        compressed_features = compressed_features.to(
            dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        inputs_embeds[:, vis_s:vis_s + compressed_vis_len, :] = compressed_features.unsqueeze(0)

        forward_inputs = {k: v for k, v in new_inputs.items()
                         if k not in ("input_ids", "pixel_values", "pixel_values_videos",
                                      "image_grid_thw", "video_grid_thw",
                                      "second_per_grid_ts", "mm_token_type_ids")}
        forward_inputs["inputs_embeds"] = inputs_embeds

        outputs = model(
            **forward_inputs,
            use_cache=True,
            return_dict=True,
        )

        past_kv = outputs.past_key_values
        rope_deltas = getattr(outputs, 'rope_deltas', None)
        if rope_deltas is None:
            rope_deltas = original_rope_deltas
        first_logits = outputs.logits[:, -1, :]

        if hasattr(past_kv, '_seen_tokens'):
            past_kv._seen_tokens = compressed_seq_len

        block_sizes = [len(g) for g in plan.groups]
        single_count = sum(1 for s in block_sizes if s == 1)
        pooled_count = sum(1 for s in block_sizes if s > 1)
        avg_pool_size = (
            sum(s for s in block_sizes if s > 1) / pooled_count
            if pooled_count > 0 else 0
        )

        stats = {
            "original_vis": num_vis,
            "compressed_vis": compressed_vis_len,
            "total_original": original_seq_len,
            "total_compressed": compressed_seq_len,
            "ratio": compressed_seq_len / original_seq_len if original_seq_len > 0 else 1.0,
            "mount": "vit_coherence",
            "plan_num_groups": len(plan.groups),
            "single_tokens": single_count,
            "pooled_blocks": pooled_count,
            "avg_pool_size": avg_pool_size,
            "plan": plan,
            "h_tok": h_tok,
            "w_tok": w_tok,
            "coherence_features": coherence_features.detach().cpu(),
        }

        return (first_logits, past_kv, compressed_seq_len, stats,
                rope_deltas, compressed_position_ids[:, :, -1:])


@register_model("qwen2_5_vl_ringzip")
class Qwen2_5_VLRingZip(lmms):

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,
        max_image_size: Optional[int] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        init_stride: int = 4,
        norm_temperature: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        valid_attn = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn:
            raise ValueError(f"attn_implementation must be one of {valid_attn}")

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        model_kwargs = {"torch_dtype": "bfloat16", "device_map": self.device_map}
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained, **model_kwargs).eval()
        _patch_merger_compat(self._model)
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames

        self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n") if reasoning_prompt else None
        self.processor = AutoProcessor.from_pretrained(
            pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self.compressor = RingZipCompressor(
            init_stride=init_stride,
            norm_temperature=norm_temperature,
        )
        self.feature_extractor = ViTFeatureExtractor()
        self.compression_mount = RingZipViTSideMount(
            compressor=self.compressor,
        )

        self._config = self.model.config
        self._max_length = 2048
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self._mrope_section = [16, 24, 24]

        if accelerator.num_processes > 1:
            assert (accelerator.distributed_type in
                    [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.MULTI_NPU])
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

        eval_logger.info(
            f"🌲 RingZip config: "
            f"init_stride={init_stride}, "
            f"norm_temperature={norm_temperature}"
        )

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests):
        raise NotImplementedError

    def flatten(self, input):
        return [j for i in input for j in i]

    def _locate_vision_tokens(self, input_ids, image_grid_thw):
        ids = input_ids[0]
        vs_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        ve_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")

        vis_start = (ids == vs_id).nonzero(as_tuple=True)[0][0].item() + 1
        vis_end = (ids == ve_id).nonzero(as_tuple=True)[0][0].item()
        text_start = vis_end + 1

        thw = image_grid_thw[0]
        h_tok = (thw[1] // 2).item()
        w_tok = (thw[2] // 2).item()
        return vis_start, vis_end, text_start, h_tok, w_tok

    @torch.no_grad()
    def _decode_with_compressed_kv(
        self, first_logits, past_kv,
        original_seq_len, compressed_seq_len, rope_deltas,
        max_new_tokens, temperature=0.0, do_sample=False,
        last_pos_id=None
    ):
        generated = []
        logits = first_logits

        delta = original_seq_len - compressed_seq_len
        if rope_deltas is not None:
            adj_rope = rope_deltas + delta
        else:
            adj_rope = torch.tensor([[delta]], device=self.device, dtype=torch.long)

        curr_pos = last_pos_id + 1 if last_pos_id is not None else None
        increment = getattr(self, "decode_increment_position", False)

        for step in range(max_new_tokens):
            next_tok = torch.argmax(logits, dim=-1)
            if next_tok.item() == self.tokenizer.eos_token_id:
                break
            generated.append(next_tok.item())

            outputs = self.model(
                input_ids=next_tok.unsqueeze(0),
                past_key_values=past_kv,
                rope_deltas=adj_rope,
                position_ids=curr_pos,
                use_cache=True,
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :]
            past_kv = outputs.past_key_values
            if increment and curr_pos is not None:
                curr_pos = curr_pos + 1

        return generated

    def _build_single_message(self, text_content, images,
                              min_pixels=None, max_pixels=None):
        processed_visuals = []
        for img in images:
            if isinstance(img, Image.Image):
                buf = BytesIO()
                img.convert("RGB").save(buf, format="JPEG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                processed_visuals.append({
                    "type": "image",
                    "image": f"data:image/jpeg;base64,{b64}",
                    "max_pixels": self.max_pixels if max_pixels is None else max_pixels,
                    "min_pixels": self.min_pixels if min_pixels is None else min_pixels,
                })
        message = [{"role": "system", "content": self.system_prompt}]
        message.append({
            "role": "user",
            "content": processed_visuals + [{"type": "text", "text": text_content}],
        })
        return message

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task, split = task[0], split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected until to be str or list, got {type(until)}")
            until = [t for t in until if t != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)
            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            batched_messages, pil_images = [], []
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")
                if self.reasoning_prompt:
                    text_content = context.strip() + self.reasoning_prompt
                    contexts[i] = text_content
                else:
                    text_content = context

                pil_image = None
                if visual_list[i] is not None and len(visual_list[i]) > 0:
                    if isinstance(visual_list[i][0], Image.Image):
                        pil_image = visual_list[i][0]
                pil_images.append(pil_image)
                batched_messages.append(
                    self._build_single_message(text_content, [pil_image]))

            default_gk = {
                "max_new_tokens": 32768, "temperature": 0.0,
                "top_p": None, "num_beams": 1,
            }
            cur_gk = {**default_gk, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id
            max_new_tokens = cur_gk["max_new_tokens"]
            temperature = cur_gk.get("temperature", 0.0)
            do_sample = cur_gk.get("do_sample", False)

            has_image = pil_images[0] is not None

            if has_image:
                texts = self.processor.apply_chat_template(
                    batched_messages, tokenize=False, add_generation_prompt=True)
                img_in, vid_in = process_vision_info([batched_messages[0]])
                inputs = self.processor(
                    text=texts, images=img_in, videos=vid_in,
                    padding=True, padding_side="left", return_tensors="pt",
                ).to(self.device)

                original_seq_len = inputs["input_ids"].shape[1]

                vs, ve, _, ht, wt = self._locate_vision_tokens(
                    inputs["input_ids"], inputs["image_grid_thw"])

                first_logits, compressed_kv, compressed_len, stats, rope_deltas, last_pos_id = \
                    self.compression_mount.prefill_with_compression(
                        model=self.model,
                        inputs=inputs,
                        vis_s=vs,
                        vis_e=ve,
                        h_tok=ht,
                        w_tok=wt,
                        mrope_section=self._mrope_section,
                        device=self.device,
                        feature_extractor=self.feature_extractor,
                    )

                gen_toks = self._decode_with_compressed_kv(
                    first_logits, compressed_kv,
                    original_seq_len, compressed_len, rope_deltas,
                    max_new_tokens, temperature, do_sample, last_pos_id)

                if gen_toks:
                    answers = self.processor.batch_decode(
                        [gen_toks], skip_special_tokens=True,
                        clean_up_tokenization_spaces=False)
                else:
                    answers = [""]

            else:
                texts = [self.processor.apply_chat_template(
                    batched_messages[0], tokenize=False,
                    add_generation_prompt=True)]
                img_in, vid_in = process_vision_info([batched_messages[0]])
                inputs = self.processor(
                    text=texts, images=img_in, videos=vid_in,
                    padding=True, padding_side="left", return_tensors="pt",
                ).to(self.device)

                cont = self.model.generate(
                    **inputs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=do_sample, temperature=temperature,
                    top_p=cur_gk["top_p"], num_beams=cur_gk["num_beams"],
                    max_new_tokens=max_new_tokens, use_cache=self.use_cache,
                )
                trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, cont)]
                answers = self.processor.batch_decode(
                    trimmed, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False)

            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                clean_ans = parse_reasoning_model_answer(ans)
                res.append(clean_ans)
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), clean_ans)
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()

        return res

    def generate_until_multi_round(self, requests):
        raise NotImplementedError("TODO: Implement multi-round generation")