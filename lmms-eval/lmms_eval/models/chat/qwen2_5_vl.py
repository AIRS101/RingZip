import time
from typing import Dict, List

import torch
from loguru import logger as eval_logger
from tqdm import tqdm

try:
    import decord
except ImportError:
    decord = None

from lmms_eval import utils
from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL as Qwen2_5_VLSimple
from lmms_eval.protocol import ChatMessages

process_vision_info, _has_qwen_vl = optional_import("qwen_vl_utils", "process_vision_info")
if not _has_qwen_vl:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen2_5_vl_chat")
class Qwen2_5_VL(Qwen2_5_VLSimple):
    is_simple = False

    def __init__(self, *args, collect_timing: bool = True, **kwargs) -> None:
        """
        Args:
            collect_timing: 累计每样本 generate() 的 wall / CPU / CUDA 时延
                + 输入 token 数, 在 generate_until 末尾汇总输出。代价是每样本
                两次 cuda.synchronize() (~1 ms 量级)。关掉就完全无开销。
                与 coherence 模型的 collect_timing 输出格式一致, 方便 baseline 对比。
        """
        super().__init__(*args, **kwargs)
        self.collect_timing = bool(collect_timing)
        self._timing_stats: List[Dict[str, float]] = []

    def generate_until(self, requests: List[Instance]) -> List[GenerationResult]:
        res = []

        # A dummy collate here to sort by doc id
        def _collate(x):
            return x[0], x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator(
            [reg.args for reg in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        total_elapsed_time = 0
        total_tokens = 0
        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            chat_messages = [doc_to_messages[idx](self.task_dict[task][split][ids]) for idx, (ids, task, split) in enumerate(zip(doc_id, task, split))]
            chat_messages: List[ChatMessages] = [ChatMessages(**{"messages": message}) for message in chat_messages]
            visuals = []
            videos = []
            for messages in chat_messages:
                visual, video, _ = messages.extract_media()
                visuals.append(visual)
                videos.append(video)
            visuals = self.flatten(visuals)
            videos = self.flatten(videos)
            gen_kwargs = all_gen_kwargs[0]

            # Apply chat template
            video_kwargs = {
                "max_pixels": self.max_pixels,
                "min_pixels": self.min_pixels,
            }
            if self.fps is not None:
                video_kwargs["fps"] = self.fps
            else:
                # Probe videos to get frame count and set nframes = min(max_num_frames, total_frames)
                # This avoids the error when video has fewer frames than max_num_frames
                if videos and decord is not None:
                    try:
                        video_path = videos[0]  # Assume batch size 1 for videos
                        vr = decord.VideoReader(video_path)
                        video_total_frames = len(vr)
                        nframes = min(self.max_num_frames, video_total_frames)
                        # qwen_vl_utils requires nframes to be a multiple of 2 (FRAME_FACTOR)
                        # and rounds using round_by_factor, so we need to floor to even number
                        # to avoid rounding up past total_frames
                        nframes = (nframes // 2) * 2  # Floor to nearest even number
                        nframes = max(2, nframes)  # At least 2 frames
                        video_kwargs["nframes"] = nframes
                    except Exception as e:
                        eval_logger.warning(f"Failed to probe video {videos[0]}: {e}, using default nframes")
                        video_kwargs["nframes"] = self.max_num_frames
                else:
                    video_kwargs["nframes"] = self.max_num_frames
            batched_messages = [chat_message.to_hf_messages(video_kwargs=video_kwargs) for chat_message in chat_messages]
            # protocol.to_hf_messages 只把 max/min_pixels 传给了 video 元素, image 元素
            # 不带这两个键, process_vision_info -> fetch_image 会回退到 qwen_vl_utils
            # 内置默认 MAX_PIXELS=16384*28*28, 把大图静默缩到 <=16384 token (与本模型
            # 的 max_pixels 设置无关). 这里逐 image 注入, 让原始分辨率真正生效.
            for _msgs in batched_messages:
                for _m in _msgs:
                    for _c in _m.get("content", []):
                        if isinstance(_c, dict) and _c.get("type") == "image":
                            _c["max_pixels"] = self.max_pixels
                            _c["min_pixels"] = self.min_pixels
            texts = self.processor.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(batched_messages)
            padding_side = "left" if self.batch_size > 1 else "right"
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                padding_side=padding_side,
                return_tensors="pt",
            )

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": None,
                "num_beams": 1,
            }
            # Update with provided kwargs
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None
                current_gen_kwargs["top_k"] = None

            # ---- 推理时延计时: 整个 generate() 调用 ----
            # baseline 不分 prefill / decode (HF 内置 generate() 一次完成),
            # 只测一个 total。三个时间维度跟 coherence 模型保持一致, 方便对比。
            if self.collect_timing:
                # 输入侧 token 统计 (image_pad 数 = visual token 数)
                _input_ids = inputs["input_ids"]
                _orig_seq_len = _input_ids.shape[1]
                try:
                    _img_tok_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
                    _orig_vis = (_input_ids == _img_tok_id).sum().item() if _img_tok_id is not None and _img_tok_id >= 0 else 0
                except Exception:
                    _orig_vis = 0

                torch.cuda.synchronize()
                _gen_wall_start = time.perf_counter()
                _gen_cpu_start = time.process_time()
                _gen_cuda_start = torch.cuda.Event(enable_timing=True)
                _gen_cuda_end = torch.cuda.Event(enable_timing=True)
                _gen_cuda_start.record()

            start_time = time.time()
            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                top_k=current_gen_kwargs.get("top_k", None),
                use_cache=self.use_cache,
            )
            end_time = time.time()

            if self.collect_timing:
                _gen_cuda_end.record()
                torch.cuda.synchronize()
                _gen_wall_ms = (time.perf_counter() - _gen_wall_start) * 1000.0
                _gen_cpu_ms = (time.process_time() - _gen_cpu_start) * 1000.0
                _gen_cuda_ms = _gen_cuda_start.elapsed_time(_gen_cuda_end)

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            # Calculate timing metrics for batch
            total_elapsed_time += end_time - start_time
            _batch_decode_tokens = sum(len(ids) for ids in generated_ids_trimmed)
            total_tokens += _batch_decode_tokens

            # 每样本累加 (baseline 没有压缩, vis_ratio = seq_ratio = 1.0,
            # 与 coherence 模型 summary 字段保持一致, 方便论文里直接对照表)
            if self.collect_timing:
                _per_sample = max(1, len(generated_ids_trimmed))
                self._timing_stats.append({
                    # baseline: prefill/decode 不分, generate_wall = total
                    # 为了字段对齐, prefill 全归 0, decode 当作 total。
                    "prefill_wall_ms": 0.0,
                    "prefill_cpu_ms": 0.0,
                    "prefill_cuda_ms": 0.0,
                    "decode_wall_ms": _gen_wall_ms / _per_sample,
                    "decode_cpu_ms": _gen_cpu_ms / _per_sample,
                    "decode_cuda_ms": _gen_cuda_ms / _per_sample,
                    "generate_wall_ms": _gen_wall_ms / _per_sample,
                    "generate_cpu_ms": _gen_cpu_ms / _per_sample,
                    "generate_cuda_ms": _gen_cuda_ms / _per_sample,
                    "num_decode_tokens": _batch_decode_tokens / _per_sample,
                    "original_vis": _orig_vis / _per_sample,
                    "compressed_vis": _orig_vis / _per_sample,  # baseline 无压缩
                    "original_seq_len": _orig_seq_len,
                    "compressed_seq_len": _orig_seq_len,
                    "vis_ratio": 1.0,
                    "seq_ratio": 1.0,
                })

            for i, (ans, context) in enumerate(zip(answers, texts)):
                res.append(GenerationResult(text=ans, token_counts=TokenCounts(output_tokens=len(generated_ids_trimmed[i]))))
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)

                eval_logger.debug(f"Question: {context}")
                eval_logger.debug(f"Model Response: {ans}")
            # reorder this group of results back to original unsorted form
            pbar.update(1)
        res = re_ords.get_original(res)

        # Calculate average speed
        avg_speed = total_tokens / total_elapsed_time if total_elapsed_time > 0 else 0
        # Log metrics
        metric_dict = {
            "total_gen_tokens": total_tokens,
            "total_elapsed_time": total_elapsed_time,
            "avg_speed": avg_speed,
            "additional_metrics": {
                "rank": self.rank,
            },
        }
        log_metrics(**metric_dict)

        # 评测全部样本跑完后输出推理时延 + token 统计 (baseline)
        self._print_timing_summary()

        pbar.close()
        return res

    def _print_timing_summary(self) -> None:
        """汇总 baseline 每样本的平均推理时延 (wall / CPU / CUDA) + token 统计。

        和 qwen2_5_vl_coherence.Qwen2_5_VLCoherence._print_timing_summary
        输出格式保持一致, 方便论文里把两边的数字直接对照。

        三个时间维度:
            * wall_ms : 墙钟时间, 包含 CPU 计算 + GPU kernel + 同步等待
            * cpu_ms  : 当前进程 CPU 线程的活跃时间 (process_time)
            * cuda_ms : GPU kernel 净执行时间 (cuda.Event)

        注意: baseline 没有 prefill/decode 拆分 (HF model.generate 一次完成),
        这里把 generate() 的整体时间归到 "Decode" 行, "Prefill" 行全 0。
        """
        if not self._timing_stats:
            return

        n = len(self._timing_stats)

        def _avg(key: str) -> float:
            return sum(s[key] for s in self._timing_stats) / n

        avg_wall = _avg("generate_wall_ms")
        avg_cpu = _avg("generate_cpu_ms")
        avg_cuda = _avg("generate_cuda_ms")
        avg_orig_vis = _avg("original_vis")
        avg_orig_seq = _avg("original_seq_len")
        avg_dec_tok = _avg("num_decode_tokens")
        decode_per_tok = (avg_wall / avg_dec_tok if avg_dec_tok > 0 else 0.0)

        bar = "=" * 76
        eval_logger.info(bar)
        eval_logger.info(
            f"  Inference Latency & Token Summary  [BASELINE]  (N = {n})"
        )
        eval_logger.info(bar)

        # ---- Tokens ----
        eval_logger.info("  [Tokens] (no compression — baseline)")
        eval_logger.info(
            f"    Visual tokens     : {avg_orig_vis:>10.1f}   "
            f"(image_pad count per sample)"
        )
        eval_logger.info(
            f"    Total seq tokens  : {avg_orig_seq:>10.1f}"
        )
        eval_logger.info(f"    Decode tokens/sample: {avg_dec_tok:.2f}")

        # ---- Latency ----
        eval_logger.info("")
        eval_logger.info("  [Latency per sample, ms]")
        eval_logger.info(
            f"    {'':<14}{'wall':>12}{'CPU':>12}{'CUDA':>12}"
        )
        eval_logger.info(
            f"    Generate      "
            f"{avg_wall:>12.2f}{avg_cpu:>12.2f}{avg_cuda:>12.2f}"
        )
        eval_logger.info(
            f"    Wall/token    {decode_per_tok:>12.3f}   "
            f"(wall ms per generated token)"
        )
        eval_logger.info(bar)
