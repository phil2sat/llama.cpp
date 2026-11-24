from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING
import re

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("MaxwellMoeForCausalLM")
@ModelBase.example("maxwell-model/Maxwell-V1-Flash")
class MaxwellMoeModel(TextModel):
    model_arch = gguf.MODEL_ARCH.MAXWELL_MOE

    _experts: list[dict[str, Tensor]] | None = None
    _hc_scale: dict[str, dict[str, Tensor]] | None = None

    supports_mtp_export = True
    mtp_only: bool = False
    no_mtp: bool = False
    _original_block_count: int | None = None
    _opt_num_mtp_layers: int = 0
    _mtp_proj_generators: dict[int, dict[str, Callable[[], Tensor]]] = {}
    _mtp_source_names: dict[int, set[str]] = {}

    # Maxwell checkpoints use a dedicated MTP namespace. Keep its accepted
    # tensor inventory explicit so an unsupported checkpoint revision cannot
    # silently produce a partially loadable draft model.
    _MTP_ROOT_TENSORS = {
        "enorm.weight": "model.layers.{bid}.enorm.weight",
        "hnorm.weight": "model.layers.{bid}.hnorm.weight",
        "final_layernorm.weight": "model.norm.weight",
        "hc_head.hc_head_fn": "model.hc_head.hc_head_fn",
        "hc_head.hc_head_base": "model.hc_head.hc_head_base",
        "hc_head.hc_head_scale": "model.hc_head.hc_head_scale",
    }
    _MTP_DECODER_TENSOR_RE = re.compile(
        r"^(?:"
        r"input_layernorm\.weight|post_attention_layernorm\.weight|"
        r"self_attn\.(?:[qkvo]_proj\.weight|[qk]_layernorm\.weight)|"
        r"self_attn_hyper_connection\.(?:alpha_(?:pre|post|res)|bias|mapping_proj\.weight)|"
        r"mlp_hyper_connection\.(?:alpha_(?:pre|post|res)|bias|mapping_proj\.weight)|"
        r"block_sparse_moe\.(?:gate\.weight|expert_bias|shared_expert\.w[123]\.weight|"
        r"experts\.\d+\.w[123]\.weight)"
        r")$"
    )

    def index_tensors(self, remote_hf_model_id: str | None = None):
        if self.mtp_only and self.no_mtp:
            raise ValueError("MaxwellMoe: --mtp and --no-mtp cannot be combined")
        type(self)._original_block_count = int(self.hparams["num_hidden_layers"])
        type(self)._opt_num_mtp_layers = 0
        type(self)._mtp_proj_generators = {}
        type(self)._mtp_source_names = {}
        tensors = super().index_tensors(remote_hf_model_id=remote_hf_model_id)
        if self.mtp_only:
            if type(self)._opt_num_mtp_layers == 0:
                raise ValueError("MaxwellMoe: --mtp requested but no mtp_layers tensors were found")
            if type(self)._mtp_proj_generators:
                raise ValueError("MaxwellMoe: MTP e_proj/h_proj pair is incomplete")

            # Refuse to emit a draft GGUF if any required tensor is absent.
            # The runtime requires a complete decoder block and the complete
            # per-layer projection/head set; a partial export is not loadable.
            decoder_names = {
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.q_layernorm.weight",
                "self_attn.k_layernorm.weight",
                *(f"{module}.{name}" for module in ("self_attn_hyper_connection", "mlp_hyper_connection")
                  for name in ("alpha_pre", "alpha_post", "alpha_res", "bias", "mapping_proj.weight")),
                "block_sparse_moe.gate.weight",
                "block_sparse_moe.expert_bias",
                "block_sparse_moe.shared_expert.w1.weight",
                "block_sparse_moe.shared_expert.w2.weight",
                "block_sparse_moe.shared_expert.w3.weight",
                *(f"block_sparse_moe.experts.{expert}.w{projection}.weight"
                  for expert in range(int(self.hparams["num_experts"])) for projection in (1, 2, 3)),
            }
            root_names = {
                "e_proj.weight", "h_proj.weight", "enorm.weight", "hnorm.weight",
                "final_layernorm.weight", "hc_head.hc_head_fn",
                "hc_head.hc_head_base", "hc_head.hc_head_scale",
            }
            for mtp_idx in range(type(self)._opt_num_mtp_layers):
                expected = {f"decoder_layer.{name}" for name in decoder_names} | root_names
                actual = type(self)._mtp_source_names.get(mtp_idx, set())
                missing = sorted(expected - actual)
                unexpected = sorted(actual - expected)
                if missing or unexpected:
                    raise ValueError(
                        f"MaxwellMoe: incomplete MTP layer {mtp_idx}; "
                        f"missing={missing}, unexpected={unexpected}"
                    )
        return tensors

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hparams = self.hparams
        if self.mtp_only:
            n_mtp = int(hparams.get("mtp_num_layers", type(self)._opt_num_mtp_layers))
            if n_mtp != type(self)._opt_num_mtp_layers:
                raise ValueError(f"MaxwellMoe: config declares {n_mtp} MTP layers but checkpoint contains {type(self)._opt_num_mtp_layers}")
            self.block_count = int(hparams["num_hidden_layers"]) + n_mtp
            self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

        if (act := hparams.get("hidden_act", "silu")) != "silu":
            raise ValueError(f"MaxwellMoe: unsupported hidden_act {act!r}")
        if hparams.get("rope_scaling") is not None:
            raise ValueError("MaxwellMoe: rope_scaling is not supported")
        if not hparams.get("qk_layernorm", False):
            raise ValueError("MaxwellMoe: only qk_layernorm=true is supported")
        if not hparams.get("enable_hyper_connections", False):
            raise ValueError("MaxwellMoe: only enable_hyper_connections=true is supported")
        if (n_hc := hparams.get("num_residual_streams")) != 4:
            raise ValueError(f"MaxwellMoe: num_residual_streams={n_hc!r}, only 4 is supported")
        if (score := hparams.get("router_score_function", "softmax")) != "sigmoid":
            raise ValueError(f"MaxwellMoe: unsupported router_score_function {score!r}")
        if not hparams.get("moe_router_enable_expert_bias", False):
            raise ValueError("MaxwellMoe: only moe_router_enable_expert_bias=true is supported")
        if hparams.get("use_shared_expert_gate", False):
            raise ValueError("MaxwellMoe: use_shared_expert_gate=true is not supported")
        if not hparams.get("shared_expert_intermediate_size"):
            raise ValueError("MaxwellMoe: a shared expert is required")

        # the reference makes every layer in mlp_only_layers dense, llama.cpp expects them to lead
        dense = sorted(hparams.get("mlp_only_layers") or [])
        if dense != list(range(len(dense))):
            raise ValueError(f"MaxwellMoe: mlp_only_layers {dense} is not a leading block")
        self.n_dense_lead = len(dense)

        # the backbone attention is block-sparse GQA with one indexer query head per KV group;
        # the block layout is stored for the sparse path, the dense path ignores it
        if (variant := hparams.get("attention_variant", "block_sparse_gqa")) != "block_sparse_gqa":
            raise ValueError(f"MaxwellMoe: unsupported attention_variant {variant!r}")
        if hparams["block_sparse_gqa_indexer_n_heads"] != hparams["num_key_value_heads"]:
            raise ValueError("MaxwellMoe: expected one indexer head per KV group")
        if any(f != 1 for f in hparams.get("block_sparse_gqa_layer_freq", [])):
            raise ValueError("MaxwellMoe: block_sparse_gqa_layer_freq must select every layer")
        block = hparams["block_sparse_gqa_block_size"]
        for key in ("block_sparse_gqa_prefix_window_size", "block_sparse_gqa_local_window_size"):
            if hparams[key] % block != 0:
                raise ValueError(f"MaxwellMoe: {key} is not a multiple of the block size {block}")

    def set_vocab(self):
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        # tokenizer_config.json sets add_bos_token, but the tokenizer has no post-processor that adds <s>,
        # so neither transformers nor the chat template ever put one in front of the prompt
        special_vocab.add_special_token["bos"] = False
        special_vocab.add_to_gguf(self.gguf_writer)

        # eos_token_id is </s>, the chat turn ends with <|maxwell_end|>: stop on it too
        tokenizer_config = self.dir_model / "tokenizer_config.json"
        if tokenizer_config.is_file():
            import json
            with open(tokenizer_config, encoding="utf-8") as f:
                added = json.load(f).get("added_tokens_decoder", {})
            for tid, tok in added.items():
                if tok.get("content") == "<|maxwell_end|>":
                    self.gguf_writer.add_eot_token_id(int(tid))
                    logger.info(f"gguf: eot_token_id = {tid}")
                    break

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams

        head_dim = hparams["attention_head_size"]
        self.gguf_writer.add_key_length(head_dim)
        self.gguf_writer.add_value_length(head_dim)
        self.gguf_writer.add_rope_dimension_count(head_dim)

        self.gguf_writer.add_leading_dense_block_count(self.n_dense_lead)
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_feed_forward_length(hparams["shared_expert_intermediate_size"])
        self.gguf_writer.add_expert_shared_count(1)
        # sigmoid scores, expert_bias only for the selection, the chosen scores renormalized
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
        self.gguf_writer.add_expert_weights_norm(True)
        self.gguf_writer.add_expert_weights_scale(1.0)

        # mHC, the reference hardcodes eps = 1e-6
        self.gguf_writer.add_hyper_connection_count(hparams["num_residual_streams"])
        self.gguf_writer.add_hyper_connection_sinkhorn_iterations(hparams["mhc_sinkhorn_iterations"])
        self.gguf_writer.add_hyper_connection_epsilon(1e-6)

        # block-sparse GQA: fixed prefix and local blocks plus top_k remote blocks per KV group
        block = hparams["block_sparse_gqa_block_size"]
        self.gguf_writer.add_indexer_head_count(hparams["block_sparse_gqa_indexer_n_heads"])
        self.gguf_writer.add_indexer_key_length(hparams["block_sparse_gqa_indexer_head_dim"])
        self.gguf_writer.add_indexer_top_k(hparams["block_sparse_gqa_num_remote_blocks"])
        self.gguf_writer.add_indexer_block_size(block)
        self.gguf_writer.add_indexer_local_blocks(hparams["block_sparse_gqa_local_window_size"] // block)
        self.gguf_writer.add_indexer_prefix_blocks(hparams["block_sparse_gqa_prefix_window_size"] // block)

        if self.mtp_only:
            self.gguf_writer.add_nextn_predict_layers(self.block_count - int(hparams["num_hidden_layers"]))
        elif not self.no_mtp and hparams.get("mtp_num_layers", 0):
            logger.warning("MaxwellMoe: dropping %d MTP layer(s); use --mtp to export a separate speculative draft",
                           hparams["mtp_num_layers"])

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        titem = TextModel.filter_tensors(item)
        if titem is None:
            return None

        name, gen = titem
        if name.startswith("model.mtp_layers."):
            name = name[len("model.") :]

        if name.startswith("mtp_layers."):
            if cls.no_mtp or not cls.mtp_only:
                return None
            if cls._original_block_count is None:
                raise ValueError("MaxwellMoe: MTP layer count was not initialized")

            parts = name.split(".", 2)
            if len(parts) != 3 or not parts[1].isdecimal():
                raise ValueError(f"MaxwellMoe: unsupported MTP tensor name {name!r}")

            mtp_idx = int(parts[1])
            bid = cls._original_block_count + mtp_idx
            rest = parts[2]
            names = cls._mtp_source_names.setdefault(mtp_idx, set())
            if rest in names:
                raise ValueError(f"MaxwellMoe: duplicate MTP tensor {name!r}")
            names.add(rest)
            cls._opt_num_mtp_layers = max(cls._opt_num_mtp_layers, mtp_idx + 1)

            if rest.startswith("decoder_layer."):
                decoder_tensor = rest[len("decoder_layer."):]
                if cls._MTP_DECODER_TENSOR_RE.fullmatch(decoder_tensor) is None:
                    raise ValueError(f"MaxwellMoe: unsupported MTP decoder tensor {name!r}")
                name = f"model.layers.{bid}.{decoder_tensor}"
            elif rest in ("e_proj.weight", "h_proj.weight"):
                projection = rest.split(".", 1)[0]
                projections = cls._mtp_proj_generators.setdefault(bid, {})
                if projection in projections:
                    raise ValueError(f"MaxwellMoe: duplicate MTP projection {projection!r} for layer {mtp_idx}")
                projections[projection] = gen
                if len(projections) < 2:
                    return None
                if set(projections) != {"e_proj", "h_proj"}:
                    raise ValueError(f"MaxwellMoe: invalid MTP projection pair for layer {mtp_idx}")

                e_gen = projections.pop("e_proj")
                h_gen = projections.pop("h_proj")
                del cls._mtp_proj_generators[bid]
                name = f"model.layers.{bid}.eh_proj.weight"

                def gen_mtp_eh_proj() -> Tensor:
                    e_weight = e_gen()
                    h_weight = h_gen()
                    if e_weight.ndim != 2 or h_weight.ndim != 2 or e_weight.shape != h_weight.shape:
                        raise ValueError(
                            "MaxwellMoe: e_proj and h_proj must be matching 2-D matrices; "
                            f"got {tuple(e_weight.shape)} and {tuple(h_weight.shape)}"
                        )
                    return torch.cat((e_weight, h_weight), dim=0)

                gen = gen_mtp_eh_proj
            elif rest in cls._MTP_ROOT_TENSORS:
                name = cls._MTP_ROOT_TENSORS[rest].format(bid=bid)
            else:
                raise ValueError(f"MaxwellMoe: unsupported MTP tensor {name!r}")

            return name, gen

        if cls.mtp_only:
            if name not in ("model.embed_tokens.weight", "lm_head.weight"):
                return None

        return name, gen

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # the reference runs the mHC mappings in fp32
        if new_name.endswith(("hc_attn_fn.weight", "hc_ffn_fn.weight", "output_hc_fn.weight")):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    @staticmethod
    def _checked_tensor_names(
        tensors: Iterable[tuple[str, Tensor]],
        source_name: str,
    ) -> Iterable[tuple[str, Tensor]]:
        # GGUF stores tensor names in a fixed 64-byte field, including the null terminator.
        for tensor_name, tensor_data in tensors:
            name_size = len(tensor_name.encode("utf-8"))
            if name_size >= 64:
                raise ValueError(
                    f"MaxwellMoe: tensor name {tensor_name!r} generated from "
                    f"{source_name!r} is {name_size} bytes; GGUF allows at most 63"
                )
            yield tensor_name, tensor_data

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.find("block_sparse_moe.experts.") != -1:
            n_experts = self.hparams["num_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch
            if len(self._experts[bid]) < n_experts * 3:
                return

            for wid in ("w1", "w2", "w3"):
                datas = [self._experts[bid].pop(f"model.layers.{bid}.block_sparse_moe.experts.{xid}.{wid}.weight")
                         for xid in range(n_experts)]
                merged = f"model.layers.{bid}.block_sparse_moe.experts.{wid}.weight"
                yield from self._checked_tensor_names(\n                    super().modify_tensors(torch.stack(datas, dim=0), merged, bid), merged\n                )
            return

        if name.endswith(".block_sparse_moe.expert_bias"):
            yield self.format_tensor_name(gguf.MODEL_TENSOR.FFN_EXP_PROBS_B, bid, ".bias"), data_torch
            return

        for hc, fn_scale, fn_base in (("self_attn_hyper_connection", gguf.MODEL_TENSOR.HC_ATTN_SCALE, gguf.MODEL_TENSOR.HC_ATTN_BASE),
                                      ("mlp_hyper_connection",       gguf.MODEL_TENSOR.HC_FFN_SCALE,  gguf.MODEL_TENSOR.HC_FFN_BASE)):
            prefix = f"model.layers.{bid}.{hc}."
            if not name.startswith(prefix):
                continue
            part = name[len(prefix):]
            if part == "bias":
                yield self.format_tensor_name(fn_base, bid), data_torch
                return
            if part in ("alpha_pre", "alpha_post", "alpha_res"):
                # the three gating factors scale the pre, post and residual mixes, in that order
                if self._hc_scale is None:
                    self._hc_scale = {}
                parts = self._hc_scale.setdefault(prefix, {})
                parts[part] = data_torch
                if len(parts) == 3:
                    del self._hc_scale[prefix]
                    scale = torch.cat([parts["alpha_pre"], parts["alpha_post"], parts["alpha_res"]]).reshape(3)
                    yield self.format_tensor_name(fn_scale, bid), scale
                return

        if name in ("model.hc_head.hc_head_fn", "model.hc_head.hc_head_base", "model.hc_head.hc_head_scale"):
            name += ".weight"

        yield from self._checked_tensor_names(super().modify_tensors(data_torch, name, bid), name)

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")
        if self._hc_scale:
            raise ValueError(f"Unprocessed mHC gating factors: {sorted(self._hc_scale)}")
        if type(self)._mtp_proj_generators:
            raise ValueError(f"Unprocessed MTP projections: {sorted(type(self)._mtp_proj_generators)}")
