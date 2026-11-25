#include "models.h"
#include "llama-kv-cache-msa.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

// Maxwell MoE (maxwell-model/Maxwell-V1-Flash): GQA with per-head QK-norm, leading dense FFN layers, then
// sigmoid-routed experts with a selection bias plus one shared expert, all wrapped in DeepSeek-V4 style mHC
// (4 residual streams, Sinkhorn-constrained mixing, learned stream collapse before the head).
//
// The reference attention is block-sparse GQA (MBSA): per KV group, 8 prefix + 8 local 128-token blocks plus
// 16 remote blocks selected by block-max scores from the learned indexer. The graph builds that route in
// bounded query/key chunks and applies its group-specific mask to the standard GQA attention operation.
// The mask is sparse, but the current attention backend still evaluates the full KV axis; KV-row execution
// is a separate optimization.

void llama_model_maxwell_moe::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,        hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,          hparams.n_layer_dense_lead);
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH,  hparams.n_ff_shexp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,                hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,               hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,                hparams.expert_weights_norm,  false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                 hparams.expert_gating_func);

    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,               hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);
    if (hparams.dsv4_hc_mult != 4) {
        throw std::runtime_error("maxwell-moe: only 4 hyper-connection streams are supported");
    }
    hparams.n_embd_nextn_impl = hparams.n_embd * hparams.dsv4_hc_mult;

    if (hparams.n_layer_nextn > 1) {
        throw std::runtime_error("maxwell-moe: only one MTP decoder layer is currently supported");
    }

    // Maxwell backbone attention requires the learned block-sparse indexer.
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,    hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,    hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,         hparams.indexer_top_k);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,    hparams.indexer_block_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS,  hparams.indexer_local_blocks);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_PREFIX_BLOCKS, hparams.indexer_prefix_blocks);
    if (!ml.get_key(LLM_KV_ATTENTION_INDEXER_DENSE_CONTEXT, hparams.indexer_dense_context, false)) {
        LLAMA_LOG_WARN("%s: GGUF has no attention.indexer.dense_context, the dense prefix is disabled; "
                       "reconvert the checkpoint to get the documented 16K dense prefix\n", __func__);
    }

    if (hparams.indexer_n_head != hparams.n_head_kv()) {
        throw std::runtime_error(format(
                "maxwell-moe: indexer head count (%u) must match KV head count (%u)",
                hparams.indexer_n_head, hparams.n_head_kv()));
    }
    if (hparams.indexer_head_size != hparams.n_embd_head_k()) {
        throw std::runtime_error(format(
                "maxwell-moe: indexer key length (%u) must match attention head size (%u)",
                hparams.indexer_head_size, hparams.n_embd_head_k()));
    }
    if (hparams.indexer_block_size == 0 ||
        hparams.indexer_top_k == 0 ||
        hparams.indexer_local_blocks == 0 ||
        hparams.indexer_prefix_blocks == 0) {
        throw std::runtime_error("maxwell-moe: indexer block size, top-k, prefix blocks and local blocks must be > 0");
    }

    switch (hparams.n_layer()) {
        case 45: type = LLM_TYPE_35B_A3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_maxwell_moe::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const auto flags = nextn_flags(ml);
    const int trunk_flags = flags.trunk;
    const int mtp_flags   = flags.mtp;

    if (n_layer_nextn > 1) {
        throw std::runtime_error("maxwell-moe: only one MTP decoder layer is currently supported");
    }

    const int64_t n_ff_exp   = hparams.n_ff_exp();
    const int64_t n_ff_shexp = hparams.n_ff_shexp;
    const int64_t hc         = hparams.dsv4_hc_mult;
    const int64_t hc_dim     = hc * n_embd;
    const int64_t hc_mix_dim = (2 + hc) * hc;
    const int64_t n_idx_head = hparams.indexer_n_head;
    const int64_t n_idx_dim  = hparams.indexer_head_size;

    if (n_expert == 0 || n_expert_used == 0) {
        throw std::runtime_error("maxwell-moe: n_expert and n_expert_used must be > 0");
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, 0);

    hc_head_fn    = create_tensor(tn(LLM_TENSOR_HC_HEAD_FN, "weight"), {hc_dim, hc}, 0);
    hc_head_base  = create_tensor(tn(LLM_TENSOR_HC_HEAD_BASE, "weight"), {hc}, 0);
    hc_head_scale = create_tensor(tn(LLM_TENSOR_HC_HEAD_SCALE, "weight"), {1}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, trunk_flags);
        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, trunk_flags);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, trunk_flags);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, trunk_flags);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, trunk_flags);

        // The learned indexer is part of Maxwell's backbone attention, not the dense MTP attention.
        layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", i), {n_embd, n_idx_head * n_idx_dim}, trunk_flags);
        layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", i), {n_embd, n_idx_dim}, trunk_flags);
        layer.index_q_norm = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", i), {n_idx_dim}, trunk_flags);
        layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", i), {n_idx_dim}, trunk_flags);

        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN, "weight", i), {hc_dim, hc_mix_dim}, trunk_flags);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE, "weight", i), {hc_mix_dim}, trunk_flags);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, trunk_flags);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN, "weight", i), {hc_dim, hc_mix_dim}, trunk_flags);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE, "weight", i), {hc_mix_dim}, trunk_flags);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE, "weight", i), {3}, trunk_flags);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, trunk_flags);

        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, trunk_flags);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, trunk_flags);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), {n_embd, n_ff}, trunk_flags);
        } else {
            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, trunk_flags);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, trunk_flags);
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, trunk_flags);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert}, trunk_flags);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, trunk_flags);
            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_shexp}, trunk_flags);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd}, trunk_flags);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "weight", i), {n_embd, n_ff_shexp}, trunk_flags);
        }
    }

    for (int i = n_layer; i < n_layer_all; ++i) {
        auto & layer = layers[i];

        // Maxwell MTP uses dense GQA and has no MSA indexer tensors.
        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, mtp_flags);
        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, mtp_flags);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, mtp_flags);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, mtp_flags);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, mtp_flags);

        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN, "weight", i), {hc_dim, hc_mix_dim}, mtp_flags);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE, "weight", i), {hc_mix_dim}, mtp_flags);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, mtp_flags);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN, "weight", i), {hc_dim, hc_mix_dim}, mtp_flags);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE, "weight", i), {hc_mix_dim}, mtp_flags);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE, "weight", i), {3}, mtp_flags);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, mtp_flags);
        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, mtp_flags);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, mtp_flags);
        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, mtp_flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert}, mtp_flags);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, mtp_flags);
        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_shexp}, mtp_flags);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd}, mtp_flags);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "weight", i), {n_embd, n_ff_shexp}, mtp_flags);

        // e_proj and h_proj are concatenated along their output rows by the converter.
        layer.nextn.eh_proj = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", i), {n_embd, 2*n_embd}, mtp_flags);
        layer.nextn.enorm   = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM, "weight", i), {n_embd}, mtp_flags);
        layer.nextn.hnorm   = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM, "weight", i), {n_embd}, mtp_flags);
    }
}
std::unique_ptr<llm_graph_context> llama_model_maxwell_moe::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(params);
    }
    return std::make_unique<graph>(params);
}

//
// mHC, the arithmetic of DeepSeek-V4 (same fused ops): x is [n_embd, hc, n_tokens]
//

static ggml_tensor * maxwell_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t i0) {
    return ggml_view_1d(ctx, t, ne0, ggml_row_size(t->type, i0));
}

static ggml_tensor * maxwell_view_2d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], ggml_row_size(t->type, i0));
}

static ggml_tensor * maxwell_hc_mean(ggml_context * ctx, ggml_tensor * x) {
    const int64_t hc = x->ne[1];

    ggml_tensor * acc = ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], 0);
    for (int64_t s = 1; s < hc; ++s) {
        acc = ggml_add(ctx, acc, ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], s*x->nb[1]));
    }
    return ggml_scale(ctx, acc, 1.0f/hc);
}

ggml_tensor * llama_model_maxwell_moe::graph::build_hc_collapse(
        ggml_tensor * x,
        ggml_tensor * weights,
        int           il) const {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];

    if (cparams.fused_dsv4_hc_pre && il >= 0) {
        ggml_tensor * result = ggml_dsv4_hc_pre(ctx0, x, weights);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_PRE, result, il});
        return result;
    }

    ggml_tensor * result = nullptr;
    for (int64_t ih = 0; ih < hc; ++ih) {
        ggml_tensor * xh = ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], ih*x->nb[1]);
        ggml_tensor * wh = ggml_view_2d(ctx0, weights, 1, nt, weights->nb[1], ih*weights->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, xh, wh);
        result = result ? ggml_add(ctx0, result, cur) : cur;
    }
    return result;
}

ggml_tensor * llama_model_maxwell_moe::graph::build_hc_sinkhorn(
        ggml_tensor * comb,
        int           il) const {
    GGML_UNUSED(il);

    // comb is [dst, src, n_tokens]: softmax over dst, one column normalization, then row/column rounds
    comb = ggml_soft_max(ctx0, comb);

    ggml_tensor * eps = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    eps = ggml_fill(ctx0, eps, hparams.dsv4_hc_eps);

    comb = ggml_add(ctx0, comb, eps);

    auto norm_cols = [&]() {
        ggml_tensor * comb_src_dst = ggml_cont(ctx0, ggml_permute(ctx0, comb, 1, 0, 2, 3));
        ggml_tensor * col_sum = ggml_sum_rows(ctx0, comb_src_dst);
        col_sum = ggml_add(ctx0, col_sum, eps);
        col_sum = ggml_permute(ctx0, col_sum, 1, 0, 2, 3);
        comb = ggml_div(ctx0, comb, col_sum);
    };

    auto norm_rows = [&]() {
        ggml_tensor * row_sum = ggml_sum_rows(ctx0, comb);
        row_sum = ggml_add(ctx0, row_sum, eps);
        comb = ggml_div(ctx0, comb, row_sum);
    };

    norm_cols();
    for (uint32_t i = 1; i < hparams.dsv4_hc_sinkhorn_iters; ++i) {
        norm_rows();
        norm_cols();
    }
    return comb;
}

llama_model_maxwell_moe::graph::hc_pre llama_model_maxwell_moe::graph::build_hc_pre(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base,
        int           il) const {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc*n_embd;
    const int64_t nt     = x->ne[2];

    // the reference scales the projection by 1/(rms + eps), not by 1/sqrt(ms + eps): the streams start at
    // the scale of the token embeddings, where the two differ
    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * rms  = ggml_sqrt(ctx0, ggml_mean(ctx0, ggml_sqr(ctx0, flat)));
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat);
    mixes = ggml_div(ctx0, mixes, ggml_scale_bias(ctx0, rms, 1.0f, hparams.dsv4_hc_eps));
    cb(mixes, "hc_mixes", il);

    ggml_tensor * pre = maxwell_view_2d(ctx0, mixes, hc, nt, 0);
    pre = ggml_add(ctx0, ggml_mul(ctx0, pre, maxwell_view_1d(ctx0, hc_scale, 1, 0)), maxwell_view_1d(ctx0, hc_base, hc, 0));
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);
    cb(pre, "hc_pre", il);

    ggml_tensor * post = maxwell_view_2d(ctx0, mixes, hc, nt, hc);
    post = ggml_add(ctx0, ggml_mul(ctx0, post, maxwell_view_1d(ctx0, hc_scale, 1, 1)), maxwell_view_1d(ctx0, hc_base, hc, hc));
    post = ggml_sigmoid(ctx0, post);
    post = ggml_scale(ctx0, post, 2.0f);
    cb(post, "hc_post", il);

    ggml_tensor * comb = nullptr;
    if (cparams.fused_dsv4_hc_comb) {
        comb = ggml_dsv4_hc_comb(ctx0, mixes, hc_scale, hc_base, hparams.dsv4_hc_eps,
                (int32_t) hparams.dsv4_hc_sinkhorn_iters);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_COMB, comb, il});
    } else {
        comb = maxwell_view_2d(ctx0, mixes, hc*hc, nt, 2*hc);
        comb = ggml_add(ctx0, ggml_mul(ctx0, comb, maxwell_view_1d(ctx0, hc_scale, 1, 2)), maxwell_view_1d(ctx0, hc_base, hc*hc, 2*hc));
        comb = ggml_reshape_3d(ctx0, comb, hc, hc, nt);
        comb = build_hc_sinkhorn(comb, il);
    }
    cb(comb, "hc_comb", il);

    return { build_hc_collapse(x, pre, il), post, comb };
}

ggml_tensor * llama_model_maxwell_moe::graph::build_hc_post(
        ggml_tensor * x,
        ggml_tensor * residual,
        ggml_tensor * post,
        ggml_tensor * comb,
        int           il) const {
    if (cparams.fused_dsv4_hc_post) {
        ggml_tensor * result = ggml_dsv4_hc_post(ctx0, x, residual, post, comb);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_POST, result, il});
        return result;
    }

    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[1];

    // out[dst] = post[dst] * x + sum_src comb[src -> dst] * residual[src]
    ggml_tensor * out = nullptr;
    for (int64_t dst = 0; dst < hc; ++dst) {
        ggml_tensor * post_dst = ggml_view_2d(ctx0, post, 1, nt, post->nb[1], dst*post->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, x, post_dst);

        for (int64_t src = 0; src < hc; ++src) {
            ggml_tensor * res_src = ggml_view_2d(ctx0, residual, n_embd, nt, residual->nb[2], src*residual->nb[1]);
            ggml_tensor * comb_src_dst = ggml_view_2d(ctx0, comb, 1, nt, comb->nb[2], dst*comb->nb[0] + src*comb->nb[1]);
            cur = ggml_add(ctx0, cur, ggml_mul(ctx0, res_src, comb_src_dst));
        }

        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, nt);
        out = out ? ggml_concat(ctx0, out, cur, 1) : cur;
    }
    return out;
}

ggml_tensor * llama_model_maxwell_moe::graph::build_hc_head(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base) const {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc*n_embd;
    const int64_t nt     = x->ne[2];

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, hparams.dsv4_hc_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_head_mixes", -1);

    ggml_tensor * pre = ggml_add(ctx0, ggml_mul(ctx0, mixes, hc_scale), hc_base);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);
    cb(pre, "hc_head_pre", -1);

    return build_hc_collapse(x, pre, -1);
}

uint64_t llama_model_maxwell_moe::graph_nodes_extra(uint32_t n_ubatch, uint32_t n_seq_max, uint32_t n_ctx_seq) const {
    constexpr uint64_t nodes_per_key_chunk   = 12;
    constexpr uint64_t nodes_per_query_chunk = 64;

    // a ubatch holds either many tokens of one stream or one token of many streams
    const uint64_t n_query_chunks = (n_ubatch + query_chunk_size - 1)/query_chunk_size + std::min(n_seq_max, n_ubatch);
    const uint64_t n_key_chunks   = (n_ctx_seq + key_chunk_size - 1)/key_chunk_size;

    return uint64_t(hparams.n_layer_all)*n_query_chunks*(n_key_chunks*nodes_per_key_chunk + nodes_per_query_chunk);
}

llama_model_maxwell_moe::graph::graph(
        const llm_graph_params & params,
        llama_model_maxwell_moe::graph::no_build) : llm_graph_context(params) {
}

llama_model_maxwell_moe::graph::graph(const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f/sqrtf(float(n_embd_head))
            : hparams.f_attention_scale;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv_msa(true, true);

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // every stream starts as the token embedding
    const int64_t hc = hparams.dsv4_hc_mult;
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
            res->t_layer_inp[il] = maxwell_hc_mean(ctx0, inpL);
            cb(res->t_layer_inp[il], "layer_inp", il);
            ggml_build_forward_expand(gf, res->t_layer_inp[il]);
        }

        // self-attention
        ggml_tensor * residual = inpL;
        const hc_pre attn_hc = build_hc_pre(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, il);
        cur = attn_hc.cur;
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);
        {
            auto [Qcur, Kcur, Vcur] = build_qkv(layer, cur, n_embd_head, n_head, n_head_kv, il);

            Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
            Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);
            cb(Kcur, "Kcur_normed", il);

            Qcur = build_rope(Qcur, inp_pos, il);
            Kcur = build_rope(Kcur, inp_pos, il);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // Project the shared indexer key and the per-KV-group indexer queries.
            ggml_tensor * indexer_q = build_lora_mm(layer.index_q_proj, cur);
            indexer_q = ggml_reshape_3d(ctx0, indexer_q,
                    hparams.indexer_head_size, hparams.indexer_n_head, n_tokens);
            indexer_q = build_norm(indexer_q, layer.index_q_norm, nullptr, LLM_NORM_RMS, il);
            indexer_q = build_rope_indexer(indexer_q, inp_pos, il);
            cb(indexer_q, "indexer_q", il);

            ggml_tensor * indexer_k = build_lora_mm(layer.index_k_proj, cur);
            indexer_k = build_norm(indexer_k, layer.index_k_norm, nullptr, LLM_NORM_RMS, il);
            indexer_k = ggml_reshape_3d(ctx0, indexer_k, hparams.indexer_head_size, 1, n_tokens);
            indexer_k = build_rope_indexer(indexer_k, inp_pos, il);
            cb(indexer_k, "indexer_k", il);

            const auto * mctx_base = inp_attn->mctx_msa->get_base();
            const auto * mctx_idx  = inp_attn->mctx_msa->get_idx();

            // Keep both cache writes ahead of the route construction and attention reads.
            ggml_build_forward_expand(gf, Qcur);
            ggml_build_forward_expand(gf, Kcur);
            ggml_build_forward_expand(gf, Vcur);
            ggml_build_forward_expand(gf, mctx_base->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il));
            ggml_build_forward_expand(gf, mctx_base->cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il));
            ggml_build_forward_expand(gf, mctx_idx->cpy_k(
                    ctx0, indexer_k, inp_attn->get_k_idxs_idx(), il));

            const int64_t n_pos = inp_attn->indexer_pos_slot->ne[0];
            const int64_t n_blocks = n_pos / hparams.indexer_block_size;
            const int64_t n_stream = ubatch.n_seqs_unq;
            const int64_t n_tokens_per_stream = n_tokens / n_stream;
            const int64_t n_indexer_groups = hparams.indexer_n_head;
            const int64_t n_top_k = std::min<int64_t>(hparams.indexer_top_k, n_blocks);

            GGML_ASSERT(n_pos % hparams.indexer_block_size == 0);
            GGML_ASSERT(n_tokens % n_stream == 0);
            GGML_ASSERT(n_indexer_groups == n_head_kv);

            // Convert the indexer cache from cache-cell order to token-position order.
            ggml_tensor * indexer_pos_slot = ggml_reshape_3d(
                    ctx0, inp_attn->indexer_pos_slot, n_pos, 1, n_stream);
            ggml_tensor * indexer_k_cache = ggml_permute(ctx0, mctx_idx->get_k(ctx0, il), 0, 2, 1, 3);

            // The cache is allocated for n_seq_max streams, while this ubatch can use fewer.
            // ggml_get_rows requires the cache stream dimension to match the index tensor.
            GGML_ASSERT(n_stream <= indexer_k_cache->ne[3]);
            indexer_k_cache = ggml_view_4d(ctx0, indexer_k_cache,
                    indexer_k_cache->ne[0], indexer_k_cache->ne[1],
                    indexer_k_cache->ne[2], n_stream,
                    indexer_k_cache->nb[1], indexer_k_cache->nb[2],
                    indexer_k_cache->nb[3], 0);

            ggml_tensor * indexer_k_pos = ggml_get_rows(ctx0, indexer_k_cache, indexer_pos_slot);
            cb(indexer_k_pos, "indexer_k_pos", il);

            ggml_tensor * k_cache = mctx_base->get_k(ctx0, il);
            ggml_tensor * v_cache = mctx_base->get_v(ctx0, il);

            ggml_tensor * stream_outputs = nullptr;

            for (int64_t stream = 0; stream < n_stream; ++stream) {
                ggml_tensor * query_outputs = nullptr;

                ggml_tensor * k_stream = ggml_view_4d(ctx0, k_cache,
                        k_cache->ne[0], k_cache->ne[1], k_cache->ne[2], 1,
                        k_cache->nb[1], k_cache->nb[2], k_cache->nb[3],
                        stream*k_cache->nb[3]);
                ggml_tensor * v_stream = ggml_view_4d(ctx0, v_cache,
                        v_cache->ne[0], v_cache->ne[1], v_cache->ne[2], 1,
                        v_cache->nb[1], v_cache->nb[2], v_cache->nb[3],
                        stream*v_cache->nb[3]);

                ggml_tensor * indexer_k_stream = ggml_view_4d(ctx0, indexer_k_pos,
                        indexer_k_pos->ne[0], indexer_k_pos->ne[1], 1, 1,
                        indexer_k_pos->nb[1], indexer_k_pos->nb[2], indexer_k_pos->nb[3],
                        stream*indexer_k_pos->nb[3]);
                ggml_tensor * indexer_valid_stream = ggml_view_4d(ctx0, inp_attn->indexer_pos_valid,
                        n_pos, 1, 1, 1,
                        inp_attn->indexer_pos_valid->nb[1],
                        inp_attn->indexer_pos_valid->nb[1],
                        inp_attn->indexer_pos_valid->nb[1],
                        stream*inp_attn->indexer_pos_valid->nb[1]);

                for (int64_t query_start = 0; query_start < n_tokens_per_stream; query_start += query_chunk_size) {
                    const int64_t query_count = std::min<int64_t>(
                            query_chunk_size, n_tokens_per_stream - query_start);
                    const int64_t token_offset = stream*n_tokens_per_stream + query_start;

                    ggml_tensor * indexer_q_chunk = ggml_view_3d(ctx0, indexer_q,
                            indexer_q->ne[0], indexer_q->ne[1], query_count,
                            indexer_q->nb[1], indexer_q->nb[2],
                            token_offset*indexer_q->nb[2]);

                    ggml_tensor * block_scores = nullptr;

                    // Bound the score temporary to key_chunk_size * query_count * groups.
                    for (int64_t key_start = 0; key_start < n_pos; key_start += key_chunk_size) {
                        const int64_t key_count = std::min<int64_t>(key_chunk_size, n_pos - key_start);
                        ggml_tensor * indexer_k_chunk = ggml_view_4d(ctx0, indexer_k_stream,
                                indexer_k_stream->ne[0], key_count, 1, 1,
                                indexer_k_stream->nb[1], indexer_k_stream->nb[2], indexer_k_stream->nb[3],
                                key_start*indexer_k_stream->nb[1]);

                        ggml_tensor * token_scores = ggml_mul_mat(ctx0, indexer_k_chunk, indexer_q_chunk);
                        ggml_prec_set_acc(token_scores, GGML_PREC_F32);
                        token_scores = ggml_permute(ctx0, token_scores, 0, 2, 1, 3);
                        token_scores = ggml_scale(ctx0, token_scores,
                                1.0f/sqrtf(float(hparams.indexer_head_size)));

                        ggml_tensor * valid_chunk = ggml_view_4d(ctx0, indexer_valid_stream,
                                key_count, 1, 1, 1,
                                indexer_valid_stream->nb[1],
                                indexer_valid_stream->nb[2],
                                indexer_valid_stream->nb[3],
                                key_start*indexer_valid_stream->nb[0]);
                        token_scores = ggml_add(ctx0, token_scores, valid_chunk);

                        ggml_tensor * pooled = ggml_pool_2d(ctx0, token_scores, GGML_OP_POOL_MAX,
                                hparams.indexer_block_size, 1,
                                hparams.indexer_block_size, 1, 0, 0);
                        cb(pooled, "indexer_block_scores", il);

                        block_scores = block_scores
                                ? ggml_concat(ctx0, block_scores, pooled, 0)
                                : pooled;
                    }

                    ggml_tensor * remote_mask = ggml_view_4d(ctx0, inp_attn->indexer_remote_mask,
                            n_blocks, query_count, 1, 1,
                            inp_attn->indexer_remote_mask->nb[1],
                            inp_attn->indexer_remote_mask->nb[2],
                            inp_attn->indexer_remote_mask->nb[3],
                            stream*inp_attn->indexer_remote_mask->nb[3]
                                + query_start*inp_attn->indexer_remote_mask->nb[1]);
                    block_scores = ggml_add(ctx0, block_scores, remote_mask);
                    cb(block_scores, "indexer_remote_scores", il);

                    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, block_scores, n_top_k));
                    cb(top_k, "indexer_top_k", il);

                    ggml_tensor * base_mask = ggml_view_4d(ctx0, inp_attn->self_attn_mask.raw,
                            inp_attn->self_attn_mask.raw->ne[0], query_count, 1, 1,
                            inp_attn->self_attn_mask.raw->nb[1],
                            inp_attn->self_attn_mask.raw->nb[2],
                            inp_attn->self_attn_mask.raw->nb[3],
                            stream*inp_attn->self_attn_mask.raw->nb[3]
                                + query_start*inp_attn->self_attn_mask.raw->nb[1]);

                    ggml_tensor * blocks = ggml_view_4d(ctx0, inp_attn->indexer_blocks,
                            hparams.indexer_block_size, n_blocks, 1, 1,
                            inp_attn->indexer_blocks->nb[1],
                            inp_attn->indexer_blocks->nb[2],
                            inp_attn->indexer_blocks->nb[3],
                            stream*inp_attn->indexer_blocks->nb[3]);

                    const int64_t n_extra =
                            (hparams.indexer_prefix_blocks + hparams.indexer_local_blocks)
                            * hparams.indexer_block_size;
                    ggml_tensor * extra = ggml_view_4d(ctx0, inp_attn->indexer_extra,
                            n_extra, query_count, 1, 1,
                            inp_attn->indexer_extra->nb[1],
                            inp_attn->indexer_extra->nb[2],
                            inp_attn->indexer_extra->nb[3],
                            stream*inp_attn->indexer_extra->nb[3]
                                + query_start*inp_attn->indexer_extra->nb[1]);

                    ggml_tensor * sparse_mask = ggml_top_k_mask(
                            ctx0, base_mask, top_k, block_scores, blocks, extra);

                    // Dense-context queries use the causal mask; later queries keep the block-sparse route.
                    if (inp_attn->indexer_dense_gate) {
                        ggml_tensor * gate = ggml_view_4d(ctx0, inp_attn->indexer_dense_gate,
                                1, query_count, 1, 1,
                                inp_attn->indexer_dense_gate->nb[1],
                                inp_attn->indexer_dense_gate->nb[2],
                                inp_attn->indexer_dense_gate->nb[3],
                                stream*inp_attn->indexer_dense_gate->nb[3]
                                    + query_start*inp_attn->indexer_dense_gate->nb[1]);

                        // Keep masked values finite: multiplying -inf by a zero gate would produce NaN.
                        // Blend in the mask type: sparse - gate*(sparse - dense). The dense mask has one head only.
                        constexpr float mask_floor = -30000.0f;
                        if (gate->type != sparse_mask->type) {
                            gate = ggml_cast(ctx0, gate, sparse_mask->type);
                        }
                        ggml_tensor * sparse_finite = ggml_clamp(ctx0, sparse_mask, mask_floor, 0.0f);
                        ggml_tensor * dense_finite  = ggml_clamp(ctx0, base_mask,   mask_floor, 0.0f);
                        ggml_tensor * delta = ggml_mul(ctx0, ggml_sub(ctx0, sparse_finite, dense_finite), gate);
                        sparse_mask = ggml_sub(ctx0, sparse_finite, delta);
                    }
                    cb(sparse_mask, "maxwell_sparse_mask", il);

                    ggml_tensor * q_chunk = ggml_view_3d(ctx0, Qcur,
                            Qcur->ne[0], Qcur->ne[1], query_count,
                            Qcur->nb[1], Qcur->nb[2],
                            token_offset*Qcur->nb[2]);

                    // Every KV group becomes a channel (ne[3]) of the attention op, so that the group's own sparse
                    // mask lines up with the KV head that the query heads of the group read:
                    // k, v [D, n_head_kv, n_kv, 1] -> [D, n_kv, 1, n_head_kv], mask [n_kv, n_q, n_head_kv] -> [n_kv, n_q, 1, n_head_kv]
                    // The GQA ratio is taken from the tensors, which keeps synthetic test models valid.
                    ggml_tensor * k_fa = ggml_permute(ctx0, k_stream, 0, 3, 1, 2);
                    ggml_tensor * v_fa = ggml_permute(ctx0, v_stream, 0, 3, 1, 2);
                    ggml_tensor * mask_fa = ggml_reshape_4d(ctx0, sparse_mask,
                            sparse_mask->ne[0], sparse_mask->ne[1], 1, sparse_mask->ne[2]);
                    // Flash Attention requires an F16 mask, the mask is F32 without -fa
                    if (mask_fa->type != GGML_TYPE_F16) {
                        mask_fa = ggml_cast(ctx0, mask_fa, GGML_TYPE_F16);
                    }

                    GGML_ASSERT(k_fa->ne[3] == mask_fa->ne[3]);
                    GGML_ASSERT(q_chunk->ne[1] % k_fa->ne[3] == 0);
                    const int64_t gqa_group_size = q_chunk->ne[1] / k_fa->ne[3];

                    ggml_tensor * attn_chunk = build_attn_msa_fa(
                            q_chunk, k_fa, v_fa, mask_fa, gqa_group_size, kq_scale, il);
                    cb(attn_chunk, "attn_sparse_chunk", il);

                    query_outputs = query_outputs
                            ? ggml_concat(ctx0, query_outputs, attn_chunk, 1)
                            : attn_chunk;
                }

                stream_outputs = stream_outputs
                        ? ggml_concat(ctx0, stream_outputs, query_outputs, 1)
                        : query_outputs;
            }

            cur = build_lora_mm(layer.wo, stream_outputs, layer.wo_s);
            if (layer.wo_b) {
                cur = ggml_add(ctx0, cur, layer.wo_b);
            }
            cb(cur, "attn_out", il);
        }
        inpL = build_hc_post(cur, residual, attn_hc.post, attn_hc.comb, il);
        cb(inpL, "attn_resid", il);

        // feed-forward
        residual = inpL;
        const hc_pre ffn_hc = build_hc_pre(inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, il);
        cur = ffn_hc.cur;
        cb(cur, "hc_ffn_pre", il);

        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                    layer.ffn_up,   nullptr, nullptr,
                    layer.ffn_gate, nullptr, nullptr,
                    layer.ffn_down, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            ggml_tensor * moe_out = build_moe_ffn(cur,
                    layer.ffn_gate_inp,
                    layer.ffn_up_exps,
                    layer.ffn_gate_exps,
                    layer.ffn_down_exps,
                    layer.ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il,
                    nullptr, nullptr, nullptr, nullptr, 0.0f);
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * ffn_shexp = build_ffn(cur,
                    layer.ffn_up_shexp,   nullptr, nullptr,
                    layer.ffn_gate_shexp, nullptr, nullptr,
                    layer.ffn_down_shexp, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        }

        inpL = build_hc_post(cur, residual, ffn_hc.post, ffn_hc.comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_out", il);
    }

    if ((size_t) n_layer < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[n_layer]) {
        res->t_layer_inp[n_layer] = maxwell_hc_mean(ctx0, inpL);
        cb(res->t_layer_inp[n_layer], "layer_inp", n_layer);
        ggml_build_forward_expand(gf, res->t_layer_inp[n_layer]);
    }

    // Preserve all batch-token states when the device extraction path is active.
    // The device buffer is indexed by original batch token, not by selected output token.
    ggml_tensor * h_nextn_all = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);

    if (inp_out_ids) {
        ggml_tensor * flat = ggml_get_rows(ctx0, h_nextn_all, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, flat, n_embd, hc, n_outputs);
    }

    // Keep the full state for device extraction and the selected rows for host extraction.
    res->t_h_nextn = cparams.embeddings_nextn_dev
            ? h_nextn_all
            : ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_outputs);
    cb(res->t_h_nextn, "h_nextn", -1);

    cur = build_hc_head(inpL, model.hc_head_fn, model.hc_head_scale, model.hc_head_base);
    cb(cur, "hc_head", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
}


llama_model_maxwell_moe::graph_mtp::graph_mtp(const llm_graph_params & params)
    : graph(params, llama_model_maxwell_moe::graph::no_build{}) {
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "MAXWELL MTP currently supports exactly one draft decoder layer");

    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MAXWELL MTP: missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm && "MAXWELL MTP: missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm && "MAXWELL MTP: missing nextn.hnorm");
    GGML_ASSERT(layer.ffn_gate_inp && "MAXWELL MTP: missing MTP router weights");
    GGML_ASSERT(layer.ffn_exp_probs_b && "MAXWELL MTP: missing MTP expert selection bias");

    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd_inp(), hparams.n_embd_nextn());
    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);
    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    ggml_tensor * tok_embd;
    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.token) {
        tok_embd = ggml_get_rows(ctx0, model.tok_embd, inp->tokens);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    inp->h = build_inp_embd_state(inp.get(), hparams.n_embd_nextn());
    ggml_tensor * h_embd = inp->h;
    res->add_input(std::move(inp));

    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    auto * inp_attn = build_attn_inp_kv();

    // Normalize each of the four residual streams independently.
    ggml_tensor * h_streams = ggml_reshape_3d(ctx0, h_embd, n_embd, hc, n_tokens);
    ggml_tensor * h_norm = build_norm(h_streams, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * eh_proj = layer.nextn.eh_proj;
    GGML_ASSERT(eh_proj->ne[0] == n_embd && eh_proj->ne[1] == 2*n_embd);
    ggml_tensor * e_proj = ggml_view_2d(ctx0, eh_proj, n_embd, n_embd, eh_proj->nb[1], 0);
    ggml_tensor * h_proj = ggml_view_2d(ctx0, eh_proj, n_embd, n_embd, eh_proj->nb[1], n_embd*eh_proj->nb[1]);

    ggml_tensor * e_projected = build_lora_mm(e_proj, e_norm);
    e_projected = ggml_reshape_3d(ctx0, e_projected, n_embd, 1, n_tokens);
    ggml_tensor * h_projected = build_lora_mm(h_proj, h_norm);
    ggml_tensor * cur = ggml_add(ctx0, ggml_repeat(ctx0, e_projected, h_projected), h_projected);
    cb(cur, "mtp_input_projection", il);

    // Maxwell's MTP decoder uses dense GQA rather than the backbone's block-sparse indexer.
    ggml_tensor * residual = cur;
    const hc_pre attn_hc = build_hc_pre(cur, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, il);
    cur = build_norm(attn_hc.cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);

    const int64_t n_embd_head = hparams.n_embd_head_v();
    auto [Qcur, Kcur, Vcur] = build_qkv(layer, cur, n_embd_head, n_head, n_head_kv, il);
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    Qcur = build_rope(Qcur, inp_pos, il);
    Kcur = build_rope(Kcur, inp_pos, il);

    const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(n_embd_head))
            : hparams.f_attention_scale;
    cur = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
    if (layer.wo_b) {
        cur = ggml_add(ctx0, cur, layer.wo_b);
    }
    cur = build_hc_post(cur, residual, attn_hc.post, attn_hc.comb, il);
    cb(cur, "mtp_attn_residual", il);

    residual = cur;
    const hc_pre ffn_hc = build_hc_pre(cur, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, il);
    cur = build_norm(ffn_hc.cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);

    ggml_tensor * moe_out = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            n_expert, n_expert_used,
            LLM_FFN_SILU, hparams.expert_weights_norm,
            hparams.expert_weights_scale,
            (llama_expert_gating_func_type) hparams.expert_gating_func,
            il,
            nullptr, nullptr, nullptr, nullptr, 0.0f);

    ggml_tensor * ffn_shexp = build_ffn(cur,
            layer.ffn_up_shexp, nullptr, nullptr,
            layer.ffn_gate_shexp, nullptr, nullptr,
            layer.ffn_down_shexp, nullptr, nullptr,
            nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
    cur = ggml_add(ctx0, moe_out, ffn_shexp);
    cur = build_hc_post(cur, residual, ffn_hc.post, ffn_hc.comb, il);
    cur = build_cvec(cur, il);
    cb(cur, "mtp_decoder_out", il);

    // Keep the multistream state for the next speculative token.
    ggml_tensor * h_nextn = ggml_reshape_2d(ctx0, cur, hc_dim, n_tokens);
    if (inp_out_ids) {
        h_nextn = ggml_get_rows(ctx0, h_nextn, inp_out_ids);
    }
    res->t_h_nextn = h_nextn;

    ggml_tensor * head_inp = ggml_reshape_3d(ctx0, h_nextn, n_embd, hc, n_outputs);
    ggml_tensor * head = build_hc_head(head_inp, model.hc_head_fn, model.hc_head_scale, model.hc_head_base);
    head = build_norm(head, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    res->t_embd = head;

    ggml_tensor * logits = build_lora_mm(model.output, head);
    res->t_logits = logits;
    ggml_build_forward_expand(gf, logits);
}
