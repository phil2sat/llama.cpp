# Maxwell-V1-Flash implementation in llama.cpp

## Purpose and scope

This document maps the Maxwell-V1-Flash (maxwell-moe) implementation: checkpoint conversion, GGUF metadata and tensor naming, model loading, graph construction, MSA KV-cache integration, MTP export/runtime, and chat-parser coverage. It distinguishes implementation from validation: a parser fixture is not an end-to-end model test, and a sparse mask is not sparse KV-row execution.

The branch also contains unrelated upstream/runtime changes. This document covers the Maxwell-specific path and its immediate shared integration points; it does not claim that every branch-wide change is part of Maxwell.

## End-to-end data flow

1. Hugging Face checkpoint conversion starts in conversion/maxwell_moe.py.
2. The converter emits architecture metadata and mapped tensors through the shared HF-to-GGUF converter and GGUF writer.
3. src/llama-arch.h and src/llama-arch.cpp define the architecture ID, GGUF key names, and architecture-name mapping.
4. src/llama-model.cpp dispatches the architecture to llama_model_maxwell_moe. src/models/maxwell-moe.cpp reads and validates model hyperparameters, requests tensors, and builds the model graph.
5. src/llama-graph.cpp supplies graph inputs and graph-reuse checks; src/llama-kv-cache-msa.cpp supplies MSA cache/indexer state.
6. models/templates/Maxwell-V1-Flash.jinja is the chat template shipped for the architecture; tests/test-chat.cpp exercises reasoning/content/tool-call parsing with it. It does not run Maxwell inference.

## Checkpoint conversion and GGUF contract

### conversion/maxwell_moe.py

Registers MaxwellMoeModel for HF architecture MaxwellMoeForCausalLM and the example repository maxwell-model/Maxwell-V1-Flash. It translates source tensor names and configuration into the architecture/tensor inventory expected by llama.cpp.

MTP conversion is explicit: accepted MTP tensor names are enumerated and optional MTP export state is tracked, so unsupported checkpoint revisions are not silently treated as valid draft weights. For a missing or unexpected tensor, compare the source-key handling here with the C++ tensor requests in src/models/maxwell-moe.cpp.

### Shared conversion and GGUF files

- convert_hf_to_gguf.py: entry point that exposes the registered converter.
- conversion/base.py: shared conversion behavior.
- gguf-py/gguf/constants.py: Python architecture, tensor, and metadata identifiers.
- gguf-py/gguf/tensor_mapping.py: shared tensor-name mapping.
- gguf-py/gguf/gguf_writer.py: serializes metadata and tensor payloads.
- src/llama-arch.h: C++ architecture enum and GGUF key declarations; identifiers must remain synchronized with Python constants.
- src/llama-arch.cpp: architecture-name mapping and GGUF-key-to-hyperparameter mapping.
- src/llama-model-saver.cpp: writes model metadata, including the dense-context field.

The Maxwell metadata key attention.indexer.dense_context is carried into hparams.indexer_dense_context. The converter defaults it to 16,384 tokens when the source configuration does not provide a value. This value chooses the attention mask and skips indexer routing construction for dense-prefix queries; indexer keys are still cached for later sparse queries.

**Required GGUF metadata for paper-compatible dense prefix.** Both the target GGUF and the standalone MTP draft GGUF must contain `maxwell-moe.attention.indexer.dense_context=16384` for Maxwell-V1-Flash. The conversion output must be checked for both files; the observed MTP output omitted this key, so do not assume the target's metadata is automatically present in the draft. If either GGUF is missing the key or has a value that is not read, the loader falls back to sparse attention from the first token (no dense prefix). This fallback is not paper-compatible. Rebuild the affected GGUF with the required metadata and verify the key/value in its metadata dump.

## Model registration, hyperparameters, and tensor loading

### src/llama-model.cpp

Architecture dispatch selects llama_model_maxwell_moe for LLM_ARCH_MAXWELL_MOE. This is the boundary from generic loading to the Maxwell-specific loader.

### src/models/models.h

Declares the Maxwell model/graph-builder types and connects them to shared model graph interfaces. The implementation remains in src/models/maxwell-moe.cpp.

### src/models/maxwell-moe.cpp

This is the central implementation file.

**Hyperparameters and validation.** The loader reads RMS-normalization epsilon, leading dense-layer count, expert feed-forward dimensions, indexer parameters, and dense-context threshold. attention.indexer.dense_context is optional; when a GGUF lacks it the loader logs a warning and runs without a dense prefix instead of failing. It validates indexer head count against KV-head count and indexer head size against model head size. Invalid zero block-size, top-k, prefix-block, or local-block settings are rejected before graph construction.

**Tensor inventory.** The loader requests backbone attention projections and per-head Q/K normalization, indexer projections and normalization, leading dense FFN tensors, routed-expert tensors, shared-expert tensors, and hyper-connection mixing/collapse tensors. MTP tensors are loaded as a separate decoder inventory, not as backbone layers.

**Residual streams.** Maxwell uses four residual streams. Learned hyper-connection parameters produce stream-mixing weights around attention and FFN updates; the learned stream-collapse projection combines the streams before the output head. This is not a conventional single-residual MoE block.

**Attention and FFN.** The model uses GQA with per-head Q/K normalization. Leading layers use dense FFNs; later layers use sigmoid-routed experts with selection bias and a shared expert. These distinctions follow the model configuration and tensor layout.

**Indexer and mask construction.** The graph projects and normalizes indexer queries/keys, computes query/key scores in bounded chunks, accumulates scores in F32, scales by inverse square root of indexer head size, and max-pools token scores into block scores. It applies the remote-position mask, selects top-k remote blocks, and combines them with configured prefix and local blocks. The resulting per-KV-head mask is the visibility map passed to attention.

**Dense-context gate.** The graph uses actual input token positions. Positions before indexer_dense_context select the dense causal mask; positions at or beyond it select the sparse indexer mask. The gate is combined with finite dense/sparse masks before attention. Dense-prefix positions bypass indexer scoring and top-k construction. Sparse routing begins at the configured dense-context boundary.

**Attention tensor layout.** K/V are permuted to [D, n_kv, 1, n_head_kv]; the mask is reshaped to [n_kv, n_q, 1, n_head_kv] and converted to F16 for FlashAttention. GQA group size is derived from tensor dimensions before the standard MSA FlashAttention builder is called. The implementation applies a sparse mask to attention; it does not implement sparse KV-row execution, so the backend may still evaluate the full KV axis.

## Graph inputs and execution state

### src/llama-graph.cpp

Owns generic graph-input preparation and graph reuse. For Maxwell, it records each query's dense/sparse route from its token position, provides MSA indexer inputs only when a batch contains sparse queries, and checks the route signature when deciding whether a graph can be reused. The gate follows actual batch positions rather than assuming a fixed prompt boundary.

### src/llama-graph.h

Declares graph input/state structures used by graph construction and execution. Maxwell-specific input data is carried through these shared structures.

### src/llama-kv-cache-msa.h and src/llama-kv-cache-msa.cpp

Implement the MSA KV-cache context used by Maxwell. This layer owns MSA-specific cache state and supplies cache/indexer context consumed by the graph. It is distinct from ordinary KV-cache handling because MSA needs indexer slot/block metadata in addition to normal K/V storage.

The main-context cache is restricted to the 45 backbone layers; the appended MTP layer is allocated by the separate MTP context and must not be duplicated in the backbone cache. The indexer cache is key-only. The full-history backbone K/V allocation is intentional: Maxwell's inference design retains historical K/V and sparsifies which blocks participate in each attention operation. The 32-block budget (8 prefix, 8 local, up to 16 remote) is an attention-access budget, not a KV-retention or cache-capacity limit. Do not cap the backbone cache at the sparse attention budget; doing so would discard blocks that later indexer queries may select.

### src/llama-hparams.h and src/llama-hparams.cpp

Declare and initialize model hyperparameters used by loading and graph construction, including dense-context/indexer settings. A new Maxwell metadata field must stay consistent across architecture key mapping, hyperparameters, writer, and converter.

## MTP and speculative decoding

The converter has an explicit Maxwell MTP tensor inventory and export controls in conversion/maxwell_moe.py. The C++ loader keeps MTP tensors separate from backbone tensors; MTP attention is dense GQA rather than the backbone's learned sparse-indexer path.

Shared speculative-decoding control flow is in common/speculative.cpp; shared graph/model interfaces are in src/models/models.h and src/llama-graph.*. Maxwell support fits those interfaces: target/draft state handling and verification are shared speculative-decoding responsibilities, while Maxwell-specific tensor loading and graph construction belong to the model and converter.

## Backend coverage for the sparse mask

GGML_OP_TOP_K_MASK is a generic ggml op; the fork does not add an opcode. Backends that cannot run it keep the op on the CPU path.

**OpenCL.** ggml/src/ggml-opencl/kernels/top_k_mask.cl masks one destination row per work item. The kernel resolves broadcast strides the way the CPU reference does (nb2/nb3 of the block map and of the extra cells), so a block map with fewer rows than the mask, or extra cells shared across rows, stays correct instead of reading neighbouring rows. ggml_opencl_supports_op therefore accepts these shapes; it still rejects non-I32/I64 ids, unsupported mask/score types, and shapes the kernel cannot address. The host code compiles with GGML_OPENCL=ON and the kernel passes a clang OpenCL C syntax check, but the change was not executed on an OpenCL device here.

**Hexagon.** ggml/src/ggml-hexagon/htp/top-k-mask-ops.c implements the op for HTP. The kernel params builder accepts the shapes Maxwell builds: the graph slices block map, extra cells, and base mask per stream into 4D views with ne[2] = ne[3] = 1 and per-query extra rows, which passes the row-flat and stride checks. Shapes the HTP kernel cannot address (a block map spanning streams, extra cells broadcast over rows, an oversized blocks row stride) return false from the kernel params builder and stay on the CPU. This path was not executed on Hexagon hardware here.

**Backend table.** docs/ops.md is generated from the CSV snapshots in docs/ops/ by scripts/create_ops_docs.py, so it reports what those snapshots contain, not what the current tree implements. The CPU rows for GGML_OP_TOP_K_MASK were regenerated on this branch with test-backend-ops support -b CPU and the table now lists the op as supported for CPU. The CUDA, Metal, SYCL, and OpenCL snapshots predate the op, and the OpenCL rows that do exist were captured before the broadcast fix, so the generated table still shows the op as unsupported for backends that carry it in this tree (ggml-cuda/top-k-mask.cu, ggml-metal/kernels/misc.metal, ggml/src/ggml-vulkan, ggml/src/ggml-opencl). Regenerate those CSVs on the matching hardware before trusting the table for non-CPU backends.

## Chat parsing fixture

### tests/test-chat.cpp

The Maxwell fixture covers a completion containing reasoning, ordinary content, and an XML-style tool call. It runs against models/templates/Maxwell-V1-Flash.jinja, the template shipped with the model, and checks the rendered turns (`<|maxwell_start|>`/`<|maxwell_end|>`), the generation prompt, the reasoning tag, ordinary content, and the XML-style tool call. The reasoning tags are read from the applied template, so the fixture stays valid if the template changes them. It does not load Maxwell weights, validate logits, exercise MSA attention, or prove tool execution by a running server.

## Investigation paths

- Wrong or missing GGUF metadata: conversion/maxwell_moe.py → gguf-py/gguf/constants.py → src/llama-arch.h and src/llama-arch.cpp → src/llama-hparams.* → src/llama-model-saver.cpp.
- Missing or mismatched tensor: converter mapping in conversion/maxwell_moe.py → tensor requests in src/models/maxwell-moe.cpp → generic model loader in src/llama-model.cpp.
- Incorrect dense/sparse boundary: metadata default and key mapping → hparams.indexer_dense_context → gate population in src/llama-graph.cpp → mask combination and MSA attention in src/models/maxwell-moe.cpp.
- Incorrect sparse block visibility: MSA cache metadata in src/llama-kv-cache-msa.* → indexer scoring, top-k, and mask construction in src/models/maxwell-moe.cpp → attention mask input.
- MTP tensor mismatch: source MTP inventory and conversion controls in conversion/maxwell_moe.py → MTP tensor loading and graph path in the C++ model → shared speculative flow in common/speculative.cpp.
- Reasoning/tool-call parsing mismatch: fixture in tests/test-chat.cpp → shared chat-template/parser code in common/.

## Limits

This document maps the Maxwell-specific implementation and immediate integration points. It does not claim a full audit of every unrelated branch change, successful compilation, backend parity, or runtime correctness. The OpenCL and Hexagon paths were reviewed and, for OpenCL, compiled; neither was run on the matching accelerator. The chat fixture is parser-level only; a real Maxwell GGUF and backend execution are required to validate model behavior.
