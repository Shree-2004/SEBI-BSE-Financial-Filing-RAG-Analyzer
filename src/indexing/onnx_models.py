"""
onnx_models.py

Lean ONNX Runtime inference for the two models the API runs per request —
the bge-base embedder and the MiniLM CrossEncoder reranker — so the API
image needs neither PyTorch nor sentence-transformers.

Why not just PyTorch / FastEmbed (measured, Windows RSS):
- PyTorch: models load at ~500 MB, then the first inference adds ~330 MB
  of runtime buffers -> ~970 MB peak. That crashed Render's 512 MB tier.
- FastEmbed with its defaults: ~1,020-1,080 MB peak (ONNX Runtime's memory
  arena + an fp16, not int8, embedder).
Here: int8 weights (embedder 416 -> 105 MB, reranker 87 -> 22 MB) and an
ONNX Runtime session with the memory arena off.

Model files are prepared once by `python src/indexing/onnx_models.py
prepare` (downloads BAAI's ONNX export + Xenova's int8 reranker, and
int8-quantizes the embedder), into MODELS_DIR (default ./models).
"""

import os
from pathlib import Path

import numpy as np

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
EMBEDDER_DIR = MODELS_DIR / "bge-base-en-v1.5"
RERANKER_DIR = MODELS_DIR / "ms-marco-MiniLM-L-6-v2"
EMBEDDER_FILES = {"int8": "model_int8.onnx", "fp32": "onnx/model.onnx"}
RERANKER_FILES = {"int8": "onnx/model_quantized.onnx", "fp32": "onnx/model.onnx"}


class OnnxEncoder:
    """A BERT-style ONNX model + its HF fast tokenizer."""

    def __init__(self, model_path: Path, tokenizer_path: Path, max_length: int = 512):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        opts = ort.SessionOptions()
        # memory over speed: no pre-grown arena, no cached allocation plans;
        # a 2-thread pool is plenty for one query + 10 rerank pairs
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        opts.intra_op_num_threads = int(os.getenv("ONNX_THREADS", "2"))
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length)
        pad_id = self.tokenizer.token_to_id("[PAD]") or 0
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]")

    def run(self, encodings) -> np.ndarray:
        feed = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
        }
        return self.session.run(None, {k: v for k, v in feed.items() if k in self.input_names})[0]


class OnnxEmbedder(OnnxEncoder):
    def __init__(self, variant: str | None = None):
        variant = variant or os.getenv("EMBED_ONNX_VARIANT", "int8")
        super().__init__(EMBEDDER_DIR / EMBEDDER_FILES[variant], EMBEDDER_DIR / "tokenizer.json")

    def embed(self, texts: list[str]) -> np.ndarray:
        hidden = self.run(self.tokenizer.encode_batch(texts))
        cls = hidden[:, 0]                                    # bge uses CLS pooling
        return cls / np.linalg.norm(cls, axis=1, keepdims=True)


class OnnxReranker(OnnxEncoder):
    def __init__(self, variant: str | None = None):
        variant = variant or os.getenv("RERANK_ONNX_VARIANT", "int8")
        super().__init__(RERANKER_DIR / RERANKER_FILES[variant], RERANKER_DIR / "tokenizer.json")

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Same interface as sentence-transformers CrossEncoder.predict."""
        logits = self.run(self.tokenizer.encode_batch([(q, d) for q, d in pairs]))
        return logits[:, 0].tolist()


def prepare(models_dir: Path = MODELS_DIR) -> None:
    """Download the ONNX exports and int8-quantize the embedder (build step;
    needs `onnx` + huggingface_hub, not used at runtime)."""
    from huggingface_hub import hf_hub_download
    from onnxruntime.quantization import QuantType, quantize_dynamic

    emb = models_dir / EMBEDDER_DIR.name
    fp32 = hf_hub_download("BAAI/bge-base-en-v1.5", EMBEDDER_FILES["fp32"], local_dir=emb)
    hf_hub_download("BAAI/bge-base-en-v1.5", "tokenizer.json", local_dir=emb)
    # per-channel, MatMul weights only (the word-embedding table stays fp32):
    # cosine 0.990 to the fp32 model vs 0.967 for default whole-model int8
    quantize_dynamic(fp32, str(emb / EMBEDDER_FILES["int8"]), op_types_to_quantize=["MatMul"],
                     per_channel=True, weight_type=QuantType.QUInt8)

    rr = models_dir / RERANKER_DIR.name
    for f in RERANKER_FILES.values():
        hf_hub_download("Xenova/ms-marco-MiniLM-L-6-v2", f, local_dir=rr)
    hf_hub_download("Xenova/ms-marco-MiniLM-L-6-v2", "tokenizer.json", local_dir=rr)
    print(f"✓ ONNX models ready in {models_dir}")


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["prepare"]:
        prepare()
