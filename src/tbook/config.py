"""Paths, model keys and thresholds. Everything tunable lives here."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOOKS_DIR = ROOT / "books"
BUILD_DIR = ROOT / "build"
OUT_DIR = ROOT / "out"
LEARNER_DIR = ROOT / "learner"
EVALS_DIR = ROOT / "evals"
TOOLS_DIR = ROOT / "tools"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
LLM_CACHE_PATH = BUILD_DIR / "llm_cache.sqlite"

LMSTUDIO_URL = "http://localhost:1234/v1"
LMS_EXE = Path.home() / ".cache" / "lm-studio" / "bin" / "lms.exe"

# LM Studio model keys (`lms ls --json`).
DIGEST_MODEL = "qwen/qwen3-30b-a3b-2507"
VISION_MODEL = "gemma-4-26b-a4b-it"
READER_MODEL = "qwen/qwen3-4b-2507"
EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"
EMBED_DIM = 768

DIGEST_CTX = 32768
DIGEST_PARALLEL = 4
# LM Studio shares one KV cache across the parallel slots (kv_unified): every slot must
# fit prompt (~6k with image, glyph layer and a previous transcription) + 8000 output.
VISION_CTX = 65536
VISION_PARALLEL = 4
READER_CTX = 6144  # mirrors the phone budget in the plan (§3.2)

# Extraction acceptance targets (plan: Stage 1).
PAGE_DPI = 200
COVERAGE_MIN = 0.995
REPAIR_ATTEMPTS = 3

# Segmentation.
UNIT_MIN_WORDS = 300
UNIT_MAX_WORDS = 500
CHUNK_TOKENS = 250

# Reader.
QA_MATCH_MIN = 0.86
RETRIEVE_K = 6
RETRIEVE_TOKEN_BUDGET = 2500
SEMANTIC_CANDIDATES = 15

# Kill gate: fixed before running.
MAX_LEAK_RATE = 0.05
