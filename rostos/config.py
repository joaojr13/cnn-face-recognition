import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
IMAGE_SIZE = 160
EXPECTED_PEOPLE = 5
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_PIXELS = 24_000_000
MAX_SIDE = 1600
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Os pesos e caches ficam no projeto. Nenhuma imagem é enviada a um serviço.
os.environ.setdefault("KERAS_HOME", str(ROOT / ".cache" / "keras"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

for directory in (DATA / "raw", DATA / "unknown", ARTIFACTS, REPORTS):
    directory.mkdir(parents=True, exist_ok=True)
