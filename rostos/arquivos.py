"""Caminhos portáteis em relatórios; resolução relativa à raiz do projeto."""
import json
from pathlib import Path
from .config import ROOT

def resolve_record_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()

def portable(value):
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(item) for item in value]
    if isinstance(value, str) and value.startswith(str(ROOT) + "/"):
        return str(Path(value).relative_to(ROOT))
    return value

def write_report(path, value):
    path.write_text(json.dumps(portable(value), ensure_ascii=False, indent=2) + "\n")
