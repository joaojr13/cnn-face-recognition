"""Monta o repositório PRIVADO de deploy: código + modelo ativo + avaliação + fotos.

O repositório público continua sem fotos e sem pesos (.gitignore). Este script copia
para uma pasta separada tudo o que o Streamlit precisa para reconhecer rostos com o
limiar calibrado, confere a consistência e cria um commit nessa pasta.

Uso:
  .venv/bin/python scripts/preparar_deploy.py                       # ../ap-02-deploy-privado
  .venv/bin/python scripts/preparar_deploy.py --remote URL --push   # também envia ao GitHub

O destino deve ser enviado SOMENTE a um repositório privado: contém fotos dos integrantes.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = ROOT.parent / "ap-02-deploy-privado"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
GITHUB_LIMIT = 100 * 1024 * 1024
GENERATED = {Path(".gitignore"), Path("DEPLOY_PRIVADO.md")}
GITIGNORE = """.venv/
.python/
.cache/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.DS_Store
output/
.streamlit/secrets.toml
.env
.env.*
"""


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def git(*args, cwd, check=True):
    return subprocess.run(["git", *args], cwd=cwd, check=check, text=True, capture_output=True)


def code_files(root=ROOT):
    """Arquivos do repositório público (versionados ou novos, nunca os ignorados)."""
    listed = git("ls-files", "--cached", "--others", "--exclude-standard", "-z", cwd=root).stdout
    return sorted(Path(item) for item in listed.split("\0") if item and (root / item).is_file())


def active_run(root=ROOT, allow_provisional=False):
    """Confere modelo, limiar e avaliação da execução ativa antes de copiar."""
    pointer = root / "artifacts" / "active.json"
    if not pointer.is_file():
        raise ValueError("Não há modelo ativo (artifacts/active.json).")
    active = json.loads(pointer.read_text())
    run_id = active["run_id"]
    run = root / "artifacts" / run_id
    model, metadata_path = run / "model.pt", run / "metadata.json"
    for path in (model, metadata_path):
        if not path.is_file():
            raise ValueError(f"Arquivo do modelo ausente: {path.relative_to(root)}")
    if model.stat().st_size < 1024 * 1024:
        raise ValueError("model.pt tem menos de 1 MB: parece um ponteiro, não os pesos.")
    if model.stat().st_size >= GITHUB_LIMIT:
        raise ValueError("model.pt passa de 100 MB, o limite de arquivo do GitHub.")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("run_id") != run_id or metadata.get("threshold") != active.get("threshold"):
        raise ValueError("artifacts/active.json e metadata.json da execução divergem. Reative o modelo.")
    report = root / "reports" / run_id
    evaluation_path = report / "evaluation.json"
    if not allow_provisional:
        if not active.get("threshold_calibrated"):
            raise ValueError("O limiar do modelo ativo ainda é provisório. Execute "
                             "`python -m rostos.avaliacao_facial --aplicar-limiar` antes do deploy.")
        if not evaluation_path.is_file():
            raise ValueError(f"Avaliação ausente: {evaluation_path.relative_to(root)}")
        evaluation = json.loads(evaluation_path.read_text())
        if evaluation.get("run_id") != run_id or abs(evaluation["threshold"] - active["threshold"]) > 1e-12:
            raise ValueError("evaluation.json não corresponde ao modelo e ao limiar ativos.")
    return active, run, report


def private_files(root, run, report):
    """Arquivos fora do repositório público que o app usa em produção."""
    files = [Path("artifacts/active.json")]
    files += [path.relative_to(root) for folder in (run, report) for path in sorted(folder.rglob("*"))
              if path.is_file() and not path.name.startswith(".")]
    for folder in ("data/raw", "data/unknown"):
        base = root / folder
        if base.exists():
            files += [path.relative_to(root) for path in sorted(base.rglob("*")) if path.is_file()
                      and (path.suffix.lower() in IMAGE_EXTENSIONS or path.name == "manifest.json")]
    return files


def sync(root, destination, files):
    """Espelha a lista no destino: copia o que mudou e remove o que saiu (exceto .git)."""
    wanted = {Path(item) for item in files}
    copied = 0
    for relative in sorted(wanted):
        source, target = root / relative, destination / relative
        if target.is_file() and target.stat().st_size == source.stat().st_size and sha256(target) == sha256(source):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1
    removed = 0
    for path in sorted(destination.rglob("*"), reverse=True):
        relative = path.relative_to(destination)
        if relative.parts[0] == ".git" or relative in wanted or relative in GENERATED:
            continue
        if path.is_file() or path.is_symlink():
            path.unlink()
            removed += 1
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    return copied, removed


def deploy_note(active, run, files, root):
    model = run / "model.pt"
    size = sum((root / item).stat().st_size for item in files)
    return f"""# Deploy privado · AP02 reconhecimento de rostos

**Mantenha este repositório PRIVADO.** Ele contém fotos dos integrantes e o modelo treinado.
Gerado por `scripts/preparar_deploy.py` na pasta do projeto; não edite os arquivos aqui.

| Item | Valor |
|---|---|
| Modelo ativo | `{active['run_id']}` |
| Limiar | {active['threshold']:.4f} ({'calibrado' if active.get('threshold_calibrated') else 'provisório'}) |
| Limiar provisório anterior | {active.get('provisional_threshold', '—')} |
| SHA256 de model.pt | `{sha256(model)}` |
| Arquivos | {len(files)} ({size / 1024 / 1024:.1f} MB) |

## Streamlit Community Cloud

1. Em **Settings → Linked accounts**, autorize o acesso a repositórios privados do GitHub.
2. **Create app** → este repositório, branch `main`, arquivo `app.py`.
3. Em **Advanced settings**, escolha **Python 3.11**.
4. Depois do deploy, confira na barra lateral: "Limiar: {active['threshold']:.1%} (calibrado na validação)".
5. Em **Settings → Sharing**, restrinja quem pode ver o app, se necessário.

Para atualizar, rode o script de novo na pasta do projeto e envie com `git push`.
"""


def github_repository(url):
    """owner/repo de uma URL do GitHub (https ou ssh); None para outros servidores."""
    match = re.match(r"^(?:https://github\.com/|git@github\.com:)([^/]+)/([^/]+?)(?:\.git)?/?$", url.strip())
    return (match.group(1), match.group(2)) if match else None


def github_visibility(url, timeout=10):
    """Consulta anônima: um repositório público responde 200; privado ou inexistente, 404."""
    repository = github_repository(url)
    if repository is None:
        return "unknown"
    request = urllib.request.Request(f"https://api.github.com/repos/{repository[0]}/{repository[1]}",
                                     headers={"Accept": "application/vnd.github+json",
                                              "User-Agent": "ap02-preparar-deploy"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return "private" if json.load(response).get("private") else "public"
    except urllib.error.HTTPError as exc:
        return "private_or_missing" if exc.code == 404 else "unknown"
    except (urllib.error.URLError, OSError, ValueError):
        return "unknown"


def stack_on_remote(destination, message):
    """Põe o pacote por cima do histórico remoto, sem force push.

    Usado quando o repositório foi criado com README/licença. Retorna os arquivos
    que só existiam no remoto e foram substituídos (continuam no histórico).
    """
    heads = subprocess.run(["git", "ls-remote", "--heads", "origin", "main"], cwd=destination,
                           text=True, stdout=subprocess.PIPE)
    if heads.returncode != 0:
        raise RuntimeError("Não foi possível ler o repositório remoto (acesso ou URL).")
    if not heads.stdout.strip():
        return []  # remoto vazio: o envio normal funciona
    if subprocess.run(["git", "fetch", "origin", "main"], cwd=destination).returncode != 0:
        raise RuntimeError("Não foi possível baixar o histórico remoto.")
    if git("merge-base", "--is-ancestor", "origin/main", "HEAD", cwd=destination, check=False).returncode == 0:
        return []  # já está por cima do remoto
    listed = lambda ref: set(filter(None, git("ls-tree", "-r", "--name-only", "-z", ref,
                                              cwd=destination).stdout.split("\0")))
    replaced = sorted(listed("origin/main") - listed("HEAD"))
    git("reset", "--soft", "origin/main", cwd=destination)
    if git("diff", "--cached", "--quiet", cwd=destination, check=False).returncode != 0:
        git("commit", "-m", message, cwd=destination)
    return replaced


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--destino", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--remote", help="URL do repositório PRIVADO no GitHub (define o origin).")
    parser.add_argument("--push", action="store_true", help="Envia o commit ao origin.")
    parser.add_argument("--permitir-provisorio", action="store_true",
                        help="Permite deploy sem limiar calibrado (não recomendado).")
    parser.add_argument("--substituir-remoto", action="store_true",
                        help="Se o remoto já tiver commits (README do GitHub), põe o pacote por cima, sem force push.")
    parser.add_argument("--permitir-publico", action="store_true",
                        help="Envia mesmo sem confirmar que o repositório é privado (não recomendado).")
    args = parser.parse_args()
    destination = args.destino.expanduser().resolve()
    if destination == ROOT or destination in ROOT.parents:
        parser.error("O destino deve ser uma pasta separada do projeto.")
    try:
        active, run, report = active_run(ROOT, args.permitir_provisorio)
    except (ValueError, KeyError) as exc:
        parser.exit(2, f"Deploy não preparado: {exc}\n")

    # O .gitignore público excluiria fotos e pesos; o destino recebe o seu próprio.
    files = [path for path in code_files() if path != Path(".gitignore")] + private_files(ROOT, run, report)
    destination.mkdir(parents=True, exist_ok=True)
    copied, removed = sync(ROOT, destination, files)
    (destination / ".gitignore").write_text(GITIGNORE)
    (destination / "DEPLOY_PRIVADO.md").write_text(deploy_note(active, run, files, ROOT))

    if not (destination / ".git").exists():
        git("init", "-b", "main", cwd=destination)
    if args.remote:
        has_origin = git("remote", "get-url", "origin", cwd=destination, check=False).returncode == 0
        git("remote", "set-url" if has_origin else "add", "origin", args.remote, cwd=destination)
    git("add", "-A", cwd=destination)
    changed = git("diff", "--cached", "--quiet", cwd=destination, check=False).returncode != 0
    message = f"Deploy {active['run_id']} · limiar {active['threshold']:.4f}"
    if changed:
        result = git("commit", "-m", message, cwd=destination, check=False)
        if result.returncode != 0:
            parser.exit(1, f"Arquivos copiados, mas o commit falhou:\n{result.stderr}\n")

    print(f"Destino: {destination}")
    print(f"Modelo {active['run_id']} · limiar {active['threshold']:.4f}"
          f" ({'calibrado' if active.get('threshold_calibrated') else 'PROVISÓRIO'})")
    print(f"{len(files)} arquivos · {copied} copiados/atualizados · {removed} removidos · "
          f"{'novo commit criado' if changed else 'nada mudou desde o último commit'}")
    origin = git("remote", "get-url", "origin", cwd=destination, check=False)
    if args.push:
        if origin.returncode != 0:
            parser.exit(2, "Sem origin. Use --remote com a URL do repositório privado.\n")
        url = origin.stdout.strip()
        visibility = github_visibility(url)
        if visibility == "public" and not args.permitir_publico:
            parser.exit(3, f"ENVIO CANCELADO: {url} é PÚBLICO e o pacote contém fotos dos integrantes.\n"
                           "Torne o repositório privado no GitHub (Settings → General → Danger Zone) e rode de novo.\n")
        if visibility == "unknown" and not args.permitir_publico:
            parser.exit(3, f"ENVIO CANCELADO: não foi possível confirmar que {url} é privado.\n"
                           "Confira no GitHub e, se for privado, rode de novo com --permitir-publico.\n")
        print("Repositório confirmado como privado." if visibility != "public" else
              "Atenção: enviando para repositório público (--permitir-publico).")
        if args.substituir_remoto:
            try:
                replaced = stack_on_remote(destination, message)
            except (RuntimeError, subprocess.CalledProcessError) as exc:
                parser.exit(1, f"{exc}\n")
            if replaced:
                print("Arquivos que só existiam no remoto, substituídos (continuam no histórico): "
                      + ", ".join(replaced))
        # Envio interativo: o git pode pedir usuário e token no Terminal.
        result = subprocess.run(["git", "push", "-u", "origin", "main"], cwd=destination)
        if result.returncode != 0:
            print("\nO envio falhou. Se o GitHub recusou por já haver commits no repositório "
                  "(por exemplo, um README criado junto), rode de novo com --substituir-remoto.")
        sys.exit(result.returncode)
    if origin.returncode != 0:
        print("\nPróximo passo: crie um repositório PRIVADO vazio no GitHub e rode:\n"
              "  .venv/bin/python scripts/preparar_deploy.py --remote <URL> --push")
    else:
        print(f"\nPara enviar: cd {destination} && git push")


if __name__ == "__main__":
    main()
