"""Preparação do repositório privado de deploy (sem git e sem rede)."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("preparar_deploy", ROOT / "scripts" / "preparar_deploy.py")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


def project(tmp_path, calibrated=True, evaluation_run="demo", threshold=.95):
    root = tmp_path / "projeto"
    run = root / "artifacts" / "demo"
    report = root / "reports" / "demo"
    run.mkdir(parents=True)
    report.mkdir(parents=True)
    metadata = {"run_id": "demo", "threshold": threshold, "threshold_calibrated": calibrated}
    (root / "artifacts" / "active.json").write_text(json.dumps(metadata))
    (run / "metadata.json").write_text(json.dumps(metadata))
    (run / "model.pt").write_bytes(b"0" * (2 * 1024 * 1024))
    (report / "evaluation.json").write_text(json.dumps({"run_id": evaluation_run, "threshold": threshold}))
    photo = root / "data" / "raw" / "ana" / "s1" / "a.jpg"
    photo.parent.mkdir(parents=True)
    photo.write_bytes(b"jpeg")
    (root / "data" / "test" / "ana").mkdir(parents=True)
    (root / "data" / "test" / "ana" / "nova.jpg").write_bytes(b"jpeg")
    return root


def test_active_run_requires_calibrated_threshold_and_matching_evaluation(tmp_path):
    root = project(tmp_path)
    active, run, report = deploy.active_run(root)
    assert active["threshold"] == .95 and run.name == report.name == "demo"
    with pytest.raises(ValueError, match="provisório"):
        deploy.active_run(project(tmp_path / "a", calibrated=False))
    assert deploy.active_run(project(tmp_path / "b", calibrated=False), allow_provisional=True)
    with pytest.raises(ValueError, match="não corresponde"):
        deploy.active_run(project(tmp_path / "c", evaluation_run="outro"))


def test_pointer_instead_of_weights_is_rejected(tmp_path):
    root = project(tmp_path)
    (root / "artifacts" / "demo" / "model.pt").write_text("version https://git-lfs.github.com/spec/v1")
    with pytest.raises(ValueError, match="ponteiro"):
        deploy.active_run(root)


def test_private_files_include_model_report_and_training_photos_but_not_test_photos(tmp_path):
    root = project(tmp_path)
    _, run, report = deploy.active_run(root)
    files = {str(path) for path in deploy.private_files(root, run, report)}
    assert {"artifacts/active.json", "artifacts/demo/model.pt", "artifacts/demo/metadata.json",
            "reports/demo/evaluation.json", "data/raw/ana/s1/a.jpg"} <= files
    assert not any(name.startswith("data/test") for name in files)


def test_sync_mirrors_files_and_keeps_git_and_generated_files(tmp_path):
    root = project(tmp_path)
    destination = tmp_path / "deploy"
    files = [Path("artifacts/active.json"), Path("data/raw/ana/s1/a.jpg")]
    assert deploy.sync(root, destination, files) == (2, 0)
    assert deploy.sync(root, destination, files) == (0, 0)
    (destination / ".git").mkdir()
    (destination / ".git" / "HEAD").write_text("ref")
    (destination / "DEPLOY_PRIVADO.md").write_text("nota")
    (destination / "artifacts" / "antigo").mkdir()
    (destination / "artifacts" / "antigo" / "model.pt").write_text("velho")
    assert deploy.sync(root, destination, files[:1]) == (0, 2)
    assert not (destination / "data" / "raw").exists()
    assert not (destination / "artifacts" / "antigo").exists()
    assert (destination / ".git" / "HEAD").exists() and (destination / "DEPLOY_PRIVADO.md").exists()


def test_github_url_parsing():
    assert deploy.github_repository("https://github.com/joaojr13/cnn-face-recognition-config.git") == \
        ("joaojr13", "cnn-face-recognition-config")
    assert deploy.github_repository("git@github.com:joaojr13/repo.git") == ("joaojr13", "repo")
    assert deploy.github_repository("https://gitlab.com/a/b.git") is None


def test_public_repository_is_detected_before_push(monkeypatch):
    import io
    import urllib.error

    def answer(body=None, code=None):
        def fake(request, timeout):
            if code:
                raise urllib.error.HTTPError(request.full_url, code, "x", {}, io.BytesIO())
            return io.BytesIO(json.dumps(body).encode())
        return fake

    url = "https://github.com/a/b.git"
    monkeypatch.setattr(deploy.urllib.request, "urlopen", answer({"private": False}))
    assert deploy.github_visibility(url) == "public"
    monkeypatch.setattr(deploy.urllib.request, "urlopen", answer(code=404))
    assert deploy.github_visibility(url) == "private_or_missing"
    monkeypatch.setattr(deploy.urllib.request, "urlopen", answer(code=500))
    assert deploy.github_visibility(url) == "unknown"


def test_package_is_stacked_on_remote_created_with_readme_without_force_push(tmp_path):
    import subprocess

    def run(*args, cwd):
        return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd,
                              check=True, text=True, capture_output=True).stdout

    remote, seed, package = tmp_path / "remoto.git", tmp_path / "seed", tmp_path / "pacote"
    run("init", "--bare", "-b", "main", str(remote), cwd=tmp_path)
    run("init", "-b", "main", str(seed), cwd=tmp_path)
    (seed / "README.md").write_text("# criado pelo GitHub")
    (seed / "LICENSE").write_text("MIT")
    run("add", "-A", cwd=seed)
    run("commit", "-m", "Initial commit", cwd=seed)
    run("push", str(remote), "main", cwd=seed)

    run("init", "-b", "main", str(package), cwd=tmp_path)
    (package / "README.md").write_text("# projeto")
    (package / "app.py").write_text("print(1)")
    run("add", "-A", cwd=package)
    run("commit", "-m", "Deploy", cwd=package)
    run("remote", "add", "origin", str(remote), cwd=package)
    import os
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    old = os.environ.copy()
    os.environ.update(env)
    try:
        assert deploy.stack_on_remote(package, "Deploy") == ["LICENSE"]
        assert deploy.stack_on_remote(package, "Deploy") == []
    finally:
        os.environ.clear()
        os.environ.update(old)
    run("push", "origin", "main", cwd=package)  # avanço normal, sem --force
    assert run("log", "--format=%s", "main", cwd=remote).split() == ["Deploy", "Initial", "commit"]
    assert sorted(run("ls-tree", "-r", "--name-only", "main", cwd=remote).split()) == ["README.md", "app.py"]
    assert run("show", "main:README.md", cwd=remote) == "# projeto"
