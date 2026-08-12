"""Unit tests for pulling KubeVirt containerDisks out of an OCI registry."""

import gzip
import hashlib
import io
import tarfile
from types import SimpleNamespace

import pytest
from cua_sandbox.image import (
    DEFAULT_LINUX_REGISTRY_IMAGE,
    DEFAULT_WINDOWS_REGISTRY_IMAGE,
)
from cua_sandbox.registry.container_disk import (
    _LOCK_POLL_INTERVAL_SECONDS,
    pull_container_disk,
)

WINDOWS_REPOSITORY = DEFAULT_WINDOWS_REGISTRY_IMAGE.rsplit(":", 1)[0]
CHILD_DIGEST = "sha256:child"
ATTESTATION_DIGEST = "sha256:attestation"


def _layer_with_disk(contents: bytes, *, path: str = "disk/disk.img") -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        info = tarfile.TarInfo(path)
        info.size = len(contents)
        archive.addfile(info, io.BytesIO(contents))
    return gzip.compress(raw.getvalue())


def _index(*, architecture: str = "amd64") -> dict:
    """An OCI index shaped like the buildx output the real registry serves."""
    return {
        "mediaType": "application/vnd.oci.image.index.v1+json",
        "manifests": [
            {
                "digest": CHILD_DIGEST,
                "mediaType": "application/vnd.oci.image.manifest.v1+json",
                "platform": {"architecture": architecture, "os": "linux"},
            },
            {
                "digest": ATTESTATION_DIGEST,
                "mediaType": "application/vnd.oci.image.manifest.v1+json",
                "platform": {"architecture": "unknown", "os": "unknown"},
                "annotations": {"vnd.docker.reference.type": "attestation-manifest"},
            },
        ],
    }


def _manifest(*layer_digests: str) -> dict:
    return {
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "layers": [
            {"digest": digest, "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip"}
            for digest in layer_digests
        ],
    }


class FakeRegistry:
    """Records the calls a pull makes, so auth and index handling stay pinned."""

    def __init__(self, manifests: dict, blobs: dict, calls: list):
        self._manifests = manifests
        self._blobs = blobs
        self._calls = calls
        self.auth = SimpleNamespace(
            load_configs=lambda container: calls.append(("auth", container.uri))
        )

    def get_container(self, ref):
        self._calls.append(("container", ref))
        return SimpleNamespace(uri=ref)

    def get_manifest(self, ref):
        self._calls.append(("manifest", ref))
        return self._manifests[ref]

    def get_blob(self, container, digest, *, stream):
        self._calls.append(("blob", digest, stream))
        return SimpleNamespace(raw=io.BytesIO(self._blobs[digest]))


def _factory(manifests: dict, blobs: dict, calls: list):
    def build(*, auth_backend):
        calls.append(("init", auth_backend))
        return FakeRegistry(manifests, blobs, calls)

    return build


def test_pull_uses_basic_auth_and_descends_into_the_platform_manifest(tmp_path):
    """ECR rejects oras' bearer-token flow, and serves an index rather than a manifest."""
    calls = []
    manifests = {
        DEFAULT_WINDOWS_REGISTRY_IMAGE: _index(),
        f"{WINDOWS_REPOSITORY}@{CHILD_DIGEST}": _manifest("sha256:layer"),
    }
    blobs = {"sha256:layer": _layer_with_disk(b"qcow2")}

    disk = pull_container_disk(
        DEFAULT_WINDOWS_REGISTRY_IMAGE,
        cache_root=tmp_path,
        architecture="amd64",
        registry_factory=_factory(manifests, blobs, calls),
    )

    assert disk.name == "disk.qcow2"
    assert disk.read_bytes() == b"qcow2"
    assert calls == [
        ("init", "basic"),
        ("container", DEFAULT_WINDOWS_REGISTRY_IMAGE),
        ("auth", DEFAULT_WINDOWS_REGISTRY_IMAGE),
        ("manifest", DEFAULT_WINDOWS_REGISTRY_IMAGE),
        ("manifest", f"{WINDOWS_REPOSITORY}@{CHILD_DIGEST}"),
        ("blob", "sha256:layer", True),
    ]


def test_pull_reads_a_plain_manifest_without_an_index(tmp_path):
    calls = []
    manifests = {"registry.example/workspace:plain": _manifest("sha256:layer")}
    blobs = {"sha256:layer": _layer_with_disk(b"qcow2")}

    disk = pull_container_disk(
        "registry.example/workspace:plain",
        cache_root=tmp_path,
        registry_factory=_factory(manifests, blobs, calls),
    )

    assert disk.read_bytes() == b"qcow2"
    assert ("manifest", "registry.example/workspace:plain") in calls


def test_pull_rejects_an_index_without_the_requested_architecture(tmp_path):
    calls = []
    manifests = {"registry.example/workspace:arm": _index(architecture="arm64")}

    with pytest.raises(FileNotFoundError, match="no linux/amd64 manifest"):
        pull_container_disk(
            "registry.example/workspace:arm",
            cache_root=tmp_path,
            architecture="amd64",
            registry_factory=_factory(manifests, {}, calls),
        )


def test_pull_searches_every_layer_for_the_disk(tmp_path):
    calls = []
    manifests = {"registry.example/workspace:latest": _manifest("sha256:empty", "sha256:disk")}
    blobs = {
        "sha256:empty": _layer_with_disk(b"ignored", path="etc/example"),
        "sha256:disk": _layer_with_disk(b"qcow2"),
    }

    disk = pull_container_disk(
        "registry.example/workspace:latest",
        cache_root=tmp_path,
        registry_factory=_factory(manifests, blobs, calls),
    )

    assert disk.read_bytes() == b"qcow2"


def test_pull_raises_when_no_layer_carries_a_container_disk(tmp_path):
    calls = []
    manifests = {"registry.example/workspace:bare": _manifest("sha256:empty")}
    blobs = {"sha256:empty": _layer_with_disk(b"ignored", path="etc/example")}

    with pytest.raises(FileNotFoundError, match="does not contain /disk/disk.img"):
        pull_container_disk(
            "registry.example/workspace:bare",
            cache_root=tmp_path,
            registry_factory=_factory(manifests, blobs, calls),
        )


def test_second_pull_is_served_from_the_cache(tmp_path):
    calls = []
    manifests = {"registry.example/workspace:latest": _manifest("sha256:disk")}
    blobs = {"sha256:disk": _layer_with_disk(b"qcow2")}
    factory = _factory(manifests, blobs, calls)

    first = pull_container_disk(
        "registry.example/workspace:latest", cache_root=tmp_path, registry_factory=factory
    )
    calls.clear()
    second = pull_container_disk(
        "registry.example/workspace:latest", cache_root=tmp_path, registry_factory=factory
    )

    assert first == second
    assert calls == []


def test_pull_waits_for_a_concurrent_pull_to_fill_the_cache(tmp_path, monkeypatch):
    ref = DEFAULT_LINUX_REGISTRY_IMAGE
    destination = tmp_path / hashlib.sha256(ref.encode()).hexdigest() / "disk.qcow2"
    lock_path = destination.with_suffix(".lock")
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("locked")
    sleeps = []

    def registry_factory(*, auth_backend):
        raise AssertionError("registry should not be used when another process fills the cache")

    def fake_sleep(interval):
        sleeps.append(interval)
        destination.write_bytes(b"cached")
        lock_path.unlink()

    monkeypatch.setattr("cua_sandbox.registry.container_disk.time.sleep", fake_sleep)

    disk = pull_container_disk(ref, cache_root=tmp_path, registry_factory=registry_factory)

    assert disk == destination
    assert disk.read_bytes() == b"cached"
    assert sleeps == [_LOCK_POLL_INTERVAL_SECONDS]
