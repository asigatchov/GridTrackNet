import ctypes
import os
import site
import sys
from pathlib import Path


os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")


def _nvidia_lib_dirs() -> list[Path]:
    dirs: list[Path] = []
    search_roots = []
    search_roots.extend(Path(path) for path in sys.path if "site-packages" in path)
    search_roots.extend(Path(path) for path in site.getsitepackages())
    search_roots.append(Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages")

    seen = set()
    for root in search_roots:
        if root in seen:
            continue
        seen.add(root)
        base = root / "nvidia"
        if not base.exists():
            continue
        for lib_dir in sorted(base.glob("*/lib")):
            if lib_dir.is_dir():
                dirs.append(lib_dir)
    return dirs


def _prepend_library_path(lib_dirs: list[Path]) -> None:
    current = [entry for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":") if entry]
    additions = [str(path) for path in lib_dirs if str(path) not in current]
    if additions:
        os.environ["LD_LIBRARY_PATH"] = ":".join(additions + current)


def _preload_cuda_libs(lib_dirs: list[Path]) -> None:
    preload_order = (
        "libcudart.so.12",
        "libcublas.so.12",
        "libcublasLt.so.12",
        "libcufft.so.11",
        "libcurand.so.10",
        "libcusparse.so.12",
        "libcusolver.so.11",
        "libcudnn.so.9",
    )
    for lib_name in preload_order:
        for lib_dir in lib_dirs:
            lib_path = lib_dir / lib_name
            if lib_path.exists():
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                break


_LIB_DIRS = _nvidia_lib_dirs()
if _LIB_DIRS:
    _prepend_library_path(_LIB_DIRS)
    _preload_cuda_libs(_LIB_DIRS)
