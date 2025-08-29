import os
import glob

from torch.utils.cpp_extension import load


def build_extension(verbose=True):
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))

    sources = []
    sources += list(
        glob.glob(os.path.join(CURRENT_DIR, "csrc", "**/*.cu"), recursive=True)
    )
    sources += list(
        glob.glob(os.path.join(CURRENT_DIR, "csrc", "**/*.cpp"), recursive=True)
    )

    extra_include_paths = [os.path.join(REPO_ROOT, "include")]
    extra_cflags = ["-O3"]
    extra_cuda_cflags = ["-O3", "-use_fast_math", "--extended-lambda"]

    if verbose:
        print("Building cugsplat extension...")
        print("Sources:")
        for source in sources:
            print(f"  - {source}")
        print("Extra include paths:")
        for path in extra_include_paths:
            print(f"  - {path}")
        print("Extra C flags:")
        print(f"  - {extra_cflags}")
        print("Extra CUDA flags:")
        print(f"  - {extra_cuda_cflags}")

    return load(
        name="cugsplat",
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_include_paths=extra_include_paths,
        verbose=verbose,
    )


_C = build_extension()
