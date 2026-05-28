import os
import sys


def ensure_local_diff_gaussian_rasterization():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_pkg_root = os.path.join(repo_root, "submodules", "diff-gaussian-rasterization")
    if not os.path.isdir(local_pkg_root):
        return
    if local_pkg_root in sys.path:
        return
    sys.path.insert(0, local_pkg_root)



