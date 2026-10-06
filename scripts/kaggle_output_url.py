"""Print signed download URLs of a Kaggle kernel's output files (one "name<TAB>url" per line), so large files can be
streamed with `curl -C -` instead of the Kaggle CLI (which was OOM-killed on a 3.9 GB .litertlm in 7.8 GB WSL).

  ~/.local/share/uv/tools/kaggle/bin/python scripts/kaggle_output_url.py harsh11067/skinnova-gemma4-e2b-export '\\.litertlm$'
"""
import re
import sys

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

kernel, pattern = sys.argv[1], re.compile(sys.argv[2] if len(sys.argv) > 2 else ".")
owner, slug = kernel.split("/")
api = KaggleApi(); api.authenticate()
token = None
with api.build_kaggle_client() as kaggle:
    while True:
        req = ApiListKernelSessionOutputRequest(); req.user_name = owner; req.kernel_slug = slug
        if token:
            req.page_token = token
        resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
        for item in resp.files or []:
            if pattern.search(item.file_name):
                print(f"{item.file_name}\t{item.url}")
        token = resp.next_page_token
        if not token:
            break
