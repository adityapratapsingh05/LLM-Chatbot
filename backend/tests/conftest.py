"""
conftest.py – runs before any test module is imported.

LITELLM_LOCAL_MODEL_COST_MAP=1 tells LiteLLM to skip the network fetch of the
model-cost map on startup. Without it, `import litellm` hangs indefinitely on
Windows (and in CI environments with strict outbound rules) while it tries to
download the cost map from GitHub.

This must be set before `import litellm` is executed anywhere in the process,
so it lives here at the top of conftest rather than inside a fixture.
"""
import os

# Must be set before litellm is imported anywhere in the test process.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "1")
