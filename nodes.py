"""ComfyUI registration shim.

Each node implementation lives in its own module. ComfyUI still imports this
file as the package entry point, so keep it small and only expose mappings here.
"""

try:
    from .pid_decode import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:  # Allows `python -c "import nodes"` from this folder.
    from pid_decode import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
