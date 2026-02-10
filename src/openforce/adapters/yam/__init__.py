"""i2rt YAM robot arm adapter."""

try:
    from openforce.adapters.yam.yam_adapter import YAMAdapter, YAMConfig

    __all__ = ["YAMAdapter", "YAMConfig"]
except ImportError:
    __all__: list[str] = []  # type: ignore[no-redef]
