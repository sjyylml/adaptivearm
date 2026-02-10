"""Universal Robots (UR) adapter via RTDE."""

try:
    from adaptivearm.adapters.ur.ur_adapter import URAdapter, URConfig

    __all__ = ["URAdapter", "URConfig"]
except ImportError:
    __all__: list[str] = []  # type: ignore[no-redef]
