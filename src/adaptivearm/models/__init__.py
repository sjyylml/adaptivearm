"""Robot model registry for multi-arm support.

Provides a simple registry that maps robot names to their model files
and metadata. Users can register custom models or use built-in ones.

Usage::

    from adaptivearm.models import get_model, list_models, register_model

    # List available models
    print(list_models())  # ['default_6dof']

    # Get model info
    info = get_model("default_6dof")
    print(info.model_path, info.n_joints)

    # Register a custom model
    register_model(RobotModelInfo(
        name="my_robot",
        model_path=Path("/path/to/robot.urdf"),
        n_joints=7,
        ee_site_name="ee_site",
        ee_body_name="ee_link",
    ))
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RobotModelInfo:
    """Metadata for a robot model.

    Attributes:
        name: Registry name (e.g. "default_6dof", "ur5e").
        model_path: Path to the MJCF (.xml) or URDF (.urdf) file.
        n_joints: Number of actuated joints.
        ee_site_name: MuJoCo site name for the end-effector.
        ee_body_name: MuJoCo body name for the end-effector.
        description: Short human-readable description.
    """

    name: str
    model_path: Path
    n_joints: int
    ee_site_name: str
    ee_body_name: str
    description: str = ""


# Global model registry
_registry: dict[str, RobotModelInfo] = {}


def register_model(info: RobotModelInfo) -> None:
    """Register a robot model.

    Args:
        info: Model metadata. The ``name`` field is used as the registry key.
    """
    _registry[info.name] = info


def get_model(name: str) -> RobotModelInfo:
    """Look up a registered model by name.

    Args:
        name: Registry name (e.g. "default_6dof").

    Returns:
        The model metadata.

    Raises:
        KeyError: If the model name is not registered.
    """
    if name not in _registry:
        available = ", ".join(sorted(_registry.keys())) or "(none)"
        raise KeyError(
            f"Unknown model '{name}'. Available models: {available}"
        )
    return _registry[name]


def list_models() -> list[str]:
    """List all registered model names.

    Returns:
        Sorted list of registered model names.
    """
    return sorted(_registry.keys())


# ---------------------------------------------------------------------------
# Built-in model registration
# ---------------------------------------------------------------------------

_MODELS_DIR = Path(__file__).parent

register_model(RobotModelInfo(
    name="default_6dof",
    model_path=_MODELS_DIR / "default_6dof" / "robot.xml",
    n_joints=6,
    ee_site_name="ee_site",
    ee_body_name="ee",
    description="Built-in 6-DOF serial manipulator for prototyping",
))

__all__ = [
    "RobotModelInfo",
    "get_model",
    "list_models",
    "register_model",
]
