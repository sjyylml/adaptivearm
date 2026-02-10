"""Force/torque estimation algorithms."""

from adaptivearm.estimation.base_observer import BaseObserver
from adaptivearm.estimation.collision_detector import (
    CollisionDetector,
    CollisionEvent,
    CollisionReaction,
)
from adaptivearm.estimation.composite_observer import (
    CompositeObserver,
    FusionStrategy,
)
from adaptivearm.estimation.ekf_observer import EKFObserver, EKFParams
from adaptivearm.estimation.momentum_observer import MomentumObserver
from adaptivearm.estimation.neural_base import NeuralObserver, ObserverMode, TrainingData

__all__ = [
    "BaseObserver",
    "CollisionDetector",
    "CollisionEvent",
    "CollisionReaction",
    "CompositeObserver",
    "EKFObserver",
    "EKFParams",
    "FusionStrategy",
    "MomentumObserver",
    "NeuralObserver",
    "ObserverMode",
    "TrainingData",
]

# Conditional imports for neural observers (require PyTorch)
try:
    from adaptivearm.estimation.pinn_observer import PINNObserver, PINNParams
    from adaptivearm.estimation.transformer_observer import (
        TransformerObserver,
        TransformerObserverParams,
    )

    __all__ += [
        "PINNObserver",
        "PINNParams",
        "TransformerObserver",
        "TransformerObserverParams",
    ]
except ImportError:
    pass
