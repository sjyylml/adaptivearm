"""Force/torque estimation algorithms."""

from openforce.estimation.base_observer import BaseObserver
from openforce.estimation.collision_detector import (
    CollisionDetector,
    CollisionEvent,
    CollisionReaction,
)
from openforce.estimation.composite_observer import (
    CompositeObserver,
    FusionStrategy,
)
from openforce.estimation.ekf_observer import EKFObserver, EKFParams
from openforce.estimation.momentum_observer import MomentumObserver
from openforce.estimation.neural_base import NeuralObserver, ObserverMode, TrainingData

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
    from openforce.estimation.pinn_observer import PINNObserver, PINNParams
    from openforce.estimation.transformer_observer import (
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
