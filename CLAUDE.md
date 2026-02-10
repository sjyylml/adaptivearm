# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

**AdaptiveArm** is a Python framework for sensorless force sensing and adaptive control on collaborative robot arms. It enables force estimation, load compensation, and compliant control using only motor currents and joint encoders — no force/torque sensors required.

**Business model**: Open-source core (MIT) + paid services (AutoTuner, system identification, web monitoring).

## Tech Stack

- **Language**: Python 3.10+
- **Build**: setuptools with pyproject.toml
- **Physics**: MuJoCo (dynamics computation + simulation)
- **Core deps**: numpy, scipy, mujoco
- **Dev tools**: pytest, ruff, mypy
- **Optional**: matplotlib (visualization)

## Build & Run Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run specific test
pytest tests/test_momentum_observer.py -v

# Lint
ruff check src/

# Type check
mypy src/

# Run examples
python examples/01_quickstart_simulation.py
python examples/02_momentum_observer_demo.py
```

## Architecture

The project uses an **Adapter pattern**. All algorithms interact with robots through the `RobotInterface` protocol, making them hardware-agnostic.

### Key data flow
```
Robot → RobotState → Observer → ObserverOutput → Controller → ControlOutput → Robot
```

### Module structure
- `core/` — Protocol definitions (`RobotInterface`, `DynamicsModel`), data types (`RobotState`, `ObserverOutput`, `ControlOutput`), configuration
- `dynamics/` — Rigid-body dynamics via MuJoCo (`MuJoCoDynamics`), friction models
- `estimation/` — Force/torque observers. `MomentumObserver` (GMO) is the primary algorithm
- `sim/` — MuJoCo simulation environment, virtual force sensors for validation
- `adapters/` — Hardware-specific implementations of `RobotInterface` (currently: `SimAdapter`)
- `control/` — Compliant controllers (Phase 2+)
- `identification/` — Parameter identification (Phase 2+)
- `utils/` — Math helpers, signal processing (low-pass filter)

### Key algorithm: Generalized Momentum Observer (GMO)
```
p(t) = M(q) · q̇
r(t) = K_O · ∫[τ_motor - C^T·q̇ - g(q) + r] dt - K_O · p(t)
```
Residual `r(t)` converges to external torque `τ_ext`. Implementation uses trapezoidal integration.

## Conventions

- Type hints required on all public functions
- Use `NDArray[np.floating]` for array type hints
- Protocols over abstract base classes for interface definitions
- Tests use pytest fixtures defined in `tests/conftest.py`
- Observer gains are diagonal matrices stored as 1D arrays
- All torques in Nm, forces in N, angles in radians, time in seconds

## Roadmap

- **Phase 1** (current): Core framework + GMO + MuJoCo simulation
- **Phase 2**: YAM hardware adapter + impedance/admittance control + collision detection
- **Phase 3**: UR adapter + Pinocchio backend + EKF observer
- **Phase 4**: PINN/Transformer estimators + paid features (AutoTuner, web dashboard)
