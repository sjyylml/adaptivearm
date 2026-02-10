#!/usr/bin/env python3
"""YAM hardware test: step-by-step validation for real robot.

Run this script to verify that the AdaptiveArm framework works correctly
with the i2rt YAM arm.  It proceeds through several stages, pausing
between each so you can observe the arm's behavior.

Prerequisites:
    1. CAN interface is up:  sudo ip link set can0 up type can bitrate 1000000
    2. i2rt SDK installed:   pip install -e /path/to/i2rt
    3. adaptivearm installed: pip install -e ".[dev]"
    4. Motor timeout disabled (optional but recommended for testing):
       python -m i2rt.motor_config_tool.set_timeout --channel can0

Usage:
    python examples/12_yam_hardware_test.py
    python examples/12_yam_hardware_test.py --channel can0 --skip-to 3
"""

from __future__ import annotations

import argparse
import signal
import sys
import time

import numpy as np


def input_continue(msg: str = "Press Enter to continue (Ctrl+C to abort)...") -> None:
    """Wait for user confirmation."""
    try:
        input(f"\n>>> {msg}")
    except (KeyboardInterrupt, EOFError):
        print("\nAborted by user.")
        sys.exit(0)


def print_header(stage: int, title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Stage {stage}: {title}")
    print(f"{'='*60}")


def stage1_connection(channel: str) -> None:
    """Test basic connection and state reading."""
    print_header(1, "Connection & State Reading")

    from adaptivearm.adapters.yam import YAMAdapter, YAMConfig

    config = YAMConfig(channel=channel, zero_gravity_mode=True)
    print(f"Connecting to YAM on {channel} ...")

    with YAMAdapter(config) as yam:
        print(f"  Connected! n_joints={yam.n_joints}, dt={yam.dt}")

        state = yam.get_state()
        print(f"\n  Joint positions (rad):  {np.array2string(state.q, precision=3)}")
        print(f"  Joint velocities (r/s): {np.array2string(state.qd, precision=3)}")
        print(f"  Motor torques (Nm):     {np.array2string(state.tau_motor, precision=3)}")
        print(f"  Timestamp: {state.timestamp:.3f}")

        temps = yam.get_temperatures()
        print(f"\n  MOS temp (°C):   {np.array2string(temps['temp_mos'], precision=1)}")
        print(f"  Rotor temp (°C): {np.array2string(temps['temp_rotor'], precision=1)}")

        # Read state at ~50 Hz for 2 seconds
        print("\n  Reading state at 50 Hz for 2 seconds...")
        t0 = time.time()
        count = 0
        while time.time() - t0 < 2.0:
            _ = yam.get_state()
            count += 1
            time.sleep(0.02)
        print(f"  Read {count} samples in 2s ({count/2.0:.0f} Hz)")

    print("\n  [PASS] Connection and state reading OK.")


def stage2_model_registry() -> None:
    """Verify YAM model is registered and loadable."""
    print_header(2, "Model Registry & MuJoCo Dynamics")

    from adaptivearm.dynamics import MuJoCoDynamics
    from adaptivearm.models import get_model, list_models

    print(f"  Registered models: {list_models()}")

    info = get_model("yam")
    print(f"\n  YAM model info:")
    print(f"    name: {info.name}")
    print(f"    path: {info.model_path}")
    print(f"    exists: {info.model_path.exists()}")
    print(f"    n_joints: {info.n_joints}")
    print(f"    ee_site: {info.ee_site_name}")
    print(f"    ee_body: {info.ee_body_name}")

    # Load dynamics from the YAM model
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(info.model_path))
    dynamics = MuJoCoDynamics(model)

    q_test = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    g = dynamics.gravity_vector(q_test)
    M = dynamics.mass_matrix(q_test)

    print(f"\n  Gravity vector at q={q_test}:")
    print(f"    g = {np.array2string(g, precision=3)}")
    print(f"    |g| = {np.linalg.norm(g):.3f} Nm")
    print(f"  Mass matrix shape: {M.shape}")
    print(f"  Mass matrix eigenvalues: {np.array2string(np.linalg.eigvalsh(M), precision=4)}")
    assert np.all(np.linalg.eigvalsh(M) > 0), "Mass matrix should be positive definite"

    print("\n  [PASS] Model registry and dynamics OK.")


def stage3_gravity_comp_live(channel: str) -> None:
    """Live gravity compensation test — compare framework dynamics to i2rt."""
    print_header(3, "Live Gravity Compensation Comparison")

    import mujoco

    from adaptivearm.adapters.yam import YAMAdapter, YAMConfig
    from adaptivearm.dynamics import MuJoCoDynamics
    from adaptivearm.models import get_model

    info = get_model("yam")
    model = mujoco.MjModel.from_xml_path(str(info.model_path))
    dynamics = MuJoCoDynamics(model)

    config = YAMConfig(channel=channel, zero_gravity_mode=True)
    print("Connecting (zero-gravity mode)...")

    with YAMAdapter(config) as yam:
        print("Connected. The arm should be floating with gravity comp.")
        print("Move the arm to a few configurations and observe:\n")

        for i in range(10):
            state = yam.get_state()
            g = dynamics.gravity_vector(state.q)

            print(f"  [{i+1}/10] q = {np.array2string(state.q, precision=2)}")
            print(f"         g(q)       = {np.array2string(g, precision=3)} Nm")
            print(f"         tau_motor  = {np.array2string(state.tau_motor, precision=3)} Nm")
            print(f"         difference = {np.array2string(state.tau_motor - g, precision=3)} Nm")
            print()

            time.sleep(1.0)

    print("  [PASS] Gravity compensation comparison complete.")


def stage4_momentum_observer(channel: str) -> None:
    """Run the GMO observer on live hardware data."""
    print_header(4, "Momentum Observer (GMO) — Live Force Estimation")

    import mujoco

    from adaptivearm.adapters.yam import YAMAdapter, YAMConfig
    from adaptivearm.dynamics import MuJoCoDynamics
    from adaptivearm.estimation import MomentumObserver
    from adaptivearm.models import get_model

    info = get_model("yam")
    model = mujoco.MjModel.from_xml_path(str(info.model_path))
    dynamics = MuJoCoDynamics(model)

    config = YAMConfig(channel=channel, zero_gravity_mode=True)
    print("Connecting...")

    with YAMAdapter(config) as yam:
        observer = MomentumObserver(
            dynamics=dynamics,
            n_joints=yam.n_joints,
            dt=yam.dt,
            gains=np.full(yam.n_joints, 20.0),
            lowpass_cutoff=5.0,
        )

        state = yam.get_state()
        observer.reset()

        print("Observer running. Try pushing/pulling the end-effector!")
        print("(The arm is in zero-gravity mode — it should feel light)\n")
        print(f"  {'time':>6s}  {'|tau_ext|':>10s}  {'tau_ext':>50s}")
        print(f"  {'----':>6s}  {'--------':>10s}  {'-------':>50s}")

        t0 = time.time()
        try:
            while time.time() - t0 < 30.0:
                state = yam.get_state()
                output = observer.update(state)

                elapsed = time.time() - t0
                tau_ext = output.tau_ext
                norm = np.linalg.norm(tau_ext)

                # Only print when there's something interesting
                if norm > 0.5 or int(elapsed * 5) % 5 == 0:
                    print(
                        f"  {elapsed:6.1f}s  {norm:10.3f}  "
                        f"{np.array2string(tau_ext, precision=2, suppress_small=True)}"
                    )

                time.sleep(yam.dt)

        except KeyboardInterrupt:
            print("\n  (stopped by user)")

    print("\n  [PASS] Momentum observer ran on live hardware.")


def stage5_ekf_observer(channel: str) -> None:
    """Run the EKF observer on live hardware data."""
    print_header(5, "EKF Observer — Live Force Estimation")

    import mujoco

    from adaptivearm.adapters.yam import YAMAdapter, YAMConfig
    from adaptivearm.dynamics import MuJoCoDynamics
    from adaptivearm.estimation import EKFObserver
    from adaptivearm.models import get_model

    info = get_model("yam")
    model = mujoco.MjModel.from_xml_path(str(info.model_path))
    dynamics = MuJoCoDynamics(model)

    config = YAMConfig(channel=channel, zero_gravity_mode=True)
    print("Connecting...")

    with YAMAdapter(config) as yam:
        observer = EKFObserver(
            dynamics=dynamics,
            n_joints=yam.n_joints,
            dt=yam.dt,
        )

        state = yam.get_state()
        observer.reset()

        print("EKF observer running. Try pushing the end-effector!\n")
        print(f"  {'time':>6s}  {'|tau_ext|':>10s}  {'tau_ext':>50s}")
        print(f"  {'----':>6s}  {'--------':>10s}  {'-------':>50s}")

        t0 = time.time()
        try:
            while time.time() - t0 < 30.0:
                state = yam.get_state()
                output = observer.update(state)

                elapsed = time.time() - t0
                tau_ext = output.tau_ext
                norm = np.linalg.norm(tau_ext)

                if norm > 0.5 or int(elapsed * 5) % 5 == 0:
                    print(
                        f"  {elapsed:6.1f}s  {norm:10.3f}  "
                        f"{np.array2string(tau_ext, precision=2, suppress_small=True)}"
                    )

                time.sleep(yam.dt)

        except KeyboardInterrupt:
            print("\n  (stopped by user)")

    print("\n  [PASS] EKF observer ran on live hardware.")


def stage6_collision_detection(channel: str) -> None:
    """Run collision detection on live hardware."""
    print_header(6, "Collision Detection — Live")

    import mujoco

    from adaptivearm.adapters.yam import YAMAdapter, YAMConfig
    from adaptivearm.dynamics import MuJoCoDynamics
    from adaptivearm.estimation import MomentumObserver
    from adaptivearm.estimation.collision_detector import CollisionDetector
    from adaptivearm.models import get_model

    info = get_model("yam")
    model = mujoco.MjModel.from_xml_path(str(info.model_path))
    dynamics = MuJoCoDynamics(model)

    config = YAMConfig(channel=channel, zero_gravity_mode=True)
    print("Connecting...")

    with YAMAdapter(config) as yam:
        observer = MomentumObserver(
            dynamics=dynamics,
            n_joints=yam.n_joints,
            dt=yam.dt,
            gains=np.full(yam.n_joints, 20.0),
            lowpass_cutoff=5.0,
        )

        # Set collision threshold — adjust based on noise floor
        detector = CollisionDetector(
            thresholds=np.array([3.0, 3.0, 3.0, 2.0, 2.0, 2.0]),
            holdoff_time=0.5,
        )

        observer.reset()
        detector.reset()

        print("Collision detector running. Try tapping the arm!\n")

        t0 = time.time()
        try:
            while time.time() - t0 < 30.0:
                state = yam.get_state()
                output = observer.update(state)
                event = detector.update(output.tau_ext, state.timestamp)

                if event.in_collision:
                    print(
                        f"  *** COLLISION at {state.timestamp - t0:.1f}s! "
                        f"severity={event.severity:.2f} "
                        f"joints={np.array2string(event.joint_mask.astype(int))}"
                    )

                time.sleep(yam.dt)

        except KeyboardInterrupt:
            print("\n  (stopped by user)")

    print("\n  [PASS] Collision detection ran on live hardware.")


def main() -> None:
    parser = argparse.ArgumentParser(description="YAM hardware test")
    parser.add_argument("--channel", default="can0", help="CAN channel (default: can0)")
    parser.add_argument("--skip-to", type=int, default=1, help="Skip to stage N")
    args = parser.parse_args()

    # Graceful Ctrl+C
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

    print("=" * 60)
    print("  AdaptiveArm — YAM Hardware Test Suite")
    print("=" * 60)
    print(f"\n  CAN channel: {args.channel}")
    print(f"  Starting from stage: {args.skip_to}")

    stages = [
        (1, "Connection & State Reading", lambda: stage1_connection(args.channel)),
        (2, "Model Registry & Dynamics", lambda: stage2_model_registry()),
        (3, "Live Gravity Comp Comparison", lambda: stage3_gravity_comp_live(args.channel)),
        (4, "GMO Observer (30s)", lambda: stage4_momentum_observer(args.channel)),
        (5, "EKF Observer (30s)", lambda: stage5_ekf_observer(args.channel)),
        (6, "Collision Detection (30s)", lambda: stage6_collision_detection(args.channel)),
    ]

    for num, name, fn in stages:
        if num < args.skip_to:
            continue

        if num > 1:
            input_continue(f"Ready for Stage {num}: {name}? Press Enter...")

        fn()

    print("\n" + "=" * 60)
    print("  All stages complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
