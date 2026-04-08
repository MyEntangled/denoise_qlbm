#!/usr/bin/env python3
"""
Run classical and quantum LBM experiments from the command line
and store *all* intermediate distributions for later processing.
"""

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import src.qlbm.qlbm_model as qlbm_model
import src.clbm.clbm_model as clbm_model

from typing import Callable, Sequence
import importlib
import argparse
import os
from datetime import datetime
from tqdm import tqdm

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run classical and quantum LBM experiments and save all intermediate F, rho, u."
    )

    # Testcase
    parser.add_argument("--testcase", type=str, required=True,
                        help="Testcase for simulation. Available: fourier, gaussian, cylinder, taylor_green.")

    parser.add_argument(
        "--parameters",
        nargs="*",
        type=float,
        default=None,
        help="Optional testcase parameters (space-separated). Leave empty if testcase needs no extra parameters.",
    )

    parser.add_argument("--steps", type=int, required=True,
                        help="Number of time steps to simulate.")

    parser.add_argument("--save-every", type=int, default=1,
                        help="Store data every k steps (default: 1 = every step).")

    # Experiment control
    parser.add_argument("--run-classical", action="store_true",
                        help="Run classical LBM.")
    parser.add_argument("--run-quantum", action="store_true",
                        help="Run quantum LBM.")
    parser.add_argument("--outdir", type=str, default=".",
                        help="Output directory.")

    args = parser.parse_args()

    # Default: if neither flag given, run both
    if not args.run_classical and not args.run_quantum:
        args.run_classical = True
        args.run_quantum = True
    return args


def run_solver(name: str,
               solver: clbm_model.ClassicalLBMSimulator | qlbm_model.QuantumLBMSimulator,
               config: dict,
               F0: np.ndarray,
               boundary: np.ndarray,
               u_boundary: np.ndarray,
               steps: int,
               save_every: int):
    """
    Run a solver (classical or quantum) and store trajectories of:
      - F distribution
      - rho
      - u = (ux,), (ux, uy), or (ux, uy, uz)
    """
    print(f"[{name}] Starting simulation for {steps} steps, save_every={save_every}.")

    def compute_macros(F: np.ndarray, obstacles: np.ndarray|None):
        # Compute macroscopic density and velocity
        rho = np.sum(F, axis=-1)
        rho = np.clip(rho, 1e-12, None)
        u = np.einsum("...q,qd->...d", F, solver.c) / (rho[..., np.newaxis])

        if obstacles is not None:
            rho[obstacles] = 0.
            u[obstacles] = 0.
        return rho, u


    assert F0.shape[:-1] == solver.grid_size
    assert F0.shape[-1] == solver.Q
    if not isinstance(boundary, Callable):
        assert boundary.shape == solver.grid_size
    if u_boundary is not None and not isinstance(u_boundary, Callable):
        assert u_boundary.shape[:-1] == solver.grid_size
        assert u_boundary.shape[-1] == solver.d


    T_save = 1 + steps // save_every

    F_traj = np.empty((T_save,) + F0.shape, dtype=float)
    rho_traj = np.empty((T_save,) + solver.grid_size, dtype=float)
    u_traj = np.empty((T_save,) + (*solver.grid_size, solver.d), dtype=float)

    # --- Process time-dependent boundaries if needed ---
    boundary_t = None
    u_boundary_t = None

    if isinstance(boundary, Callable):
        print("Obstacles are time-dependent.")
        boundary_t = boundary

    if isinstance(u_boundary, Callable):
        print("Obstacle velocities are time-dependent.")
        u_boundary_t = u_boundary


    # --- Initial distributions and macros ---
    F_traj[0] = F0
    init_boundary = boundary_t(0) if boundary_t is not None else boundary
    rho_traj[0], u_traj[0] = compute_macros(F0, init_boundary)
    save_idx = 1


    # --- Main time-stepping loop ---

    if name == "classical":
        F = F0.copy()

        u_prescribed = config.get("u_adv", None)

        if isinstance(u_prescribed, Callable):
            print("Reference velocity is time-dependent.")
            u_prescribed_t = u_prescribed
        elif isinstance(u_prescribed, (Sequence, np.ndarray)):
            print("Reference velocity is time-independent (fixed u0).")
            u_prescribed_t = None
        else:
            assert u_prescribed is None, "u0 must be either None, an array-like, or a callable."
            u_prescribed_t = None


        for i in tqdm(range(1, steps + 1),  desc=f'{name}'):
            boundary_i = boundary_t(i) if boundary_t is not None else boundary
            u_boundary_i = u_boundary_t(i) if u_boundary_t is not None else u_boundary

            if u_prescribed_t is not None:
                u_prescribed_i = np.array(u_prescribed_t(i))
            else:
                u_prescribed_i = u_prescribed

            F = solver.step(F, boundary_i, u_boundary_i, u_adv=u_prescribed_i)


            if i % save_every == 0:
                F_traj[save_idx] = F
                rho_traj[save_idx], u_traj[save_idx] = compute_macros(F, boundary_i)
                save_idx += 1


    elif name == "quantum":
        encoding_type = 'sqrt'  ## Hardcoded
        # Embedding function |psi> --> |0>|psi>
        multiply_ket0 = lambda psi: np.concatenate([psi, np.zeros_like(psi)], axis=-1)

        u0 = config["u0"]

        if isinstance(u0, Callable):
            print("Reference velocity is time-dependent.")
            u0_t = u0
        elif isinstance(u0, (Sequence, np.ndarray)):
            print("Reference velocity is time-independent (fixed u0).")
            # This matches the old behavior where we called init_collision_operator(u0=...)
            solver.init_collision_operator(u0=np.array(u0))
            u0_t = None
        else:
            assert u0 is None, "u0 must be either None, an array-like, or a callable."
            assert solver.U_col is not None, "Collision operator must be initialized before simulation if u0 is None."
            u0_t = None

        if encoding_type == 'sqrt':
            init_states = F0 ** 0.5
        else: # self.encoding_type == 'full':
            init_states = F0.copy()


        states = init_states.copy()
        init_norm = np.linalg.norm(states)
        states = states / init_norm
        print("Initial norm:", init_norm)

        for i in tqdm(range(1, steps + 1),  desc=f'{name}'):

            boundary_i = boundary_t(i) if boundary_t is not None else boundary
            u_boundary_i = u_boundary_t(i) if u_boundary_t is not None else u_boundary

            if u0_t is not None:
                u0_i = np.array(u0_t(i))
                U_col, op_seq, gate_seq = solver._init_denoising_collision_(u0_i)
            else:
                ## Using fixed collision operator
                U_col, op_seq, gate_seq = solver.U_col, solver.U_col_op_seq, solver.U_col_gate_seq

            col_op = {"full_unitary": U_col, "subsystem_unitary": op_seq, "quantum_gates": gate_seq}

            states = solver.step(states, obstacles=boundary_i, u_obstacles=u_boundary_i, embed_fn=multiply_ket0, collide_operation=col_op)

            if i % save_every == 0:
                selected_sign = np.sign(np.sum(states, axis=-1, keepdims=True))

                states_rec = states * selected_sign
                states_rec = states_rec * init_norm

                # Reconstruct F from states
                if encoding_type == 'sqrt':
                    #F_rec = np.clip(states_rec, a_min=0, a_max=None) ** 2
                    F_rec = np.sign(states_rec) * states_rec ** 2
                else:
                    #F_rec = np.clip(states_rec, a_min=0, a_max=None)
                    F_rec = states_rec.copy()

                # F_rec[boundary_i] = 0

                F_traj[save_idx] = F_rec
                rho_traj[save_idx], u_traj[save_idx] = compute_macros(F_rec, boundary_i)
                save_idx += 1

    else:
        raise ValueError(f"Unknown solver name: {name}")


    print(f"[{name}] Done. Stored {save_idx} snapshots.")

    return {
        "F": F_traj,
        "rho": rho_traj,
        "u": u_traj,
    }


def build_output_path(outdir: str, exp_name: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{exp_name}_{timestamp}.npz"
    return os.path.join(outdir, fname)


# ---------------------------
# Main entry point
# ---------------------------
def main():
    args = parse_args()

    # Initialize F0 via your code
    print("Initializing state...")

    testcase_module = importlib.import_module(f"src.testcases.{args.testcase}")
    testcase_init = getattr(testcase_module, "setup_testcase")

    # Handle optional testcase parameters (some testcases may not need any)
    if args.parameters is None:
        params = ()
    else:
        params = tuple(args.parameters)

    config, F0, solid, u_solid = testcase_init(*params)

    print("Initial state shape:", F0.shape)

    classical_traj = None
    quantum_traj = None


    if args.run_classical:
        classical_lbm_solver = clbm_model.ClassicalLBMSimulator(
            lattice=config["lattice"],
            grid_size=config["grid_size"],
            omega=1.
        )

        classical_traj = run_solver(
            name="classical",
            solver=classical_lbm_solver,
            config=config,
            F0=F0,
            boundary=solid,
            u_boundary=u_solid,
            steps=args.steps,
            save_every=args.save_every,
        )

    if args.run_quantum:
        quantum_lbm_solver = qlbm_model.QuantumLBMSimulator(
            lattice=config["lattice"],
            grid_size=config["grid_size"],
            encoding_type="sqrt",
            collision_model_type="denoising",
            is_scalar_field=config["is_scalar_field"],
            apply_operators_as="full_unitary",
        )

        quantum_traj = run_solver(
            name="quantum",
            solver=quantum_lbm_solver,
            config=config,
            F0=F0,
            boundary=solid,
            u_boundary=u_solid,
            steps=args.steps,
            save_every=args.save_every,
        )

    # Modify lambda function in config to string for serialization
    if "u_adv" in config and isinstance(config["u_adv"], Callable):
        config["u_adv"] = "time-dependent function"
    if "u0" in config and isinstance(config["u0"], Callable):
        config["u0"] = "time-dependent function"

    # Build metadata
    meta = {
        "testcase": args.testcase,
        "parameters": args.parameters,
        "config": config,
        "steps": args.steps,
        "save_every": args.save_every,
        "run_classical": args.run_classical,
        "run_quantum": args.run_quantum,
        "timestamp": datetime.now().isoformat(),
    }

    exp_name = "lbm_" + args.testcase + '_'.join(f"{p}" for p in params)
    outpath = build_output_path(args.outdir, exp_name)
    print("Saving results to:", outpath)

    # Save everything in a single npz
    np.savez_compressed(
        outpath,
        F0=F0,
        classical_traj=classical_traj,
        quantum_traj=quantum_traj,
        meta=meta,
    )


if __name__ == "__main__":
    main()
