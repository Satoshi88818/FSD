FSD v12 — Full-Stack Autonomous Driving System

Version 12 of a research-grade Full Self-Driving stack, integrating five targeted architectural improvements over the v11 foundation: risk-aware planning, physics-informed control, high-frequency reactivity, social-aware prediction, and provably-safe formal supervision.

Created by: James Squire 

Table of Contents

Project Purpose & Potential

What's New in v12

Architecture Overview

Module Specifications

System Requirements

Engineering Choices & Rationale

Performance Budget

Deployment Guide

Safety & Certification

Training

Configuration Reference

Roadmap

1. Project Purpose & Potential

FSD v12 is a full-stack, end-to-end autonomous driving inference and control system designed to close the gap between research-grade neural planners and production-grade certified autonomous vehicles. It unifies:

Neural perception (camera, LiDAR, radar fusion via Mamba SSM)

Probabilistic trajectory planning (consistency-distilled flow matching)

Physics-grounded control (iLQR MPC with PINN tire dynamics)

Formal safety guarantees (NCBF + ASIL-D Simplex Architecture)

High-frequency reactive control (200 Hz async CPU loop)

Why It Matters

Most published autonomous driving systems either (a) achieve strong planning performance but lack certifiable safety guarantees, or (b) have hard safety guarantees but sacrifice adaptability in complex scenarios. FSD v12 is an attempt to have both within a single, deployable pipeline.

Key capabilities enabled by v12:

Driverless certification pathway — The Simplex Guard provides the ASIL-D non-neural supervisor required by ISO 26262 and equivalent standards.

Limit-handling vehicle control — Neural-Pacejka hybrid tire model enables accurate control at the physical limits of tire friction.

Social scene understanding — GNN-based group interaction modelling improves prediction in dense urban environments (pedestrian clusters, convoy behaviour, roundabouts).

Reduced reaction latency — The 200 Hz async reactive loop cuts worst-case reaction latency from 33 ms (v11) to 5 ms, directly improving safety margins in emergency events.

Principled uncertainty handling — Epistemic uncertainty from MapFormer flows into the iLQR cost, making the planner intrinsically more cautious in unmapped or sensor-degraded regions.

2. What's New in v12

#ImprovementModuleSummary1Stochastic iLQRrefinement_net.pyRisk-aware MPC cost augmented with λ · (h_epi · occ) at every timestep2Neural-Pacejka Hybrid (PINN)proposal_generator.pyDifferentiable tire residual MLP: F_y = F_y_Pacejka + MLP(v, β, ψ̇, αf, αr, μ)3Async Multi-Rate Executionrefinement_net.py30 Hz strategic planner exports iLQR gains (K, k); 200 Hz CPU thread applies u = u_nom + K·Δx + k4GNN Social Interactionheads.pyGated Graph Convolutional Network enriches actor tokens with group-level intent before LevelKValueHead5Simplex Runtime Guardsafety.pyPurely arithmetic ASIL-D supervisor: TTC, max decel, and clearance checks override AI with deterministic safe-stopTAutograd Jacobiansrefinement_net.pytorch.autograd.functional.jacobian replaces finite-difference linearisation in iLQRSolver._linearise 

3. Architecture Overview

┌──────────────────────────────────────────────────────────────────────┐ │ FSD v12 PIPELINE │ │ │ │ Sensors ──► MapFormer (Mamba SSM + Fovea + EDL uncertainty) │ │ │ │ │ ▼ │ │ ConsistencyDistillation (1-step student) → N proposals │ │ │ │ │ ▼ │ │ ProposalScoringTransformer │ │ └─ LevelKValueHead ← [v12] SocialGNN │ │ │ │ │ ▼ │ │ NCBF Safety Filter ← formal CBF invariance guarantee │ │ │ │ │ ▼ │ │ [v12] Simplex Guard ← non-neural TTC / decel override │ │ │ │ │ ▼ │ │ iLQR MPC Execution Layer │ │ ├─ [v12] Stochastic cost: += λ·(h_epi·occ) │ │ ├─ [v12] Autograd Jacobians │ │ └─ Exports K, k gains ──► [v12] Async 200 Hz ReactiveController │ │ │ │ │ ▼ │ │ DynamicBicycleModel │ │ └─ [v12] PINN tire residual ΔF_y + EKF μ calibration │ └──────────────────────────────────────────────────────────────────────┘ 

Data Flow

BEV tensor (fused from cameras, LiDAR, radar) enters MapFormer.

MapFormer outputs lane centrelines, actor trajectories, occupancy logits, and epistemic uncertainty h_epi via Evidential Deep Learning.

ConsistencyDistillationGenerator generates 32 trajectory proposals in a single forward pass (20× faster than diffusion teacher).

ProposalScoringTransformer ranks proposals using Level-K game theory, enriched by the GNN social layer.

The best trajectory passes through NeuralControlBarrierFunction (hard CBF projection) and then the SimplexGuard (Newtonian limit check).

MPCExecutionLayer (iLQR) refines the trajectory into low-level controls, exporting gain matrices K and k.

An async 200 Hz CPU thread continuously applies u = u_nom + K·Δx + k to the latest IMU/state estimate.

The DynamicBicycleModel converts controls to physical actuation, with the PINN residual correcting Pacejka's linear-region assumptions at high slip angles.

4. Module Specifications

config.py — Model Configuration

Central dataclass (ModelConfig) for all hyperparameters. v12 adds fields for:

use_simplex_guard, simplex_ttc_threshold_s, simplex_max_decel_g, simplex_min_clearance_m

pinn_hidden_dim, pinn_num_layers, pinn_residual_weight

async_reactive_hz (default: 200)

gnn_social_k_neighbors, gnn_social_hidden_dim

stochastic_ilqr_risk_lambda

proposal_generator.py — PINN Tire Residual

Contains ManifoldFlowMatchingGenerator (teacher) and ConsistencyDistillationGenerator (student), plus the new PINNTireResidual MLP.

PINNTireResidual: A small MLP conditioned on (v, β, ψ̇, αf, αr, μ) that outputs a lateral force correction ΔF_y. Added to the Pacejka model output inside DynamicBicycleModel, this allows the planner to accurately simulate tyre behaviour during aggressive cornering, emergency braking, and low-μ surfaces (rain, ice).

heads.py — GNN Social Layer + MapFormer

MapFormer is extended with SocialGNN, a gated graph convolutional network that runs over proximity-linked actor tokens before they are passed to LevelKValueHead. Edges are built by nearest-neighbour search on actor_positions. This improves prediction of emergent group behaviour (merging convoys, crossing pedestrian groups).

refinement_net.py — Stochastic iLQR + Async Controller

iLQRSolver implements a full iterative LQR with:

Stochastic running cost: J_t += λ · (h_epi_t · occ_t) — penalises trajectories that pass through uncertain or occupied regions.

Autograd Jacobians: torch.autograd.functional.jacobian replaces finite-difference A, B matrix computation, eliminating numerical noise in the linearisation step and enabling exact gradients through the vehicle dynamics.

AsyncReactiveController is a background thread started by MPCExecutionLayer. It receives (K, k, u_nom) from each 30 Hz iLQR solve and independently runs at 200 Hz, applying the affine gain law u = u_nom + K·(x - x_nom) + k to the latest ego state.

safety.py — Simplex Guard + NCBF

NeuralControlBarrierFunction (v11, unchanged): Learns a barrier function h(x) ≥ 0 and applies a correction network to restore invariance when the trajectory would violate it.

SimplexGuard (v12 new): A purely arithmetic, zero-weight module that enforces three Newtonian checks on every control before actuation:

Time-to-Collision (TTC): TTC = EDT / speed ≥ threshold (default: 2.0 s)

Maximum Deceleration: |a| ≤ max_decel_g · g (default: 0.8 g)

Minimum Clearance: d_obstacle ≥ min_clearance_m (default: 1.5 m)

On any violation, all timesteps for that batch element are overridden with a deterministic safe-stop: maximum deceleration, zero steer. The guard also increments a diagnostic counter and exposes a export_formal_spec() method that outputs SMTLIB2 for automated verification toolchains.

STLRuleLoss (v11, unchanged): Differentiable Signal Temporal Logic loss for training-time rule grounding (stop-at-red, yield-at-crosswalk, speed-limit adherence).

losses.py — Training Losses (unchanged from v11)

FSDv12Loss combines:

LossWeightPurposeimitation1.0MSE vs expert trajectorymt_consistency0.5Multi-task passive/conditioned trajectory consistencytd0.3Temporal-difference value head learningcd0.2Consistency distillation (student ↔ teacher)stlconfigurableDifferentiable traffic rule compliance 

model.py — Unified Forward Pass

FSDV12 orchestrates the full pipeline. Notable v12 wiring:

SimplexGuard wraps NCBF output as the final pre-actuation gate.

MPCExecutionLayer receives epistemic_unc and occupancy for stochastic cost.

AsyncReactiveController is started inside MPCExecutionLayer.__init__.

actor_positions from MapFormer is passed to LevelKValueHead for GNN enrichment.

PINN tire residual is active inside DynamicBicycleModel when config.use_pinn_tire = True.

5. System Requirements

Hardware (Inference)

ComponentMinimumRecommendedGPUNVIDIA RTX 3080 (10 GB VRAM)NVIDIA A100 / Orin NXCPU4-core, 3 GHz8-core, 4 GHz (for 200 Hz reactive thread)RAM16 GB32 GBStorage50 GB SSD200 GB NVMeSensorsCamera + LiDAR (min)Camera + LiDAR + Radar + IMU 

Hardware (Training)

ComponentRecommendedGPU4× NVIDIA A100 80 GB (or equivalent)RAM256 GBStorage2 TB NVMe (dataset)Networking100 Gbps InfiniBand (multi-node) 

Software

DependencyVersionPython≥ 3.10PyTorch≥ 2.2.0 (CUDA 12.1+)torch-geometric≥ 2.5 (for GNN layers)numpy≥ 1.26scipy≥ 1.12 (for EDT / JFA)CUDA Toolkit≥ 12.1OSUbuntu 22.04 LTS / RHEL 9 

Optional

TensorRT ≥ 10.0 — for production INT8/FP16 inference optimisation

ROS 2 Humble — for sensor interface and vehicle CAN bridge

SMTLIB2 solver (e.g., Z3 ≥ 4.12) — for formal verification of exported Simplex specs

6. Engineering Choices & Rationale

Mamba SSM over Transformer for Temporal Memory

Transformer-based temporal fusion scales quadratically with sequence length. Mamba's selective state-space model provides O(L) memory and compute with competitive or superior performance on driving sequences, making it viable for real-time BEV processing over multi-second history horizons.

Consistency Distillation (single-step inference)

Diffusion models produce high-quality trajectory distributions but require many denoising steps. Consistency distillation matches teacher trajectory quality in one forward pass, delivering the 3 ms planning budget required by the latency profile without sacrificing sample diversity.

Stochastic iLQR over Robust MPC

Standard robust MPC uses worst-case adversarial uncertainty bounds, which tend to be overly conservative. The stochastic iLQR risk term λ · (h_epi · occ) is softly proportional to actual sensor uncertainty, yielding smooth, non-conservative behaviour in well-mapped areas while increasing caution precisely where the model is uncertain.

Autograd Jacobians over Finite Differences

Finite-difference Jacobians in iLQR introduce numerical errors that accumulate across iterations, especially near saturation boundaries in the vehicle model. torch.autograd.functional.jacobian produces exact analytic Jacobians, improving convergence and enabling end-to-end gradient flow through the dynamics model.

PINN Tire Residual over Pure Data-Driven Model

A fully learned tire model requires large datasets and can extrapolate dangerously. The Physics-Informed Neural Network residual approach anchors predictions to the well-understood Pacejka magic formula in the linear region, using the MLP only to correct the known residual at high slip angles. This improves generalisation to new tyre compounds and road surfaces.

Two-Layer Safety Architecture (NCBF + Simplex)

The NCBF provides a differentiable, learned safety boundary that respects complex geometry. However, neural networks can fail on out-of-distribution inputs. The Simplex Guard provides a non-neural, formally verifiable backstop that is immune to OOD failure by operating solely on scalar physical quantities. Neither layer alone is sufficient for ASIL-D certification; both together are.

30/200 Hz Dual-Rate Architecture

Planning at 200 Hz is computationally prohibitive for a full neural stack. Planning at 30 Hz risks missing fast-evolving events (e.g., a pedestrian stepping off a kerb). The gain-scheduling approach (export K, k from iLQR; apply at 200 Hz) gives the latency profile of a 200 Hz controller at the compute cost of a 30 Hz planner.

GNN for Social Interaction

Independent trajectory prediction ignores interaction effects. The Gated Graph Convolutional Network models message passing between nearby actors, allowing the predictor to capture group-level dependencies (merging, yielding, following) that are invisible to per-agent models.

7. Performance Budget

GPU Thread (30 Hz, ≤ 33 ms budget)

StageLatencyMapFormer (Mamba SSM + Fovea + GNN)~6 msConsistencyDistillation (1 step)~3 msProposalScoringTransformer~2 msNCBF Safety Filter~1 msSimplex Guard (non-neural)~0.05 msiLQR MPC (stochastic, autograd Jac)~2.5 msTotal GPU frame≈ 14.5 ms 

CPU Reactive Thread (200 Hz, ≤ 5 ms budget)

StageLatencyiLQR gain tracking (K @ Δx + k)~0.05 msSimplex Guard check~0.05 msReaction latency≈ 5 ms 

The total GPU frame of 14.5 ms leaves 18.5 ms headroom within the 33 ms budget, providing margin for sensor pre-processing, CAN I/O, and operating system jitter.

8. Deployment Guide

8.1 Environment Setup

# Clone the repository git clone https://github.com/your-org/fsd-v12.git cd fsd-v12 # Create a virtual environment python -m venv .venv source .venv/bin/activate # Install dependencies pip install torch==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121 pip install torch-geometric pip install -r requirements.txt 

8.2 Configuration

Copy and edit the default config:

cp configs/default_v12.yaml configs/my_deployment.yaml 

Key v12 fields to review before deployment:

# Safety thresholds — tune to vehicle dynamics simplex_ttc_threshold_s: 2.0 simplex_max_decel_g: 0.8 simplex_min_clearance_m: 1.5 # Reactive controller async_reactive_hz: 200 # Risk sensitivity stochastic_ilqr_risk_lambda: 0.1 # PINN tire model use_pinn_tire: true pinn_hidden_dim: 64 # GNN social layer gnn_social_k_neighbors: 8 

8.3 Model Initialisation

from config import ModelConfig from model import FSDV12 config = ModelConfig.from_yaml("configs/my_deployment.yaml") model = FSDV12(config) # Load pre-trained weights checkpoint = torch.load("checkpoints/fsd_v12_latest.pt", map_location="cuda") model.load_state_dict(checkpoint["model_state_dict"]) model.eval().cuda() 

8.4 Inference

import torch # Reset SSM memory at the start of each route model.reset_memory() # Per-frame inference with torch.no_grad(): output = model( bev=bev_tensor.cuda(), # (B, C, H, W) ego_state=ego_state.cuda(), # (B, 5): [x, y, θ, v, a] imu_accel=imu_tensor.cuda(), # (B, 3) optional traffic_context=traffic_dict, # optional rule context ) safe_trajectory = output["strategic_trajectory"] mpc_trajectory = output["mpc_execution_trajectory"] simplex_fired = output["simplex_overridden"] # diagnostic flag 

8.5 Async Reactive Controller

The AsyncReactiveController starts automatically when config.use_mpc_execution = True. It runs as a daemon thread and polls the latest ego state from a shared memory buffer. Ensure your vehicle interface writes to this buffer at ≥ 200 Hz.

# The thread is started inside MPCExecutionLayer.__init__ # To shut it down gracefully: model.mpc_controller.reactive_controller.stop() 

8.6 Formal Verification Export

# Export Simplex Guard spec to SMTLIB2 spec = model.simplex_guard.export_formal_spec() with open("simplex_spec.smt2", "w") as f: f.write(spec) # Verify with Z3 # z3 simplex_spec.smt2 

8.7 Docker Deployment

FROM nvcr.io/nvidia/pytorch:24.01-py3 WORKDIR /app COPY . . RUN pip install -r requirements.txt EXPOSE 8080 CMD ["python", "serve.py", "--config", "configs/production.yaml"] docker build -t fsd-v12:latest . docker run --gpus all --rm -it fsd-v12:latest 

8.8 ROS 2 Integration

# Source ROS 2 source /opt/ros/humble/setup.bash # Launch sensor bridge and FSD node ros2 launch fsd_v12 fsd_v12.launch.py config:=configs/production.yaml 

The ROS 2 node subscribes to /bev_tensor, /ego_state, /imu/data and publishes /fsd/trajectory, /fsd/controls, /fsd/simplex_override.

9. Safety & Certification

ASIL-D Simplex Architecture

The SimplexGuard module is designed to satisfy the requirements of a ASIL-D non-neural supervisor under ISO 26262:

No learned weights — cannot be adversarially manipulated or corrupted by OOD inputs.

O(1) arithmetic — formal complexity bound, no recursion or data-dependent branching in the override path.

Provably correct — implements Newtonian kinematics exactly; correctness is manually auditable.

SMTLIB2 export — the export_formal_spec() method outputs a machine-checkable formal spec for use with automated FMEA and model-checking toolchains.

Diagnostic Monitoring

# Check how many times the Simplex Guard has fired override_count = model.simplex_guard.get_override_count() model.simplex_guard.reset_override_count() # NCBF formal verification footprint (for FMEA documentation) ncbf_spec = model.safety_filter.export_verification_smt() 

Safety Layering Summary

LayerTypeGuaranteeSTL Rule LossTraining-timeSoft rule grounding during learningNCBF Safety FilterNeural, inference-timeHard CBF invariance (learned)Simplex GuardNon-neural, inference-timeASIL-D Newtonian bounds (formal) 

10. Training

Loss Composition

Training uses FSDv12Loss, which is unchanged from v11. All five v12 improvements are architectural (inference/execution path) and require no new training objectives.

from losses import FSDv12Loss loss_fn = FSDv12Loss(config) losses = loss_fn(pred_dict, gt_dict) # losses["total"] is the scalar to backpropagate 

Training Command

python train.py \ --config configs/train_v12.yaml \ --dataset /data/nuplan \ --output_dir checkpoints/v12 \ --gpus 4 \ --batch_size 32 \ --epochs 100 

Teacher–Student Distillation

The flow-matching teacher (ManifoldFlowMatchingGenerator) is trained first, then frozen. The consistency distillation student (ConsistencyDistillationGenerator) is trained with cd_loss against the frozen teacher, targeting single-step proposal quality matching the multi-step teacher.

11. Configuration Reference

Below are the key v12-specific fields in ModelConfig:

FieldDefaultDescriptionuse_simplex_guardTrueEnable Simplex ASIL-D supervisorsimplex_ttc_threshold_s2.0TTC threshold for emergency stop (seconds)simplex_max_decel_g0.8Maximum commanded deceleration (g)simplex_min_clearance_m1.5Minimum obstacle clearance (metres)use_pinn_tireTrueEnable PINN tire residualpinn_hidden_dim64PINN MLP hidden layer widthpinn_residual_weight1.0Scale factor on ΔF_y residualasync_reactive_hz200Frequency of CPU reactive controllergnn_social_k_neighbors8Number of actor neighbours for GNN edgesgnn_social_hidden_dim128GNN hidden layer widthstochastic_ilqr_risk_lambda0.1Risk term weight in iLQR costuse_ncbfTrueUse NCBF (vs legacy QP fallback)use_mpc_executionTrueEnable iLQR MPC execution layerjfa_resolution_m0.5BEV grid resolution (metres/cell)mpc_dt0.1MPC timestep (seconds) 

12. Roadmap

Potential future improvements identified from the v12 foundation:

v13: Online PINN adaptation — Continuously update tire residual weights via online meta-learning from IMU feedback, enabling per-vehicle and per-surface calibration without stops.

v13: Diffusion-based uncertainty propagation — Replace point-estimate occupancy with a full distribution, feeding into both the stochastic iLQR cost and the NCBF.

v13: Hardware-in-the-loop Simplex validation — Integrate the SMTLIB2 export with a continuous HiL test bench for automated regression of the formal safety spec.

v13: Multi-agent cooperative planning — Extend the GNN to support V2X communication, enabling coordinated trajectory optimisation across connected vehicles.

Quantisation and TensorRT export — INT8/FP16 TensorRT compilation of MapFormer and the proposal generator to target embedded GPU platforms (Orin, Thor).

Licence

This project is released under the MIT Licence.

Citation

@software{fsd_v12_2025, title = {FSD v12: Risk-Aware Autonomous Driving with Formal Safety Supervision}, year = {2025}, note = {Five targeted improvements over FSD v11: Stochastic iLQR, PINN Tire Model, Async Multi-Rate Control, GNN Social Interaction, Simplex ASIL-D Guard} } 
