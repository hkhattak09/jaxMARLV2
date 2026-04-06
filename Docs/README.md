# MARL-LLM Documentation Index

**Agent-Optimized Documentation for Multi-Agent Reinforcement Learning with LLM Regularization**

This documentation is designed for AI agents and developers to quickly understand and work with the MARL-LLM codebase without reading the entire source code.

## 📚 Documentation Structure

The documentation is organized hierarchically for efficient navigation:

### 🎯 Start Here

**[SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md)** — Main entry point
- Complete system overview
- Component hierarchy and relationships  
- File organization and navigation guide
- Quick lookup table for common tasks

### 🔧 Core Documentation

Read in order for comprehensive understanding:

1. **[CORE_COMPONENTS.md](CORE_COMPONENTS.md)** — Algorithm implementation
   - MADDPG algorithm (orchestrator)
   - DDPGAgent (individual agents)
   - Replay buffers and experience storage
   - Neural network architectures
   - Exploration strategies

2. **[ENVIRONMENT_INTERFACE.md](ENVIRONMENT_INTERFACE.md)** — Environment mechanics
   - AssemblyEnv (JAX environment)
   - JaxAssemblyAdapterGPU (PyTorch bridge)
   - State representation and physics
   - Observation/action spaces
   - Reward computation
   - Prior actions (Reynolds flocking)

3. **[DATA_FLOW.md](DATA_FLOW.md)** — Tensor shapes and transformations
   - Complete shape reference for all data
   - JAX → NumPy → PyTorch conversions
   - Column-major vs row-major conventions
   - Buffer storage layout
   - Device transfer patterns
   - Memory consumption estimates

4. **[TRAINING_PIPELINE.md](TRAINING_PIPELINE.md)** — End-to-end training
   - Initialization and setup
   - Episode rollout mechanics
   - Network update procedure
   - Evaluation protocol
   - Checkpointing and logging
   - Hyperparameter effects

5. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** — Fast lookup
   - Function signatures
   - Common code patterns
   - Configuration parameters
   - Debugging scenarios
   - Where to add features
   - Quick commands

### 🧠 Design Decision Records

6. **[CTM_ACTOR_DESIGN.md](CTM_ACTOR_DESIGN.md)** — CTM actor design decisions
   - Why stateless rollout (vs stateful / R-MADDPG)
   - AggregatingCritic (centralised critic) design and failure history
   - CTMActor and CTMDDPGAgent implementation details
   - Prior-seeded iterative reasoning research direction

7. **[REWARD_PHYSICS_REDESIGN.md](REWARD_PHYSICS_REDESIGN.md)** — Physics and reward redesign decisions *(all implemented)*
   - Physics fix: k_ball=2000, 4 substeps (prevents tunneling)
   - r_avoid redesign: dynamic formula → fixed 0.10, correct radius definition
   - New reward structure: stepping stone + physical contact penalty
   - Metrics redesign: sensing_coverage, r_avoid_violation_count

## 🚀 Quick Start for Agents

### First-Time Navigation

1. **Understand the system**: Read [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) (10 min)
2. **Find what you need**: Use the navigation table in SYSTEM_ARCHITECTURE.md
3. **Deep dive**: Read relevant section from CORE_COMPONENTS, ENVIRONMENT_INTERFACE, or DATA_FLOW
4. **Quick reference**: Bookmark QUICK_REFERENCE.md for function signatures

### Common Agent Tasks

| Task | Documentation to Read |
|------|----------------------|
| Understand training loop | TRAINING_PIPELINE.md → Section 2-4 |
| Fix shape mismatch error | DATA_FLOW.md → Section 9 (Common Errors) |
| Modify reward function | ENVIRONMENT_INTERFACE.md → Section 1 (Reward Function) + QUICK_REFERENCE.md → Section 5 |
| Change network architecture | CORE_COMPONENTS.md → Section 4 + QUICK_REFERENCE.md |
| Debug slow training | TRAINING_PIPELINE.md → Section 9 + QUICK_REFERENCE.md → Section 4 |
| Add new metric | QUICK_REFERENCE.md → Section 5 (Add New Metric) |
| Understand data shapes | DATA_FLOW.md → Section 11 (Quick Reference Table) |

## 💡 Key Concepts

### System Design

- **Hybrid Framework**: JAX (environment) + PyTorch (learning) + DLPack (zero-copy GPU bridge)
- **Parallelization**: `jax.vmap` for multiple environments on GPU
- **Data Flow**: JAX GPU → DLPack → PyTorch GPU (no CPU intermediate during rollout)
- **Memory Split**: JAX uses 15% GPU (~2GB), PyTorch uses rest (~12GB)
- **Performance**: 15-25% faster than CPU-based pipeline due to eliminating PCIe transfers

### Shape Conventions

**Critical for agents**: Different components use different conventions!

```python
# Environment outputs (column-major for agents)
observations: (obs_dim, n_agents)     # Features in rows
actions:      (action_dim, n_agents)  # For buffer storage
rewards:      (1, n_agents)

# Environment inputs (row-major)
actions:      (n_agents, action_dim)  # Standard format

# Buffer storage (row-major)
All data:     (n_samples, feature_dim)

# PyTorch networks (row-major)
All data:     (batch_size, feature_dim)
```

### Training Flow

```
Reset Env → Select Actions → Step Env → Store in Buffer
    ↓
[After episode_length steps]
    ↓
Sample Batch → Update Critic → Update Actor → Soft Update Targets
    ↓
[Repeat 20 times]
    ↓
Next Episode
```

## 📊 Codebase Statistics

- **Total Lines**: ~3,500 (core implementation)
- **Main Training Script**: `train_assembly_jax_gpu.py` (380+ lines, GPU-optimized)
- **JAX Environment**: `assembly.py` (1065 lines)
- **GPU Adapter**: `jax_assembly_wrapper_gpu.py` (~250 lines, DLPack-based)
- **MADDPG Algorithm**: `maddpg.py` (~400 lines)
- **Documentation**: ~75,000 words across 7 files

## 🐛 Common Pitfalls

1. **Shape transpose confusion**: Remember env outputs are column-major!
2. **Forgetting to .detach()**: Must call `.detach()` on actions before env.step() (DLPack requirement)
3. **Forgetting to switch modes**: Call `prep_rollouts(device='gpu')` before rollout, `prep_training()` before updates
4. **Alpha not updating**: Use `env.alpha` not `env.env.alpha`
5. **Buffer not full**: Check `len(buffer) >= batch_size` before sampling
6. **JAX memory**: Set `XLA_PYTHON_CLIENT_MEM_FRACTION` **before** any JAX imports
7. **Missing GPU**: GPU-optimized version requires CUDA (no CPU fallback)

## 📁 Repository Structure

```
new_marl_llm_implementation/
├── MARL-LLM/                  # Main implementation
│   ├── marl_llm/
│   │   ├── train/            # Training scripts ⭐
│   │   │   ├── train_assembly_jax_gpu.py  # GPU-optimized entry point (main)
│   │   │   └── eval_render.py             # GIF rendering utility (shared)
│   │   ├── algorithm/        # MADDPG, buffers, networks ⭐
│   │   │   └── utils/
│   │   │       ├── ctm_actor.py    # CTMActor (wraps ContinuousThoughtMachineRL)
│   │   │       ├── ctm_agent.py   # CTMDDPGAgent
│   │   │       ├── agents.py      # DDPGAgent (MLP actor)
│   │   │       ├── networks.py    # MLPNetwork, AggregatingCritic
│   │   │       └── buffer_agent.py
│   │   ├── cfg/              # Configuration ⭐
│   │   ├── eval/             # Evaluation
│   │   │   └── eval_shapes.py  # Standalone post-training eval
│   │   └── tests/            # Test suite
│   │       └── test_ctm_implementation.py
│   └── cus_gym/              # Environment adapter (GPU, DLPack) ⭐
│       └── gym/wrappers/customized_envs/
│           ├── jax_assembly_wrapper_gpu.py  # Active (DLPack, GPU)
│           └── jax_assembly_wrapper.py      # CPU adapter (legacy)
├── JaxMARL/                   # JAX environment ⭐
│   └── jaxmarl/environments/mpe/assembly.py
├── continuous-thought-machines/ # CTM base model
│   └── models/ctm_rl.py
├── fig/                       # Target shapes
│   └── results.pkl           # Preprocessed coordinates
└── Docs/                      # **This documentation** ⭐
    ├── README.md
    ├── SYSTEM_ARCHITECTURE.md
    ├── CORE_COMPONENTS.md
    ├── ENVIRONMENT_INTERFACE.md
    ├── DATA_FLOW.md
    ├── TRAINING_PIPELINE.md
    ├── QUICK_REFERENCE.md
    ├── CTM_ACTOR_DESIGN.md      # CTM design decisions
    └── REWARD_PHYSICS_REDESIGN.md  # Physics + reward + metrics redesign (all implemented)
```

⭐ = Most frequently accessed by agents

## 🔍 Navigation Tips

### By Agent Goal

**"I want to understand how X works"**
→ Read relevant section in CORE_COMPONENTS or ENVIRONMENT_INTERFACE

**"I want to modify Y"**
→ QUICK_REFERENCE → Section 5 (Where to Add New Features)

**"I'm getting error Z"**
→ DATA_FLOW → Section 9 (Common Errors) or QUICK_REFERENCE → Section 4 (Debugging)

**"I want to train a model"**
→ TRAINING_PIPELINE or QUICK_REFERENCE → Pattern 1

**"I want to evaluate a model"**
→ TRAINING_PIPELINE → Section 7 or QUICK_REFERENCE → Pattern 2

### By Experience Level

**New to codebase**:
1. SYSTEM_ARCHITECTURE (overview)
2. QUICK_REFERENCE → Pattern 1 (training setup)
3. Experiment and refer back as needed

**Familiar with MARL**:
1. SYSTEM_ARCHITECTURE → Navigation Guide
2. Jump to relevant section (CORE_COMPONENTS, ENVIRONMENT_INTERFACE, etc.)
3. QUICK_REFERENCE for function signatures

**Debugging specific issue**:
1. DATA_FLOW → Section 9 (shape errors) or QUICK_REFERENCE → Section 4
2. Check function signature in QUICK_REFERENCE
3. Review component docs if needed

## 🎓 Learning Path

### Path 1: Quick Implementation (30 min)
1. SYSTEM_ARCHITECTURE → Overview + Navigation (10 min)
2. QUICK_REFERENCE → Pattern 1 (5 min)
3. Run training script, observe (10 min)
4. Refer back to docs as questions arise (5 min)

### Path 2: Deep Understanding (2-3 hours)
1. SYSTEM_ARCHITECTURE (15 min)
2. CORE_COMPONENTS (30 min)
3. ENVIRONMENT_INTERFACE (30 min)
4. DATA_FLOW (30 min)
5. TRAINING_PIPELINE (30 min)
6. Experiment and debug (30+ min)

### Path 3: Targeted Modification (15-45 min)
1. SYSTEM_ARCHITECTURE → Navigation (5 min)
2. Find relevant section in QUICK_REFERENCE → Section 5 (5 min)
3. Read detailed component docs (5-20 min)
4. Implement and test (varies)

## 🤝 Contributing

When modifying the codebase:

1. **Update docs**: If you change interfaces, update relevant .md files
2. **Test shape compatibility**: Run `print(tensor.shape)` at key points
3. **Document new features**: Add to QUICK_REFERENCE → Section 5
4. **Update metrics**: Log new metrics to TensorBoard

## 📞 Support

For issues:
1. Check QUICK_REFERENCE → Section 4 (Debugging Scenarios)
2. Verify shapes with DATA_FLOW → Section 11 (Quick Reference Table)
3. Review training pipeline in TRAINING_PIPELINE
4. Check existing documentation hasn't been updated

## 🔄 Version Information

This documentation was created for the MARL-LLM codebase as of April 2026.

**Key Dependencies**:
- JAX (GPU environment simulation)
- PyTorch (neural network training)
- NumPy (data conversion layer)
- TensorBoard (logging and visualization)

**Compatibility**:
- Python 3.8+
- CUDA 11.0+ (for GPU training)
- 16GB+ GPU memory recommended

---

## 📖 Documentation Philosophy

This documentation is optimized for **token efficiency** in AI agent workflows:

✅ **Hierarchical structure** — Start broad, drill down as needed  
✅ **Clear navigation** — Always know where to find information  
✅ **Complete examples** — Copy-paste-run code snippets  
✅ **Shape references** — Critical for debugging, always included  
✅ **Quick lookup** — Function signatures and common patterns  
✅ **Cross-references** — Links between related concepts  

**Goal**: An agent should be able to understand and modify the codebase by reading <20% of total documentation, targeted to their specific task.

---

**Ready to start?** → [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md)

the directory cfg_files_defaults contains the cfg setups for different things, the convention is the info will be encoded in the name. so i.e 1env_base_assembly_cfg.py, this part of the name will be common across files assembly_cfg.py, while the content before it 1env means only 1 env runs, base means the default everything else.unless you specifically wish to consult something ignore this direcotry.