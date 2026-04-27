# HAPPO-HAP: Heterogeneous-Agent PPO with Hidden-State Advantage Priors and Low-Rank Agent-Specific Adaptation

---

## Problem

MAPPO has two fundamental weaknesses in cooperative MARL. First, simultaneous agent updates produce conflicting gradients with no monotonic improvement guarantee on the joint return. Second, the shared policy and shared advantage signal treat agents as identical when they are not — suppressing specialization and misattributing credit.

HAPPO fixes the first problem through sequential updates with conditioned advantages. But this fix is only as strong as the advantage estimates driving it. Early in training, the recurrent critic produces noisy, unreliable estimates. This means HAPPO's sequential update — which depends on accurate advantages to compute conditioned updates — is theoretically sound but empirically undermined throughout early training. Nobody has addressed this directly.

Additionally, HAPPO retains the shared policy, leaving the specialization problem unsolved.

---

## Method

**HAPPO-HAP**: Recurrent HAPPO augmented with a Hidden-State Advantage Prior and Low-Rank Agent-Specific Adaptation, evaluated on SMAX in JaxMARL.

---

### Component 1 — Recurrent HAPPO (foundation)

Replace MAPPO's simultaneous update with HAPPO's sequential update scheme. Agents update one at a time in each training round. Each agent $i$'s advantage is conditioned on the ratio corrections of already-updated agents $j < i$:

$$A_i^{HAPPO} = \frac{\prod_{j<i} \pi_j^{new}(a_j|o_j)}{\prod_{j<i} \pi_j^{old}(a_j|o_j)} \times A^t$$

This gives a provable monotonic improvement guarantee on the joint return that vanilla MAPPO lacks. Both actor and critic are recurrent — GRU-based — to handle SMAX's partial observability.

---

### Component 2 — Hidden-State Advantage Prior (our first contribution)

A small auxiliary network $f_\phi$ takes each agent's GRU hidden state $h_i^t$ and predicts a scalar advantage prior:

$$\hat{A}_i^t = f_\phi(h_i^t)$$

The hidden state already encodes rich local signal — health, nearby enemies, teammate positions — before the critic has warmed up. The prior network is trained with a fast auxiliary objective predicting n-step returns from hidden states, which has far lower variance than full GAE early in training.

The effective advantage used in the HAPPO update blends GAE and the prior:

$$A_i^{eff} = (1 - \alpha_t) \cdot A_i^{HAPPO} + \alpha_t \cdot \hat{A}_i^t$$

Where $\alpha_t$ anneals linearly from 1 to 0 over the first third of training. Early on the prior stabilizes the sequential update. As the critic warms up GAE takes over completely and the method reduces to standard recurrent HAPPO. At convergence there is zero overhead.

---

### Component 3 — Low-Rank Agent-Specific Adaptation (our second contribution)

The shared recurrent policy suppresses specialization — on SMAX maps with heterogeneous units, a marine and a medic should behave differently even in similar situations, but a shared policy resists this.

Rather than training fully separate policies per agent — which is expensive and data-inefficient — we add lightweight low-rank adapter matrices to each agent on top of the shared backbone, following LoRASA. For each agent $i$ and each weight matrix $W$ in the shared policy:

$$W_i = W + \Delta W_i = W + B_i A_i$$

Where $B_i \in \mathbb{R}^{d \times r}$ and $A_i \in \mathbb{R}^{r \times d}$ with rank $r \ll d$. The shared weights $W$ capture common behavior across all agents. The adapter $\Delta W_i$ captures agent-specific deviations.

All agents share the same backbone weights updated from joint experience. Each agent has its own tiny adapter updated from its own trajectories only. This gives heterogeneous behavior at a fraction of the parameter cost of separate policies.

---

## Philosophical Coherence

Both contributions address the same root problem from different angles:

> Shared MAPPO machinery treats agents as identical when they are not.

The hidden-state prior individualizes the **advantage signal** — each agent gets an advantage estimate informed by its own local hidden state rather than purely a shared global signal.

The low-rank adapters individualize the **policy** — each agent can specialize its behavior on top of a shared foundation.

Together they form a coherent method: sequential updates for joint improvement, individualized advantages for stable early training, individualized policies for specialization.

---

## Key Claim

HAPPO-HAP produces significantly better **sample efficiency** than recurrent MAPPO and recurrent HAPPO on SMAX — reaching equivalent win rates in fewer environment steps — with the largest gains on harder maps where agent coupling is tightest, early advantage noise is most damaging, and unit heterogeneity is highest.

---

## Differentiation from Existing Work

| Paper | How we differ |
|---|---|
| HAPPO | We build on it, fixing its early training brittleness with the hidden-state prior |
| MACA | They do counterfactual credit assignment at convergence via attention. We stabilize advantage estimation throughout early training via hidden states |
| PMAT | They learn action generation order inside a transformer. We stabilize advantage estimation in sequential policy updates |
| LoRASA | They apply low-rank adaptation to MAPPO and A2PO. We apply it to HAPPO and pair it with our prior, motivated jointly by the individualization argument |
| ATM | Transformer memory for partial observability. Orthogonal, not competing |

---

## One Paragraph for Your Meeting

"HAPPO improves on MAPPO by updating agents sequentially with conditioned advantages, giving a monotonic improvement guarantee on the joint return. However this guarantee is only as strong as the advantage estimates driving it — early in training a recurrent critic cannot provide reliable estimates, undermining the sequential update empirically. We propose HAPPO-HAP, which augments HAPPO with two contributions motivated by a single principle: shared MAPPO machinery treats agents as identical when they are not. First, a lightweight auxiliary network predicts per-agent advantage priors from GRU hidden states, stabilizing early sequential updates through annealed mixing with GAE. Second, low-rank agent-specific adapters on the shared policy backbone allow heterogeneous behavior without the cost of fully separate policies. On SMAX we show this produces significantly better sample efficiency than recurrent MAPPO and recurrent HAPPO, with the largest gains on harder maps where our contributions matter most."