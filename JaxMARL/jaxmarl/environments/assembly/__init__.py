"""
Assembly Swarm Environment for JaxMARL.

Multi-agent environment where agents must assemble into target formations.
Based on the MARL-LLM assembly environment but implemented in JAX.
"""

from jaxmarl.environments.assembly.assembly_env import AssemblyEnv

__all__ = ["AssemblyEnv"]
