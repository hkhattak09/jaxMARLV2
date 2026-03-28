#!/bin/bash
# Cleanup script for JaxMARL - keep only essentials

cd /Users/hassan/repos/jaxMARL/JaxMARL

echo "Cleaning up JaxMARL directory..."

# Delete all training baselines
echo "Removing baselines..."
rm -rf baselines

# Delete unnecessary environments
echo "Removing unused environments..."
rm -rf jaxmarl/environments/coin_game
rm -rf jaxmarl/environments/hanabi
rm -rf jaxmarl/environments/jaxnav
rm -rf jaxmarl/environments/mabrax
rm -rf jaxmarl/environments/overcooked
rm -rf jaxmarl/environments/overcooked_v2
rm -rf jaxmarl/environments/robotarium
rm -rf jaxmarl/environments/smax
rm -rf jaxmarl/environments/storm
rm -rf jaxmarl/environments/switch_riddle

# Keep only simple.py as reference, delete other MPE variants
echo "Cleaning MPE directory..."
cd jaxmarl/environments/mpe
rm -f simple_adversary.py
rm -f simple_crypto.py
rm -f simple_facmac.py
rm -f simple_push.py
rm -f simple_reference.py
rm -f simple_speaker_listener.py
rm -f simple_spread.py
rm -f simple_tag.py
rm -f simple_world_comm.py
rm -f mpe_visualizer.py
rm -f ReadMe.md
cd ../../..

# Delete gridworld (not needed)
echo "Removing gridworld..."
rm -rf jaxmarl/gridworld

# Delete viz (not needed for now)
echo "Removing visualization..."
rm -rf jaxmarl/viz

# Delete tutorials
echo "Removing tutorials..."
rm -rf jaxmarl/tutorials

# Delete tests
echo "Removing tests..."
rm -rf tests

# Delete docs and site
echo "Removing documentation..."
rm -rf docs site

# Delete CI/CD
echo "Removing CI/CD..."
rm -rf .github

# Delete build/packaging files
echo "Removing build files..."
rm -f Dockerfile Makefile .dockerignore
rm -f pyproject.toml MANIFEST.in mkdocs.yml
rm -f CONTRIBUTING.md CHANGELOG.md

# Keep only essential wrappers (baselines.py)
echo "Cleaning wrappers..."
cd jaxmarl/wrappers
rm -f gymnax.py transformers.py
cd ../..

echo "Cleanup complete!"
echo ""
echo "Remaining structure:"
find jaxmarl -name "*.py" | grep -v __pycache__ | sort
