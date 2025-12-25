# The Dimensional Hierarchy of Cortical Oscillations

**Journal:** J. Computational Neuroscience
**Status:** Under review (submitted Nov 29, 2025)

## Abstract

We propose that cortical oscillations implement a *dimensional hierarchy*: a cascade of progressively tighter information bottlenecks from slow to fast frequencies. Slow eigenmodes engage substantially more oscillators than fast modes (r = -0.75), establishing the high-dimensional geometric substrate. Discrete symbolic codes emerge at a critical bottleneck width of k=2, while k=3 preserves continuous "compliant" dynamics capable of representing self-referential structures without trajectory collision.

| Band | Bottleneck | Topology | Function |
|------|-----------|----------|----------|
| Delta/Theta | k >> 3 | Volumetric | Raw substrate |
| Beta | k ≈ 3 | Compliant manifold | Manipulation, meta-cognition |
| Gamma | k ≈ 2 | Discrete clusters | Symbols, decisions |

## Key Results

1. **Laplacian Analysis**: Slow modes engage 3× more oscillators than fast modes
2. **Bottleneck Compression**: k=2 forces discrete symbols; k≥3 preserves analog dynamics
3. **Paradox Topology**: Self-reference produces 1511 collisions at k=2, zero at k=3 (helix)

## Running Simulations

```bash
pip install numpy scipy matplotlib torch scikit-learn
cd src && python laplacian_modes.py
```

## GitHub

https://github.com/todd866/brainwavedimensionality
