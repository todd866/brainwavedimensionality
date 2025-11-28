# The Dimensional Hierarchy of Cortical Oscillations

Code and figures for the paper: **"The Dimensional Hierarchy of Cortical Oscillations: From Analog Substrate to Symbolic Codes"**

## Abstract

We propose that cortical oscillations implement a *dimensional hierarchy*: a cascade of progressively tighter information bottlenecks from slow to fast frequencies. Slow eigenmodes engage substantially more oscillators than fast modes (r = -0.75), establishing the high-dimensional geometric substrate. Discrete symbolic codes emerge at a critical bottleneck width of k=2, while k=3 preserves continuous "floppy" dynamics capable of representing self-referential structures without trajectory collision.

| Band | Bottleneck | Topology | Function |
|------|-----------|----------|----------|
| Delta/Theta | k >> 3 | Volumetric | Raw substrate |
| Beta | k ≈ 3 | Floppy manifold | Manipulation, meta-cognition |
| Gamma | k ≈ 2 | Discrete clusters | Symbols, decisions |

## Key Results

1. **Laplacian Analysis**: Slow modes are geometrically high-dimensional (3× more oscillators than fast modes)
2. **Bottleneck Compression**: k=2 forces discrete symbol formation; k≥3 preserves analog dynamics
3. **Paradox Topology**: Self-referential logic (Liar's Paradox) produces 1511 collisions in k=2 but 0 in k=3 (helix)

## Repository Structure

```
├── paper/
│   ├── slow_waves_high_D.tex    # Main manuscript
│   ├── slow_waves_high_D.pdf    # Compiled PDF
│   └── references.bib           # Bibliography
├── src/
│   ├── laplacian_pr.py          # Graph Laplacian participation ratio
│   ├── code_formation.py        # Bottleneck compression simulation
│   ├── paradox_topology.py      # Liar's Paradox collision dynamics
│   └── real_data_pr.py          # Synthetic validation
└── figures/
    ├── fig1_laplacian_participation.pdf
    ├── fig2_code_formation_bottleneck.pdf
    ├── fig3_synthetic_pr_validation.pdf
    └── fig4_paradox_topology.pdf
```

## Running the Simulations

```bash
# Install dependencies
pip install numpy scipy matplotlib torch scikit-learn

# Generate all figures
cd src
python laplacian_pr.py
python code_formation.py
python paradox_topology.py
python real_data_pr.py
```

## Status

Manuscript in preparation. Target journal: Journal of Computational Neuroscience.

## License

MIT
