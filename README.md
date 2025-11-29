# The Dimensional Hierarchy of Cortical Oscillations

Code and figures for the paper: **"The Dimensional Hierarchy of Cortical Oscillations: From Analog Substrate to Symbolic Codes"**

## Abstract

We propose that cortical oscillations implement a *dimensional hierarchy*: a cascade of progressively tighter information bottlenecks from slow to fast frequencies. Slow eigenmodes engage substantially more oscillators than fast modes (r = -0.75), establishing the high-dimensional geometric substrate. Discrete symbolic codes emerge at a critical bottleneck width of k=2, while k=3 preserves continuous "compliant" dynamics capable of representing self-referential structures without trajectory collision.

| Band | Bottleneck | Topology | Function |
|------|-----------|----------|----------|
| Delta/Theta | k >> 3 | Volumetric | Raw substrate |
| Beta | k ≈ 3 | Compliant manifold | Manipulation, meta-cognition |
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
│   ├── laplacian_modes.py           # Graph Laplacian participation ratio
│   ├── code_formation_bottleneck.py # Bottleneck compression simulation
│   ├── paradox_topology.py          # Liar's Paradox + linear autoencoder proof
│   └── real_data_pr.py              # Synthetic validation
└── figures/
    ├── fig1_laplacian_participation.pdf   # Figure 1: Laplacian PR
    ├── fig2_synthetic_pr_validation.pdf   # Figure 2: Synthetic validation
    ├── fig3_code_formation_bottleneck.pdf # Figure 3: Bottleneck compression
    ├── fig4_paradox_topology.pdf          # Figure 4: Helix proof
    ├── fig5_stress_collapse.pdf           # Figure 5: Noise shifts optimal k
    ├── figS1_laplacian_robustness.pdf     # Figure 6: Robustness across topologies
    └── figS2_category_sweep.pdf           # Figure 7: Category sweep
```

## Running the Simulations

```bash
# Install dependencies
pip install numpy scipy matplotlib torch scikit-learn

# Generate all figures
cd src
python laplacian_modes.py
python code_formation_bottleneck.py
python paradox_topology.py
python real_data_pr.py
```

## Workflow

This project was developed using an AI-assisted workflow:
- **Primary drafting**: Claude Code (Opus 4.5) for manuscript text, simulations, and figures
- **Feedback/red-teaming**: GPT-5.1 Pro and Gemini 3 Pro for critical review
- **Integration**: Claude Code for implementing revisions

The author supervised direction, evaluated outputs, and made final decisions. AI assistance is disclosed in the manuscript's Statements and Declarations.

## Status

This repository accompanies the manuscript *"The Dimensional Hierarchy of Cortical Oscillations: From Analog Substrate to Symbolic Codes"* (submitted to Journal of Computational Neuroscience).

## License

MIT
