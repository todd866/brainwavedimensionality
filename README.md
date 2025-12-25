# The Dimensional Hierarchy of Cortical Oscillations

**Journal:** Journal of Computational Neuroscience
**Status:** Under review (submitted Nov 29, 2025)

## Repository Structure

```
├── slow_waves_high_D.tex    # Original submission (Nov 2025)
├── slow_waves_high_D.pdf    # Original compiled PDF
├── references.bib           # Original references
├── code/                    # Simulation code
│   ├── laplacian_modes.py
│   ├── code_formation_bottleneck.py
│   ├── paradox_topology.py
│   ├── paradox_learning.py
│   └── real_data_pr.py
├── figures/                 # Original figures
└── _revision/               # R1 revision (in progress)
    ├── slow_waves_high_D.tex
    ├── slow_waves_high_D.pdf
    ├── references.bib
    └── figures/
```

**Root folder** contains the original submission to JCN (Nov 2025) and all associated code/figures.

**`_revision/`** contains the R1 revision addressing reviewer feedback:
- Added terminology box clarifying participation ratio ≠ latent dimension
- Reframed eigenvalue as "spatial smoothness index" (not frequency)
- Reframed synthetic PR test as "demonstration" not "validation"
- Generalized "Liar's Paradox" to "periodic process with hidden context"
- Changed emotion from strict "rank-1" to "low-rank (1-3)"
- Shifted "maturity" language to "regulatory capacity" (less normative)
- Added formal tri-level notation (substrate/interface/expressed)
- Tied concentration of measure to effective N ~ PR, not raw neuron count
- Added Eckmann-Ruelle (1985) citation for dimension-entropy distinction
- Integrated Chen et al. (2026) "spatial computing" empirical support
- Added limitation notes on ensemble robustness and seed variability

## Abstract

We argue that the framing of slow oscillations as "low-dimensional control" commits a category error by conflating three distinct notions: *substrate dimensionality* (how many oscillators participate coherently), *interface dimensionality* (how many degrees of freedom a readout exposes), and *expressed structure* (the complexity of downstream patterns).

Using graph Laplacian analysis, we show that slow eigenmodes engage substantially more oscillators than fast modes (r = -0.75), establishing slow waves as high-participation substrates. Using encoder-decoder networks, we demonstrate that k=2 produces discrete categorical codes while k≥3 preserves continuous dynamics capable of representing periodic processes without cycle aliasing.

## Key Results

1. **Laplacian Analysis**: Slow modes engage 3× more oscillators than fast modes
2. **Bottleneck Compression**: k=2 forces discrete symbols; k≥3 preserves analog dynamics
3. **Cycle Aliasing**: Periodic processes produce aliasing at k=2, none at k≥3 (helix geometry)

## Running Simulations

```bash
pip install numpy scipy matplotlib torch scikit-learn
cd code && python laplacian_modes.py
```

## License

MIT
