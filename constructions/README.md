# Constructions

This directory contains curated shareable construction artifacts from the local
experiments.

## Best verified 5-color almost-coloring candidate

- Directory: `best_m018_2048/`
- Source run: `best128_paper_params_20260621_120017_1291273_m018_seed7000018_combo000`
- Verifier grid: `2048 x 2048`
- Verified bonus-color fraction: `3.86731625%`
- Solver: CBC, component MILP verifier

The package includes the trained model checkpoint, config, verifier CSV,
compressed fixed coloring grid, and PNG/PDF/SVG visualizations.

## Paper construction direct-fill improvement

- Directory: `paper_agent2_direct_fill_1945x1970/`
- Source: paper `grid.csv` and `parallelogram.csv`
- Grid: `1945 x 1970`
- Patch: 91 bonus-color cells recolored to real colors by Agent 2
- Verified bonus-color fraction: `3.733221980%`
- Independent discrete verifier: zero same-real-color unit-distance conflicts

The package includes the compact patched coloring grid, exact CSV/JSON patch,
independent verification output, and Agent 2's instruction/report artifacts
describing how the improvement was found.
