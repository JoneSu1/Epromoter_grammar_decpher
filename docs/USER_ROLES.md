# User roles

| Role | First command | Success criterion | What they may change |
|---|---|---|---|
| Reader/reviewer | `drosophila-repro doctor` | Assets and registry resolve; scope is explicit | Nothing required |
| Reproduction scientist | `drosophila-repro reproduce --figure figure5` | Registered computational panels are regenerated from frozen inputs | Outputs in a separate run directory or release export area |
| Method developer | `drosophila-repro extension-init --name my_analysis` | A new analysis has an explicit question, input/output contract, and validation placeholder | Only extension files; never frozen inputs |

All roles use the same `configs/figure_registry.json`; that registry is the
single interface between paper figure numbers and package-local renderers.
