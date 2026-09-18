# Five-round, three-role acceptance protocol

| Round | Role | Command / action | Passing evidence |
|---|---|---|---|
| 1 | Reader | `drosophila-repro doctor` | Registry and release assets resolve. |
| 2 | Reader | `drosophila-repro list` | Every final Figure 1–6 has a stated question and renderer registry. |
| 3 | Reproduction scientist | `drosophila-repro reproduce --figure figure5` | All registered Figure 5 computational panels render from frozen inputs. |
| 4 | Reproduction scientist | `drosophila-repro reproduce --figure figure4` | Deployment panels render and report expected frozen-data outputs. |
| 5 | Method developer | `drosophila-repro extension-init --name smoke_extension` | A separate extension scaffold is created without changing release assets. |

Rounds 6–8, required before public tagging: run Figure 1/2/3/6, open the
notebook in a clean environment, and perform visual/hash comparison against
the released Figure 1–6 composites plus manual panel exports.
