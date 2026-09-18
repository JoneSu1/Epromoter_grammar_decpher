# Acceptance evidence (development archive)

This log records executed evidence, not planned checks. Paths refer to the
current source archive; a public tag must repeat the protocol from a clean
clone plus its deposited release assets.

| Round | Role | Executed evidence | Result |
|---|---|---|---|
| 1 | Reader/reviewer | Editable install, then `drosophila-repro doctor` | PASS after correcting repository-root resolution. Registry and candidate assets resolve. |
| 2 | Reader/reviewer | `drosophila-repro registry-check`; `pytest -q` | PASS: registry check clean; 3 registry contract tests pass. |
| 3 | Reproduction scientist | `reproducibility_package/quickstart/reproduce.py --group figure5` | PASS: five DeepISA panels regenerated from frozen local inputs. A path adapter resolved the registered Figure 5 distance/context summary. |
| 4 | Reproduction scientist | `reproducibility_package/quickstart/reproduce.py --group figure4` | PASS: TSS, density/complexity, and contribution panels regenerated. Warnings were rendering-library warnings, not failures. |
| 5 | Method developer | `drosophila-repro extension-init --name smoke_extension` in a clean temporary directory | PASS: isolated `README.md` and `analysis.py` scaffold created without modifying release assets. |
| 6 | Reproduction scientist | `drosophila-repro reproduce --figure figure5 --dry-run` | PASS: the public CLI resolves all five Figure 5 targets. |
| 7 | Reproduction scientist | `drosophila-repro reproduce --figure figure4 --dry-run` | PASS: the public CLI resolves all three Figure 4 targets. |
| 8 | Reproduction scientist | `drosophila-repro reproduce --figure figure5` through the installed CLI | PASS: all Figure 5 registered computational panels rerendered; output of the distance/context panel was written successfully. |
| 9 | Reproduction scientist | `drosophila-repro reproduce --figure figure1` | PASS: DeepSTARR annotation, activity-class, and predicted-activity panels regenerated. |
| 10 | Reproduction scientist | `drosophila-repro reproduce --figure figure2` | PASS: DeepCAGE prediction and correlation panels regenerated, including QA tables. |
| 11 | Reproduction scientist | `drosophila-repro reproduce --figure figure3` after package-local adapters were added | PASS: similarity, motif-logo, distribution, and core-promoter logo panels regenerated from localized frozen assets. |
| 12 | Reproduction scientist | Figure 6 main renderer plus localized branch-plate and motif-bar adapters | PASS: main evolution outputs regenerated; 60/60 HK/DEV branch-plate exports completed; motif-gain bars regenerated. |
| 13 | Reader/reviewer + reproduction scientist | `drosophila-repro doctor`, `registry-check`, `pytest -q`, and `reproduce --figure all --dry-run --continue-on-error` after adding the release-data mount point | PASS: the empty mount point no longer shadows the audited development archive; all six figure groups resolve, and 3/3 registry tests pass. |
| 14 | Method developer | `python -m pip wheel . --no-deps --wheel-dir dist` | PASS: a standards-compliant pure-Python wheel was built (`drosophila_promoter_repro-0.1.0-py3-none-any.whl`). |
| 15 | Reader/reviewer | `drosophila-repro assets-verify` | PASS: all six final Figure 1–6 composites, S1–S5 exports, editable PPTX/PDF sources, byte counts, and SHA-256 hashes match the frozen visual inventory. |
| 16 | Release maintainer | ZIP64 asset build, sidecar SHA-256, and central-directory audit | PASS: the public asset archive has 2,066 members, exceeds 4 GiB safely, contains all registered Figure 5 and visual-asset paths, and contains zero `Ep_ISA_NEW_src` members. |
| 17 | Reproduction scientist | Install the built wheel into an isolated target, use an independently extracted ZIP asset root, then run `doctor`, `assets-verify`, and `reproduce --figure all --dry-run --continue-on-error` | PASS: the installed wheel reads its packaged registry (not the checkout), the extracted archive resolves, final visual hashes pass, and every Figure 1–6 command resolves. |
| 18 | Release maintainer | Create an asset-free Git archive in an unrelated temporary directory; force imports from its own `src/`; run tests and build a wheel | PASS: 4 code-level tests pass, 2 asset-mount tests skip by design, and the wheel builds. This matches the GitHub Actions workflow rather than accidentally reading the development archive. |

## Remaining public-tag evidence

Execute the GitHub notebook from a clean environment; deposit and test the
public ZIP at its persistent DOI; and compare all rebuilt exports plus manual
PPTX panels to the final manuscript composites.
