# Public-release gates

The current repository passes code/registry and Figure 1–6 computational-panel
acceptance in the development archive. It is not yet publicly taggable until:

1. An immutable data archive is deposited and its DOI/checksum is recorded.
2. Editable manual PowerPoint panels, panel exports, and final Figure 1–6/S1–S5
   composites are included or linked in a versioned visual-asset archive.
3. The DeepISA derivative redistribution license is resolved. As checked on
   2026-09-18, the public upstream repository has neither a `LICENSE` file nor
   a license declaration in `pyproject.toml`; exclude `Ep_ISA_NEW_src/` from a
   public data archive unless its authors grant permission. Then add the
   explicit project `LICENSE` and `CITATION.cff`.
4. The notebook and `--figure all` run from a clean clone against the downloaded
   release assets.
5. Rebuilt exports are visually/hash-compared with the tagged manuscript
   composites.
