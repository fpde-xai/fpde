# Release Notes

This file summarizes user-visible package changes.

## v0.1.0

Initial open-source package layout for FPDE.

### Added

- FPDE implementation under `src/fpde/`.
- Public APIs for Diff-FPDE, Cos-FPDE, prototype helpers, metrics, and
  `FPDEEngine`.
- Hyb-FPDE grid search and validation-based lambda selection through
  `FPDEEngine`.
- Minimal scikit-learn example.
- Pytest coverage for core behavior.
- Apache License 2.0 license text.
- Citation metadata.
- Repository metadata documentation.

### Known Limitations

- The repository focuses on the core package, examples, and tests.
- The package currently provides class-mean prototypes.
