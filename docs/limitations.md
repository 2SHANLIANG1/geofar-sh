# Limitations

GeoFAR-SH is an appearance refinement method, not a geometry repair method.

Known limitations:

- It cannot fix incorrect Gaussian positions.
- It cannot fix insufficient density or missing geometry.
- It cannot repair severe geometry errors.
- It cannot recover entirely missing views or unobserved content.
- It should not be used to disguise geometry errors as colour corrections.
- It is better suited as post-convergence appearance refinement after the base 3DGS geometry is already reasonable.
- The residual branch adds parameters and may introduce overhead compared with a pure 3DGS model.
- The exact CUDA/PyTorch build may require adaptation to local GPU drivers and toolkits.

These limitations should be reported alongside quantitative and qualitative results.



