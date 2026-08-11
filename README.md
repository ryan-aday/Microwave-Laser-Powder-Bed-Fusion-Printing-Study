# Microwave vs Laser Metal Sintering / Melting Streamlit App

## Run

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## What is included

- Laser Gaussian heat-flux + Beer-Lambert absorption model
- Absorbed VED, normalized enthalpy, Péclet/diffusion metrics
- Microwave wavelength, bulk skin depth, effective loss, penetration and source-coupling diagnostics
- Shared sensible/latent heat target model
- **Elliptic paraboloid** processed-zone model: `V = 1/2*pi*a*b*d = pi*L*W*d/8`
- 3D Plotly melt/heated-zone visualization
- Thermal expansion and constrained elastic stress proxy
- Powder-size / sphericity sensitivity diagnostics
- Differential Evolution + SLSQP process optimization
- Random feasible sampling and time-vs-energy Pareto front
- Source links and explicit modeling limitations

## Important

The material presets are illustrative and editable. The model is a design-screening surrogate, not a substitute for temperature-dependent properties, calibrated absorptivity/coupling, EM cavity simulation, thermal FE, melt-pool CFD, process qualification, or safe machine engineering.
