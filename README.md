# Algebraic Graph Theory Engine

Computes advanced algebraic invariants of the ETF correlation graph:

- **Weisfeiler‑Lehman colouring** (approximates automorphism orbits)
- **Characteristic polynomial** of adjacency matrix
- **Ihara zeta function** (evaluated at u=0.1)
- **Ramanujan graph test** (spectral gap)
- **Algebraic score** = (1 / orbit_size) × eigenvector_centrality

Higher score indicates ETFs that are structurally unique and centrally important. Multi‑window evaluation selects the best window per ETF.

- **Graph:** absolute correlation > threshold (0.5)
- **Algebraic invariants:** spectral gap, Ramanujan, Ihara zeta, orbits
- **Windows:** 63, 252, 504, 1008, 2016 days (best per ETF)
- **Output:** top 3 ETFs per universe

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
