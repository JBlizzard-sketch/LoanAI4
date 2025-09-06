
---

📊 LoanIQ — Credit Scoring & Loan Risk Platform

A Streamlit-based microfinance credit scoring and loan risk analysis platform, designed for Kenyan microfinance institutions.

Supports two roles:

Clients → register/login, upload or generate loan data, view dashboards, get credit scores, risk insights, and downloadable reports.

Admins → access a secure sandbox with full control over synthetic data generation, ML engine training/versioning, fraud stress tests, and more.



---

🚀 Features

👩‍💼 Clients

Register/login (auto-login after registration).

Upload loan datasets (CSV/Excel).

Or generate synthetic data reflecting Kenyan microfinance distributions:

Loan sizes (KES 5k–100k, skewed small).

Branches across 70+ Kenyan counties/towns.

Occupations (mama mboga, shop owner, boda boda, farmer, etc.).

Gender bias toward women & small businesses.


View dashboards:

Loan eligibility & repayment risk.

Credit scores (scaled 300–900).

Fraud flags & repayment behavior.

Rich visualizations (histograms, scatter plots, metrics panels).


Download results:

Scored dataset (CSV).

Credit report (PDF with key insights).




---

🔐 Admin Sandbox

Generate/preview synthetic datasets with adjustable distributions (women bias, fraud rate, loan size skew).

Train/retrain six ML families:

Logistic Regression

RandomForest

GradientBoosting

XGBoost

LightGBM

Hybrid (ensemble of RF + GB)


Track model versioning & metrics (AUC, accuracy, recall).

Auto-deploy best model by AUC.

Utilities:

Database backup/restore.

Impersonate client dashboards.

Fraud stress test (simulate adversarial scenarios).




---

🧠 Machine Learning Engine

Robust preprocessing (handles missing demographic features).

One-hot encoding of categorical features.

Metrics stored in SQLite for version history.

Auto-deployment pipeline triggered from Streamlit.



---

🖥️ User Interface

Built with Streamlit (clean, responsive layout).

Role-based navigation (clients vs admins).

Rich Plotly charts, metrics, and interactive dashboards.

Styled buttons, tags, and download controls.



---

📦 Installation & Running

Clone & Install

git clone https://github.com/your-repo/loaniq.git
cd loaniq
pip install -r requirements.txt

Run Streamlit App

streamlit run app.py --server.port 3000 --server.address 0.0.0.0

Replit

The app auto-starts in Replit thanks to .replit:

run = "streamlit run app.py --server.port 3000 --server.address 0.0.0.0"


---

🔑 Login Credentials

Admin → admin / Shady868...

Client → Register a new account (auto-login).



---

📂 Project Structure

├── app.py                # Main Streamlit app
├── utils/                # Core modules
│   ├── auth.py           # Authentication
│   ├── db.py             # SQLite storage & schema
│   ├── ml.py             # ML training, versioning, deployment
│   ├── report.py         # PDF report builder
│   ├── synth.py          # Synthetic data generator
│   └── ui.py             # UI helpers
├── models/               # Saved models
├── data/                 # SQLite DB, sample CSVs
├── pages/                # Optional Streamlit pages
├── assets/               # Images / visuals
├── requirements.txt      # Dependencies
├── .replit               # Replit startup config
└── README.md             # This file


---

🧪 Quick Demo

1. Login as admin and generate a synthetic dataset.


2. Train all 6 models → best model auto-deploys.


3. Switch to client role → upload/generate data → run predictions.


4. Download your credit report (CSV & PDF).




---

🌍 Context

Designed for Kenyan microfinance use cases.

Skewed toward small loan sizes and women-led businesses.

Simulates fraud patterns for robustness.



---

✨ Roadmap

[ ] Integrate OpenAI/NLP for natural-language explanations.

[ ] Add multi-client datasets for institutions.

[ ] Expand fraud detection stress-testing.

[ ] Improve PDF report styling.



---

📜 License

MIT License — free to use and adapt.


---



