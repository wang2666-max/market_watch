# MarketWatch

A modular premarket market monitoring pipeline for ingesting market data, generating sector-level and equity-level signals, and producing structured daily reports.

---

## Overview

MarketWatch is an actively developed research and automation project designed to support systematic market monitoring and decision-making workflows.

The system pulls market data, computes cross-asset and equity-level signals, and generates structured daily artifacts that can be used for macro interpretation, idea generation, and downstream reporting.

This project is built with a focus on:

- modular design
- reproducibility
- scalability across data sources and signal pipelines

---

## Current Features

### 1. Market Data Pipeline
- Daily price ingestion via Polygon API
- Cross-asset proxy tracking (equities, rates, credit, commodities, crypto, volatility)

### 2. Sector & Macro Snapshot
Generates a daily sector overview including:

- S&P 500 (SPY)
- Nasdaq (QQQ)
- Gold (GLD)
- Oil (USO)
- Long Treasuries (TLT)
- Credit (LQD / HYG)
- Dollar (UUP)
- Volatility proxy (VXX)
- Sector ETFs (XLE, XLF)
- Thematic indices (SOXX, IBIT)

Example output:
SECTOR OVERVIEW:
         symbol                    name     ret_d
            GLD                    Gold  0.009681
            HYG       High Yield Credit  0.000251
           IBIT                 Bitcoin -0.010684
            LQD Investment Grade Credit  0.001101
            QQQ                  Nasdaq  0.000153
           SOXX                   Chips  0.010580
            SPY                 S&P 500  0.000440
            TLT         Long Treasuries -0.000114
            USO                     Oil -0.006209
            UUP               US Dollar -0.002880
            VXX                     VIX  0.033094
            XLE                  Energy  0.008010
            XLF              Financials  0.000000
Treasury HY spr                         -0.000366


### 3. Equity Signal Engine

Daily screening across a dynamic universe using:

- rolling return features (daily / weekly / monthly)
- EWMA-based relative strength
- cross-sectional ranking vs benchmark (SPY)
- z-score normalization

Produces outputs such as:

- Top relative strength names
- abnormal movement flags
- volatility spike detection

Example:
VolSpikeFlags: (none)
RecentAbnormalFlags: PANW

Top 10 by ratio_ewma_vs_21:
symbol     ret_d    ret_w     ret_m  ratio_ewma_vs_21  ratio_ewma_vs_spy  z_last_21
    AI -0.004571 0.120668 -0.050054          1.319018           3.985763  -0.041159
  IBIT -0.010684 0.037686  0.012953          1.292378           2.827539  -0.385805
  IONQ -0.025984 0.071455 -0.202631          1.274176           4.666264  -0.418691
    ZS  0.018253 0.035189 -0.133914          1.236708           3.321655   0.904858
  PSTG  0.011702 0.085980  0.016256          1.185360           3.251267   0.406092
  RGTI -0.025679 0.072868 -0.186361          1.151871           4.442896  -0.424922
  QBTS -0.028696 0.058552 -0.260893          1.133464           4.888136  -0.332967
  CSCO  0.003041 0.052842  0.031420          1.128684           1.877164   0.141236
   NET  0.021072 0.111288  0.108100          1.100167           3.276876   0.528522
  ZETA -0.005709 0.061445 -0.165605          1.098341           3.966300   0.074190


### 4. Report Generation

- Structured text-based artifacts
- Organized by date
- Designed for downstream email / dashboard integration

### 5. Project Structure

MarketWatch/
├─ jobs/
│  └─ premarket.py              # orchestration entry point
├─ src/
│  ├─ common/                  # shared helpers
│  ├─ prices/                  # market data ingestion (Polygon)
│  ├─ options/                 # options pipeline (in progress)
│  ├─ news/                    # news parsing (in progress)
│  ├─ reddit/                  # sentiment pipeline (in progress)
│  ├─ processor/               # report assembly / formatting
│  └─ utility/                 # misc utilities (dates, paths, etc.)
├─ .env.example
├─ .gitignore
├─ requirements.txt
└─ README.md

### 6. Project Stratus
Project Status

This is an actively developed project.

Currently implemented:
- Polygon-based price ingestion
- sector / macro overview
- equity signal engine
- report artifact generation

In progress:
- options analytics module
- news parsing pipeline
- Reddit sentiment ingestion
- enhanced reporting and distribution layer

### 7. Notes

-Data and generated reports are intentionally excluded from the repository
-API keys and secrets are managed through environment variables
-Output shown above is a sample from the current working pipeline