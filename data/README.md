# Dataset Setup

## US Accidents Dataset (Kaggle)

This project uses the US Accidents dataset as a public surrogate for the original TxDOT CRIS EV crash dataset used in the IEEE ICMLA 2025 paper.

### Download Instructions

1. Visit: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
2. Download: `US_Accidents_March23.csv`
3. Place the file in: `data/US_Accidents_March23.csv`

### Dataset Overview

- **Source**: Kaggle - US Accidents (2016-2023)
- **Records**: ~7.7 million accident records across 49 US states
- **Features**: Environmental conditions, location, time, road attributes
- **Target Variable**: Severity (1-4 scale)
  - Severity 1: Minor impact
  - Severity 2: Moderate impact
  - Severity 3: Serious impact
  - Severity 4: Fatality involved

### Attribution

This dataset is a public surrogate. The original research used TxDOT CRIS (Crash Records Information System) data.
