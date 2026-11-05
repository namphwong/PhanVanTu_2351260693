# Self-supervised Autoregressive Domain Adaptation for Time Series Data (SLARDA) [[Paper](https://arxiv.org/abs/2111.14834)]
#### *by: Mohamed Ragab, Emadeldeen Eldele, Zhenghua Chen, Min Wu, Chee Keong Kwoh, and  Xiaoli Li*
#### IEEE Transactions on Neural Networks and Learning Systems (TNNLS-22) (Impact Factor: 10.451).

## Abstract
<img src="SLARDA_1.PNG" width="1000">
Unsupervised domain adaptation (UDA) has successfully addressed the domain shift problem for visual applications. Yet, these approaches may have limited performance for time series data due to the following reasons. First, they mainly rely on the large-scale dataset (i.e., ImageNet) for the source pretraining, which is not applicable for time-series data. Second, they ignore the temporal dimension on the feature space of the source and target domains during the domain alignment step. Last, most of the prior UDA methods can only align the global features without considering the fine-grained class distribution of the target domain. To address these limitations, we propose a Self-supervised autoregressive Domain Adaptation (SLARDA) framework. In particular, we first design a self-supervised learning module that utilizes forecasting as an auxiliary task to improve the transferability of the source features. Second, we propose a novel autoregressive domain adaptation technique that incorporates temporal dependency of both source and target features during domain alignment. Finally, we develop an ensemble teacher model to align the class-wise distribution in the target domain via a confident pseudo labeling approach.
Extensive experiments have been conducted on three real-world time series applications with 30 cross-domain scenarios. Results demonstrate that our proposed SLARDA method significantly outperforms the state-of-the-art approaches for time series domain adaptation. Our source code is available at: \href{https://github.com/mohamedr002/SLARDA}{https://github.com/mohamedr002/SLARDA}.

## Requirmenets:
- Python3.x
- Pytorch==1.7
- Numpy
- Sklearn
- Pandas
- mat4py (for Fault diagnosis preprocessing)

### A- Download the dataset from this following links
    - MFD: https://mb.uni-paderborn.de/en/kat/main-research/datacenter/bearing-datacenter/data-sets-and-download
    - HAR: https://archive.ics.uci.edu/ml/datasets/OPPORTUNITY+Activity+Recognitio
    - SSC:https://sleepdata.org/datasets/shhs, https://physionet.org/content/sleep-edf/1.0.0/
### B-  Add dataset to the framework
1- Add the data files in the following format:

    - Each domain splitted to train, val, test files
    - name the domains using small letters, i.e., a, b, c,...
    - Each sample has the following dict format: 
        - samples = data['samples']
        - labels = data['labels']
### Running the model 

1- from args:
    - Select the domain adaptation method as 'SLARDA'
    - Select your target dataset: Paderborn_FD, HAR, EEG
    - Select the corresponding based model: CNN_SL_bn, CNN_Opp_HAR_SL, EEG_M_SL

2- Run 'train_CD.py' script

## Updated Architecture

The rain forecasting adaptation plan is being implemented in phases on the
`codex/rain-dinov3` branch. The task predicts whether precipitation at the next
hourly timestamp is at least 0.1 mm, using Hanoi as the source domain and Hai
Phong as the target domain. See `configs/rain_hanoi_haiphong.yaml` for the
experiment contract and `docs/legacy-audit.md` for the Phase 0 review of the
original code.

### Development setup

Use Python 3.10, 3.11, or 3.12 for the legacy SLARDA code. Install the project and
development tools with `python -m pip install -e ".[dev]"`, then run
`python -m pytest -q`. The original implementation targeted PyTorch 1.7; this
branch upgrades the compatibility baseline to PyTorch 2.5 or newer. The legacy
smoke test uses synthetic data and does not download datasets or checkpoints.

### Phase checklist

- [x] Phase 0: scope, environment, and legacy smoke test
- [x] Phase 1: hourly weather data and Kaggle publication
- [x] Phase 2: baseline and SLARDA evaluation
- [ ] Phase 3: DINOv3 encoder integration
- [ ] Phase 4: Streamlit application
- [ ] Phase 5: Docker image publication
- [ ] Phase 6: CI/CD, deployment, and handoff guide

### Data pipeline

Phase 1 retrieves hourly ERA5 reanalysis for Hanoi and Hai Phong from 2023
through 2025, stores yearly raw JSON with checksums, and creates hourly and
24-hour window datasets. ERA5-Land was probed but returned nulls for several
required fields, so the pipeline uses the fixed ERA5 model for both cities.
The dataset is published at
https://www.kaggle.com/datasets/phantus/vietnam-hourly-rain-next-hour-domain-adaptation.
To publish a new version from the command line, install publishing tools with
`uv sync --extra publish` and configure Kaggle credentials locally, then run
`uv run python scripts/publish_kaggle.py --username <kaggle-username>`. The
installed CLI accepts an API key file at
`%USERPROFILE%\.kaggle\kaggle.json`, or an access token at
`%USERPROFILE%\.kaggle\access_token` / `KAGGLE_API_TOKEN`. Do not commit or
share those credentials.

```powershell
uv run python scripts/collect_open_meteo.py --config configs/rain_hanoi_haiphong.yaml
uv run python scripts/prepare_dataset.py --config configs/rain_hanoi_haiphong.yaml
```

Dataset provenance, schema, splits, limitations, license, and attribution are
documented in `data/DATA_CARD.md`. Raw and processed data stay out of Git. The
Kaggle dataset contains the extracted raw and processed files, data card,
license, and citation. Current counts, checksums, and verification results are
in `docs/phase1-report.md`.

### Phase 2: baselines and CNN SLARDA

Run the reproducible Phase 2 experiment from the repository root:

```powershell
python scripts/run_phase2.py
```

The run evaluates B0 persistence and trains the B1 logistic-regression
reference, then fits B2 source-only CNN and B3 CNN + SLARDA for the three
configured seeds. Scaling is fit on Hanoi source training windows only. Epoch and probability-threshold
selection use Hanoi source validation only; Hai Phong 2025 labels are loaded
after all checkpoints and thresholds are fixed. Target adaptation labels are
never used. Training settings are in `configs/rain_hanoi_haiphong.yaml`.

Per-run JSON, `baseline_summary.csv`, and a four-panel confusion-matrix SVG are
written to `artifacts/metrics/`. CNN and logistic-regression checkpoints are
written under `artifacts/checkpoints/`; checkpoints are ignored by Git. Results and metric
interpretation are recorded in `docs/phase2-report.md`.

Target final-test results (2025):

| Model | Macro-F1 | PR-AUC | Rain recall | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| B0 Persistence | 0.8801 | 0.7472 | 0.8343 | 0.8801 |
| B1 Logistic regression | 0.8572 | 0.8805 | 0.7887 | 0.8541 |
| B2 CNN source-only | 0.7839 ± 0.0095 | 0.8004 ± 0.0083 | 0.7617 ± 0.0927 | 0.7962 ± 0.0185 |
| B3 CNN + SLARDA | 0.7889 ± 0.0050 | 0.8005 ± 0.0091 | 0.7519 ± 0.0416 | 0.7976 ± 0.0119 |

B2 and B3 show mean ± standard deviation across seeds 17, 23, and 42. B0 and
B1 are deterministic single runs. SLARDA makes a small Macro-F1 gain over the
source-only CNN but does not beat persistence or logistic regression; see the
report for limitations and the confusion matrices.

## Citation
If you found this work useful for you, please consider citing it.
```
@article{amda_tim,
  author={Ragab, Mohamed and Eldele, Emadeldeen and Chen, Zhenghua and Wu, Min and Li, Haoliang and Kwoh, Chee-Keong and Li, Xiaoli},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Self-supervised Autoregressive Domain Adaptation for Time Series Data}, 
  year={2022},
  volume={},
  number={},
  pages={},
  doi={}}
```

## Contact
For any issues/questions regarding the paper or reproducing the results, please contact me.   
Mohamed Ragab    
School of Computer Science and Engineering (SCSE),   
Nanyang Technological University (NTU), Singapore.   
Email: mohamedr002{at}e.ntu.edu.sg   
