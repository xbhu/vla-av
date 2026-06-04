# Smart Mobility Lab — GitHub Repo Compliance Checklist

**Purpose:** Ensure all lab repos meet minimum standards for reproducibility, citability, and public visibility.  
**Applies to:** All research repos associated with Smart Mobility Lab publications or datasets.  
**Maintained by:** Xianbiao Hu (xbhu@psu.edu)

---

## How to Use This Checklist

Students complete self-assessment before each milestone (see §4).  
XB reviews during advising meetings at key milestones.  
A repo must pass **all Tier 1 items** before the paper is submitted.  
All items must pass before the repo goes public.

---

## Tier 1 — Required at Paper Submission

These are non-negotiable. A repo missing any of these is not ready for submission.

### README.md

- [ ] **Title matches the paper title exactly**
- [ ] **Paper DOI is linked** (even if "under review," include the preprint or placeholder)
- [ ] **BibTeX citation block is included** (copy-paste ready)
- [ ] **Authors listed** — student first, XB as corresponding/PI
- [ ] **One-paragraph plain-language description** (what problem, what the code does, who it's for)
- [ ] **Environment setup instructions** are present and accurate (Python version, `pip install` or `conda` command)
- [ ] **At least one runnable example** — not just "see the paper," but an actual command that produces output
- [ ] **Data section** — explains where data lives and how to get it (Zenodo link, or "available upon request" with contact)

### Code

- [ ] **Code runs end-to-end** — at least one other person (labmate or XB) has verified this
- [ ] **No hardcoded absolute paths** — all paths are relative or set via config
- [ ] **No API keys, passwords, or credentials** in any file
- [ ] **`requirements.txt` or `environment.yml` is present** and matches what the code actually needs

### Data

- [ ] **Large data files are NOT in the repo** (>10MB files should be external)
- [ ] **`data/README.md` exists** and describes the data format, columns, and units
- [ ] **Data source is clearly credited** (original collection method, external source, or own collection)

### Licensing

- [ ] **`LICENSE` file exists** (default: MIT for code; CC BY 4.0 for data)

---

## Tier 2 — Required Before Repo Goes Public (at or after paper acceptance)

### Citeability

- [ ] **Zenodo DOI is created** for the dataset (upload to [zenodo.org](https://zenodo.org))
- [ ] **GitHub repo is linked in the paper** ("Code available at: github.com/...")
- [ ] **Zenodo dataset is linked in the paper** ("Data available at: doi.org/...")
- [ ] **README Zenodo link is updated** with the actual DOI (not placeholder)

### Reproducibility

- [ ] **Key results from the paper can be reproduced** with the provided code and data
- [ ] **Config files are used** — hyperparameters are not buried in the middle of training scripts
- [ ] **Random seeds are set** where applicable, for reproducibility

### Visibility

- [ ] **GitHub repo is linked from Atlas paper page** (atlas.mobilitypsu.com)
- [ ] **Repo description field is filled in** on GitHub (one sentence + topic tags)
- [ ] **GitHub Topics are set** — e.g., `transportation`, `autonomous-vehicles`, `ev-charging`, `lidar`, `trajectory`

---

## Tier 3 — Best Practice (Encouraged, Not Enforced)

- [ ] `notebooks/demo.ipynb` — a working demo notebook with visualizations
- [ ] `results/figures/` — key paper figures saved as PNG or PDF
- [ ] GitHub Actions CI — at least a basic test that the code imports without errors
- [ ] Slides or video link in README (if available)
- [ ] Chinese README or Atlas page linked (if applicable)
- [ ] Dataset also mirrored on HuggingFace (if structured tabular data)

---

## Milestone Schedule

| Milestone | Tier 1 Complete? | Tier 2 Complete? |
|-----------|-----------------|-----------------|
| Paper draft sent to XB for review | ✅ Required | — |
| Paper submitted to journal/conference | ✅ Required | — |
| Paper accepted | ✅ Required | ✅ Required |
| Repo made public | ✅ Required | ✅ Required |

---

## Quick Self-Assessment (Student Signs Off)

```
Repo name: ___________________________________
Paper title: _________________________________
Milestone: ___________________________________
Date: ________________________________________

Tier 1 items — all checked? [ ] Yes  [ ] No (list missing items below)
_______________________________________________

Tier 2 items — all checked? [ ] Yes  [ ] No / Not yet applicable
_______________________________________________

Verified code runs end-to-end? [ ] Yes — verified by: _______________
Data accessible via external link? [ ] Yes — link: ___________________

Student signature: ____________________________
```

---

## Repo Naming Convention

```
[topic]-[method/keyword]-[year]
```

Examples:
- `ev-charging-gnn-2024`
- `atma-workzone-safety-2023`
- `cav-cooperative-localization-2022`
- `traffic-flow-dtq-modeling-2024`

**Rules:**
- All lowercase, hyphens only (no underscores)
- Year = year of submission or publication
- Avoid generic names like `paper-code` or `research-project`

---

## Contact

Questions about this checklist: **xbhu@psu.edu**  
Lab website: [sites.psu.edu/xbhu](https://sites.psu.edu/xbhu/)  
Research Atlas: [atlas.mobilitypsu.com](https://atlas.mobilitypsu.com)
