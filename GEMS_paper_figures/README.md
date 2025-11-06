# GEMS_paper_figures

## [cell culture experiment data](data/241106_CCDS_253g1-hek293a_report)

## [colour water optimisation data](data/gems_ot2_colour-water-optimisation)

# Scripts to generate figures for GEMS paper

## [Common setup]

Create and activate a virtual environment, then install the required packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt # If you have issues, try GEMS_paper_figures/pip_install.txt
```

## [Cell culture figure scripts]

### Preprocessing

```bash
python script/cell/analyse.py
```

#### Logistic curve fitting (because rounding error may cause slight differences in results, we provide the output data used in the paper in the GEMS_paper_figures/data/241106_CCDS_253g1-hek293a_report/growth_curve folder)

To re-run the fitting, use the following script and amend the input path within the script for the “Figure Generation” step as appropriate.

```bash
python script/cell/logistic_fit.py
```

To confirm that your results match those used in the paper, you can use the following script to compare the output files.

```bash
utils/diff_check.sh
```

### Figure generation

Figures 5D:

```bash
python script/cell/manual_schedule.py
```

Figures 5E:

```bash
python script/cell/curve_estimatitor.py
```

## [Colour water optimisation figure scripts]

### Preprocessing

```bash
python script/CW/scoring.py
```bash```
python script/CW/df_process.py
```

### Figure generation

Figures 4F:

```bash
python script/CW/plot_with_ratio.py
```
