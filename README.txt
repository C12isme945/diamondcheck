DiamondCheck — Does a forecast beat carrying last season forward?
Created October 26, 2026. Prototype for student sports analytics submissions.
Status: local working analysis and interactive report; not yet submitted.

OPEN THE RESULT
Open DiamondCheck.html in a modern browser. It is a standalone interactive
research report: no installation, server, login, API key or network required.
Choose a historical season and team, compare forecast, outcome and baseline,
inspect the calibration range and largest misses, and export the predictions.
Browser interaction checks were not performed: the browser policy rejected
opening a local file URL. JavaScript syntax was checked; the Python analysis,
four leakage checks and independent scalar audit passed. Treat the optional
HTML interface as not browser-verified. The command-line prototype is verified.

REPRODUCE
Python 3.12+ with numpy, pandas and scikit-learn:
  pip install -r requirements.txt
  python analyze.py
  python audit_results.py
  python -m unittest test_leakage.py
  python review.py --year 2025 --team NYY
The committed data snapshot lets these commands work without network access.
The HTML is regenerated from report_template.html and the actual results.
review.py displays one recorded historical holdout forecast, its baseline,
calibration range and observed outcome; it does not forecast a future season.

METHOD
Predict a team's next-season runs per game from the immediately preceding
season's runs/game, OBP, SLG, walks/AB, strikeouts/AB and league runs/game.
Franchise IDs preserve continuity across relocations and team-name changes.
Both seasons need at least 100 games. An absent prior season is never filled
with a different year's record. 2020 and 2021 are excluded by that rule.

Train: target seasons 1980–2011, 894 rows.
Calibration: 2012–2015, 120 rows. Test: 2016–2025, eight eligible years, 240 rows.
Scaler and Ridge(alpha=10) fit on training only. Features, splits, alpha and
random seed were fixed before first holdout evaluation, with no parameter
search. The model stays frozen throughout calibration and test.
The 90% absolute-error interval uses the finite-sample calibration quantile.
The bootstrap resamples whole test years, keeping teams within a year together.

MEASURED RESULTS
Historical test MAE: 0.33830400818981016 runs/game.
Carry-forward baseline MAE: 0.40358919050959147 runs/game.
Mean improvement: 0.06528518231978137 runs/game.
95% year-bootstrap interval for improvement: [0.03705725116302175, 0.09110020841463867].
Nominal 90% range covered 215/240 outcomes (89.5833%).
See results/metrics.json, all split rows, predictions and scalar model parameters.

LIMITS
One retrospective chronological split, only eight test seasons, revised
historical data, and no live prospective deployment. Calibration exchangeability
can fail under era change and team dependence. Not a player valuation, causal
claim, gambling strategy or 2026 forecast. Team-level data omits roster changes,
injuries and rule changes. A useful review tool does not guarantee decisions.

PROVENANCE & LICENSE
Original database: SABR / Sean Lahman, https://sabr.org/lahman-database/ .
CSV transport: attributed Mycelium mirror. See data/provenance.json for the
pinned mirror commit, exact URL, byte count and SHA-256.
The official SABR README was fetched separately and is preserved unchanged.
CSV byte identity against Box was not verified. No StatsBomb data is included.
Source data and derived analysis/report: CC BY-SA 3.0. Software: MIT.
Preserve data/DATA_LICENSE.txt and data/readme2025.txt when sharing the data.

AI ASSISTANCE
OpenAI Codex generated and revised code, analysis, checks, report and prose for
Yansong Li. All baseball values come from the cited source; none are generated.
No paid API or external LLM inference. No statement of the entrant's personal
workshop attendance, development struggles or unobserved learning is fabricated.
