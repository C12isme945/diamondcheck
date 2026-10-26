"""Review a recorded historical forecast; does not train or predict future years."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--year', type=int, required=True, help='Historical holdout season')
    parser.add_argument('--team', required=True, help='Lahman franchise ID, e.g. NYY')
    args = parser.parse_args()
    path = Path(__file__).resolve().parent / 'results' / 'holdout_predictions.csv'
    with path.open(encoding='utf-8', newline='') as stream:
        matches = [r for r in csv.DictReader(stream)
                   if int(r['yearID']) == args.year and r['franchID'] == args.team.upper()]
    if not matches:
        parser.error('No recorded holdout forecast for this season/franchise. See results/holdout_predictions.csv.')
    if len(matches) != 1:
        raise ValueError('Ambiguous historical record')
    row = matches[0]
    f = lambda key: float(row[key])
    print(f"DiamondCheck | {row['name']} | target {row['yearID']}")
    print(f"Inputs from {row['feature_year']} only; frozen model trained through 2011.")
    print(f"Forecast: {f('prediction'):.4f} runs/game")
    print(f"Carry-forward baseline: {f('baseline'):.4f} runs/game")
    print(f"Nominal 90% range: [{f('lower'):.4f}, {f('upper'):.4f}]")
    print(f"Historical observed outcome: {f('rpg'):.4f} runs/game")
    print(f"Absolute error: model {f('model_error'):.4f}; baseline {f('baseline_error'):.4f}")
    print('Retrospective review, not a live future forecast or coverage guarantee.')


if __name__ == '__main__':
    main()
