"""Independent scalar reconstruction from raw CSV, without analyze.py."""
from pathlib import Path
import csv, hashlib, json, math
from collections import defaultdict

here=Path(__file__).parent
with (here/'data'/'Teams.csv').open(encoding='utf-8-sig',newline='') as f:
    rows=list(csv.DictReader(f))
source={(int(r['yearID']),r['franchID']):r for r in rows
        if r['lgID'] in ('AL','NL') and 1979<=int(r['yearID'])<=2025}
league=defaultdict(lambda:[0.,0.])
for (year,franchise),r in source.items():
    league[year][0]+=float(r['R']); league[year][1]+=float(r['G'])
params=json.loads((here/'results'/'model_parameters.json').read_text())
metrics=json.loads((here/'results'/'metrics.json').read_text())
assert hashlib.sha256((here/'data'/'Teams.csv').read_bytes()).hexdigest()==metrics['data_sha256']

def reconstruct(record):
    year,franchise=int(record['yearID']),record['franchID']
    prior=source[(year-1,franchise)]; current=source[(year,franchise)]
    assert int(record.get('feature_year',year-1))==year-1
    assert int(current['G'])>=100 and int(prior['G'])>=100
    n=lambda key:float(prior[key])
    x=[n('R')/n('G'),(n('H')+n('BB')+n('HBP'))/(n('AB')+n('BB')+n('HBP')+n('SF')),
       (n('H')+n('2B')+2*n('3B')+3*n('HR'))/n('AB'),
       n('BB')/n('AB'),n('SO')/n('AB'),league[year-1][0]/league[year-1][1]]
    for name,value in zip(params['feature_names'],x):
        assert math.isclose(float(record[name]),value,abs_tol=1e-12)
    prediction=params['intercept']+sum(c*(v-m)/s for c,v,m,s in zip(params['coefficients'],x,params['mean'],params['scale']))
    return prediction,float(current['R'])/float(current['G']),x[0]

with (here/'results'/'holdout_predictions.csv').open(newline='') as f:
    test=list(csv.DictReader(f))
assert len(test)==240 and len({(r['yearID'],r['franchID']) for r in test})==240
errors=[]; baselines=[]; covered=0; max_error=0
for r in test:
    prediction,actual,baseline=reconstruct(r)
    error=abs(prediction-float(r['prediction'])); max_error=max(error,max_error)
    assert error<1e-10
    assert math.isclose(actual,float(r['rpg']),abs_tol=1e-12)
    assert math.isclose(baseline,float(r['baseline']),abs_tol=1e-12)
    errors.append(abs(prediction-actual));baselines.append(abs(baseline-actual))
    covered+=int(float(r['lower'])<=actual<=float(r['upper']))
    assert math.isclose(prediction-float(r['lower']),metrics['calibration_radius'],abs_tol=1e-10)
with (here/'results'/'calibration_rows.csv').open(newline='') as f:
    calibration=list(csv.DictReader(f))
residuals=sorted(abs(p-a) for p,a,b in map(reconstruct,calibration))
assert math.isclose(residuals[math.ceil((len(residuals)+1)*.9)-1],metrics['calibration_radius'],abs_tol=1e-10)
assert math.isclose(sum(errors)/len(test),metrics['model_mae'],abs_tol=1e-12)
assert math.isclose(sum(baselines)/len(test),metrics['baseline_mae'],abs_tol=1e-12)
assert covered==metrics['covered_rows']==215
report=dict(status='PASS',real_holdout_predictions_reconstructed=len(test),
            calibration_rows_reconstructed=len(calibration),max_prediction_difference=max_error,
            source_hash_verified=True,all_inputs_from_previous_season=True,
            interval_radius_and_empirical_coverage_verified=True,
            platform_submission=False)
(here/'results'/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
