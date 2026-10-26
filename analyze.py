"""Historical MLB run-scoring forecasts with a frozen chronological audit.

Code: MIT. Source data and derived tables: SABR / Sean Lahman, CC BY-SA 3.0.
No credentials, paid APIs, current-season outcome features or fabricated data.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES=['prior_rpg','prior_obp','prior_slg','prior_bb_per_ab',
          'prior_so_per_ab','prior_league_rpg']


def prepare(raw):
    df=raw.loc[(raw.lgID.isin(['AL','NL'])) & raw.yearID.between(1979,2025)].copy()
    assert not df.duplicated(['yearID','franchID']).any()
    df['rpg']=df.R/df.G
    df['obp']=(df.H+df.BB+df.HBP)/(df.AB+df.BB+df.HBP+df.SF)
    df['slg']=(df.H+df['2B']+2*df['3B']+3*df.HR)/df.AB
    df['bb_per_ab']=df.BB/df.AB
    df['so_per_ab']=df.SO/df.AB
    league=df.groupby('yearID').agg(runs=('R','sum'), games=('G','sum'))
    df['league_rpg']=df.yearID.map(league.runs/league.games)
    prior=df[['yearID','franchID','G','rpg','obp','slg','bb_per_ab','so_per_ab','league_rpg']].copy()
    prior['feature_year']=prior.yearID
    prior['yearID']+=1
    prior=prior.rename(columns={c:'prior_'+c for c in prior.columns if c not in ['yearID','franchID','feature_year']})
    rows=df.merge(prior,on=['yearID','franchID'],validate='one_to_one')
    rows=rows.loc[(rows.G>=100)&(rows.prior_G>=100)&rows.yearID.ge(1980)].copy()
    assert (rows.feature_year==rows.yearID-1).all()
    assert np.isfinite(rows[FEATURES+['rpg']].to_numpy()).all()
    return rows.sort_values(['yearID','franchID']).reset_index(drop=True)


def evaluate(raw):
    rows=prepare(raw)
    train=rows[rows.yearID.le(2011)].copy()
    calibration=rows[rows.yearID.between(2012,2015)].copy()
    test=rows[rows.yearID.ge(2016)].copy()
    assert train.yearID.max()<calibration.yearID.min()<test.yearID.min()
    model=make_pipeline(StandardScaler(),Ridge(alpha=10.0))
    model.fit(train[FEATURES],train.rpg)
    calibration_errors=np.abs(calibration.rpg-model.predict(calibration[FEATURES]))
    rank=min(len(calibration),math.ceil((len(calibration)+1)*0.9))
    radius=float(np.sort(calibration_errors)[rank-1])
    test['prediction']=model.predict(test[FEATURES])
    test['baseline']=test.prior_rpg
    test['lower']=test.prediction-radius
    test['upper']=test.prediction+radius
    test['model_error']=np.abs(test.prediction-test.rpg)
    test['baseline_error']=np.abs(test.baseline-test.rpg)
    test['covered']=test.rpg.between(test.lower,test.upper)
    # Resample entire target seasons, preserving within-year team dependence.
    by_year=test.groupby('yearID').apply(lambda g: float((g.baseline_error-g.model_error).mean()),include_groups=False).to_numpy()
    rng=np.random.default_rng(20261026)
    boot=by_year[rng.integers(0,len(by_year),size=(2000,len(by_year)))].mean(axis=1)
    metrics=dict(train_rows=len(train),calibration_rows=len(calibration),test_rows=len(test),
        train_years=[int(train.yearID.min()),int(train.yearID.max())],
        calibration_years=[2012,2015],test_years=sorted(map(int,test.yearID.unique())),
        features=FEATURES,model='Training-only StandardScaler + Ridge(alpha=10)',
        model_mae=float(test.model_error.mean()),baseline_mae=float(test.baseline_error.mean()),
        baseline='Previous season runs/game for same franchise',
        mae_improvement=float((test.baseline_error-test.model_error).mean()),
        improvement_95pct_season_bootstrap=list(map(float,np.quantile(boot,[.025,.975]))),
        nominal_interval_coverage=.9,calibration_radius=radius,
        observed_test_coverage=float(test.covered.mean()),
        covered_rows=int(test.covered.sum()),excluded_years=[2020,2021],
        exclusion='Both target and preceding season must have at least 100 games; expansion teams without an immediately preceding season also omitted.',
        caveat='One frozen historical split; no causal, betting, player-level, future-season or live-deployment claim. Nominal interval coverage assumes exchangeability, which era changes and correlated teams can violate.',
        tuning='Features, splits, alpha=10 and seed fixed before first holdout evaluation; no parameter search.',
        created_date='2026-10-26',ai_assistance='OpenAI Codex generated implementation, validation, documentation and presentation; no paid model calls.')
    columns=['yearID','feature_year','franchID','name','G','rpg','prediction','baseline','lower','upper',
             'model_error','baseline_error','covered']+FEATURES
    return metrics,test[columns],model,train,calibration


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path(__file__).parent/'data'/'Teams.csv')
    args=parser.parse_args()
    here=Path(__file__).parent
    metrics,test,model,train,calibration=evaluate(pd.read_csv(args.data,encoding='utf-8-sig'))
    metrics['data_sha256']=hashlib.sha256(args.data.read_bytes()).hexdigest()
    (here/'results').mkdir(exist_ok=True)
    (here/'results'/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    test.to_csv(here/'results'/'holdout_predictions.csv',index=False)
    for part,label in [(train,'training'),(calibration,'calibration')]:
        part[['yearID','franchID','rpg']+FEATURES].to_csv(here/'results'/f'{label}_rows.csv',index=False)
    parameters=dict(feature_names=FEATURES,mean=model[0].mean_.tolist(),scale=model[0].scale_.tolist(),
                    coefficients=model[1].coef_.tolist(),intercept=float(model[1].intercept_))
    (here/'results'/'model_parameters.json').write_text(json.dumps(parameters,indent=2),encoding='utf-8')
    payload=json.dumps(dict(metrics=metrics,rows=json.loads(test.to_json(orient='records'))),ensure_ascii=True)
    template=(here/'report_template.html').read_text(encoding='utf-8')
    assert template.count('__PAYLOAD__')==1
    (here/'DiamondCheck.html').write_text(template.replace('__PAYLOAD__',payload),encoding='utf-8')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    main()
