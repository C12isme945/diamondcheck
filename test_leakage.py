import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from analyze import prepare,FEATURES


class TemporalDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=pd.read_csv(Path(__file__).parent/'data'/'Teams.csv')
        cls.prepared=prepare(cls.raw)

    def test_target_outcome_cannot_change_its_features(self):
        changed=self.raw.copy()
        changed.loc[changed.yearID.eq(2025),'R']+=1000
        actual=prepare(changed)
        np.testing.assert_array_equal(actual.loc[actual.yearID.eq(2025),FEATURES],
                                      self.prepared.loc[self.prepared.yearID.eq(2025),FEATURES])
        self.assertTrue((actual.loc[actual.yearID.eq(2025),'rpg'].to_numpy()>
                         self.prepared.loc[self.prepared.yearID.eq(2025),'rpg'].to_numpy()).all())

    def test_missing_prior_year_is_not_silently_bridged(self):
        changed=self.raw.loc[~((self.raw.yearID==2024)&(self.raw.franchID=='NYY'))]
        actual=prepare(changed)
        self.assertFalse(((actual.yearID==2025)&(actual.franchID=='NYY')).any())

    def test_shuffled_source_does_not_change_pairing(self):
        actual=prepare(self.raw.sample(frac=1,random_state=7))
        pd.testing.assert_frame_equal(actual,self.prepared)

    def test_duplicate_franchise_season_is_rejected(self):
        row=self.raw[(self.raw.yearID==2025)&(self.raw.franchID=='NYY')]
        with self.assertRaises(AssertionError):
            prepare(pd.concat([self.raw,row],ignore_index=True))


if __name__=='__main__':unittest.main()
