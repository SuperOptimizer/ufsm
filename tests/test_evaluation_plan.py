"""Acceptance and partial-label scores must never fit the serving cutoff."""
import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from production import evaluation_plan, summarize_scores


def scores(values, constant=.1):
    return dict(rows=[dict(threshold=t, f1=f, band_f1=f) for t, f in zip((.3,.6), values)],
                level=0, constant_foreground_f1=constant, positive_fraction=.05)


class EvaluationSplit(unittest.TestCase):
    def setUp(self):
        self.data = {'cal-a':scores((.5,.2)), 'cal-b':scores((.1,.8)),
                     'accept':scores((1,0)), 'partial':scores((1,0),.9)}
        self.plan = dict(thresholds=[.3,.6], report_threshold=.3,
                         groups={'dense':['cal-a','cal-b','accept'], 'partial':['partial']},
                         calibration={'group':'dense','sources':['cal-a','cal-b']}, acceptance_sources=['accept'])

    def test_only_calibration_sources_fit_one_global_cutoff(self):
        result = summarize_scores(self.data, self.plan)
        self.assertEqual(result['threshold'], .6)
        self.assertEqual(result['acceptance']['mean_f1'], 0)
        self.assertAlmostEqual(result['groups']['dense']['mean_f1'], 1/3)
        self.assertEqual(result['groups']['partial']['constant_foreground_mean_f1'], .9)
        changed = copy.deepcopy(self.data)
        changed['accept'] = scores((0,1)); changed['partial'] = scores((0,1),.9)
        other = summarize_scores(changed, self.plan)
        self.assertEqual(other['calibration'], result['calibration'])
        self.assertEqual(other['acceptance']['mean_f1'], 1)

    def test_ties_choose_lower_cutoff_independent_of_grid_order(self):
        data = {n:scores((.4,.4)) for n in self.data}
        self.plan['thresholds'] = [.6,.3]
        self.assertEqual(summarize_scores(data,self.plan)['threshold'], .3)

    def test_fixed_cutoff_legacy_recipe_is_preserved(self):
        result = summarize_scores(self.data, dict(thresholds=[.3,.6],report_threshold=.6))
        self.assertEqual(result['threshold'], .6)
        self.assertNotIn('calibration',result)
        self.assertEqual(result['groups'],{})

    def test_overlapping_incomplete_and_unknown_partitions_are_rejected(self):
        for change in [dict(acceptance_sources=['cal-a']),dict(acceptance_sources=[]),
                       dict(groups={'dense':['cal-a','cal-b','accept','partial'],'other':['partial']}),
                       dict(groups={'dense':['cal-a','cal-b','accept']}),
                       dict(calibration={'group':'unknown','sources':['cal-a','cal-b']}),
                       dict(source_levels={'missing':0}),dict(source_levels={'cal-a':True})]:
            plan = dict(self.plan,**change)
            with self.assertRaises(ValueError): evaluation_plan(plan,set(self.data))

    def test_incomplete_and_nonfinite_scores_cannot_be_exported_as_summary(self):
        for rows in [[dict(threshold=.3,f1=.5,band_f1=.6)],
                     [dict(threshold=.3,f1=math.nan,band_f1=.6),dict(threshold=.6,f1=.5,band_f1=.6)]]:
            data = copy.deepcopy(self.data);data['accept']['rows']=rows
            with self.assertRaises(ValueError): summarize_scores(data,self.plan)


if __name__ == '__main__': unittest.main()
