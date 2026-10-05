import copy
import json
from pathlib import Path
import unittest
from independent_task_recompute import recompute_records,evaluate_frozen_gates,TaskEvidenceError

class RecomputeTests(unittest.TestCase):
    def fixture(self):
        matrix=json.loads((Path(__file__).parent/'experiment-matrix.json').read_text(encoding='utf-8'))
        episodes=[];costs=[]
        for p in matrix['splits']['task_confirmation']:
            for repeat in range(3):
                for method in ('G0','T','G1','H'):
                    for seed in ([None] if method=='H' else matrix['policy_seeds']):
                        utility={'G0':.1,'T':.12,'G1':.2,'H':.14}[method]
                        row={'method':method,'seed':seed,'parent':p['parent'],'repeat':repeat,
                             'scenario_sha256':p['scenario_sha256'],'structural_sha256':p['structural_sha256'],
                             'exogenous_key':p['exogenous_key'].rsplit('|repeat-',1)[0]+f'|repeat-{repeat}',
                             'preference':[.8,.2],'steps':1,'utility_valid':True,'outcome_valid':True,
                             'utility':utility,'terminated':True,'truncated':False,'opportunity':True,
                             'actions':[{'vector_reward':[utility/.4,0],'selected_action':0}]}
                        episodes.append(row)
                        costs.append({k:row[k] for k in ('method','seed','parent','repeat')}|
                                     {'decision_step':0,'candidate_count':2,'cpu_seconds':.005,'wall_seconds':.008,
                                      'world_model_forwards':int(method=='G1')})
        return matrix,episodes,costs
    def test_full_reward_recomputation_and_frozen_gates(self):
        matrix,episodes,costs=self.fixture();metrics=recompute_records(episodes,costs,matrix)
        self.assertEqual(metrics['episode_count'],240)
        self.assertEqual(metrics['complete_parent_count'],8)
        self.assertTrue(evaluate_frozen_gates(matrix,metrics)['research_success'])
        for m,n in [('G0',72),('T',72),('G1',72),('H',24)]:self.assertEqual(metrics['decision_costs']['by_method'][m]['sample_count'],n)
    def test_fast_g0_does_not_hide_slow_g1(self):
        matrix,episodes,costs=self.fixture()
        for row in costs:row['cpu_seconds']=.020 if row['method']=='G1' else .0001
        verdict=evaluate_frozen_gates(matrix,recompute_records(episodes,costs,matrix))
        self.assertTrue(verdict['task_gate']['pass']);self.assertFalse(verdict['cost_gate']['pass'])
    def test_slow_g0_not_attributed_to_fast_g1(self):
        matrix,episodes,costs=self.fixture()
        for row in costs:
            if row['method']=='G0':row.update(cpu_seconds=.100,wall_seconds=.200)
        metrics=recompute_records(episodes,costs,matrix)
        self.assertTrue(evaluate_frozen_gates(matrix,metrics)['cost_gate']['pass'])
        self.assertEqual(len(metrics['decision_costs']['by_method_parent_repeat']),96)
    def test_no_opportunity_cost_unevaluated_utility_known(self):
        matrix,episodes,costs=self.fixture()
        for row in episodes:row['opportunity']=False;row['actions'][0]['selected_action']=24
        for row in costs:row.update(candidate_count=1,world_model_forwards=0)
        metrics=recompute_records(episodes,costs,matrix)
        self.assertEqual(metrics['unknown_episode_count'],0)
        self.assertEqual(evaluate_frozen_gates(matrix,metrics)['cost_gate']['status'],'not_evaluated')
    def test_unknown_not_filled_zero(self):
        matrix,episodes,costs=self.fixture()
        episodes[0].update(utility_valid=False,outcome_valid=False,utility=None)
        metrics=recompute_records(episodes,costs,matrix)
        self.assertEqual(metrics['unknown_episode_count'],1)
        self.assertFalse(evaluate_frozen_gates(matrix,metrics)['task_gate']['pass'])
    def test_known_reward_utility_survives_unknown_lifecycle(self):
        matrix,episodes,costs=self.fixture()
        episodes[0]['outcome_valid']=False
        metrics=recompute_records(episodes,costs,matrix)
        self.assertEqual(metrics['unknown_episode_count'],0)
        self.assertEqual(metrics['lifecycle_unevaluated_episode_count'],1)
        self.assertTrue(evaluate_frozen_gates(matrix,metrics)['task_gate']['pass'])
    def test_reward_identity_cost_omission_and_nan_rejected(self):
        matrix,episodes,costs=self.fixture()
        bad=copy.deepcopy(episodes);bad[0]['utility']+=.01
        with self.assertRaisesRegex(TaskEvidenceError,'RAW_REWARD_UTILITY_MISMATCH'):recompute_records(bad,costs,matrix)
        with self.assertRaisesRegex(TaskEvidenceError,'DECISION_COST_RECORDS_INCOMPLETE'):recompute_records(episodes,costs[1:],matrix)
        costs[0]['wall_seconds']=float('nan')
        with self.assertRaises(TaskEvidenceError):recompute_records(episodes,costs,matrix)
if __name__=='__main__':unittest.main()
