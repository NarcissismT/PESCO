import unittest
from scripts.write_tier1_p233_gate import build_gate
import numpy as np
from research_strategy_optimization.environments.tier1_tabular_env import Tier1TabularEnvironment

class P233GateTests(unittest.TestCase):
    def _inputs(self):
        methods=['GRPO-Atomic','Atomic+State','Atomic+State+Flip','PESCO-Full']
        matrix={'methods':['SFT',*methods],'seeds':[1,2], 'method_matrix_contract':{'sft_shared_checkpoint':True},'training_logs':{str(s):{m:{'atomic_reward_shared':True,'budget_contract':{'policy_rollout_calls':1,'counterfactual_branch_calls':1,'exploration_seed_executions':1,'confirmation_seed_executions':1,'optimizer_steps':1,'forward_backward_flops':1}} for m in methods} for s in [1,2]}}
        rec=[]
        for s in [1,2]:
            for m in ['Atomic+State','Atomic+State+Flip','PESCO-Full']:
                rec.append({'method':m,'seed':s,'split':'promotion','state_macro_f1':.5,'state_recall':{'invalid':.2},'invalid_recall':.2,'confirmation_rate':.9,'erroneous_repair_rate':0.,'question_metric_rows':[{'question_id':'q','family':'f','normalized_regret':.1,'pairwise_reversal_ranking_accuracy':.8}]})
        matrix['records']=rec
        agg={'receipt_derived':True,'comparisons':{n:{'regret_delta_left_minus_right':{'upper':-.1},'pairrank_delta_left_minus_right':{'lower':.1}} for n in ['AB-A','ASB-AS','ABF-AF','Full-ASF','flip_vs_atomic','full_vs_atomic','full_vs_best_nonfull']},'methods':{'PESCO-Full':{'mean_normalized_regret':.1},'GRPO-Atomic':{'mean_normalized_regret':.3},'Atomic+State':{'mean_normalized_regret':.2}},'direction_summary':{'full_vs_atomic':{'positive_seed_fraction':1.}},'family_leave_one_out':{'f':{'PESCO-Full':.1,'Atomic+State+Flip':.2}}}
        conv={'gates':{'all_steps_executed':True,'all_seeds_executed':True},'records':[{'method':m,'steps':1024,'plateau_gate':True,'tail_entropy':.5,'tail_kl':.1} for m in ['GRPO-Atomic','Atomic+Branch','PESCO-Full']]}; shortcut={'models':{'without_confirmation:logistic_regression':{'metrics_by_split':{'promotion':{'normalized_regret':.3}}}}}; est={'no_hidden_truth_used_by_estimator':True,'forced_hidden_truth_invariance':True}; stab={'pass':True}; return matrix,agg,conv,shortcut,est,stab

    def test_positive_fixture_is_go(self):
        out=build_gate(*self._inputs())
        self.assertEqual(out['status'],'GO')
        self.assertTrue(all(out['p233_go'].values()))
        self.assertTrue(str(out.get('audit_sha256','')).startswith('sha256:'))
    def test_negative_regret_gate(self):
        x=self._inputs(); x[1]['comparisons']['Full-ASF']['regret_delta_left_minus_right']['upper']=.01
        out=build_gate(*x); self.assertFalse(out['p233_go']['full_vs_atomic_state_flip_regret_ci_upper_lt_zero']); self.assertEqual(out['status'],'NO_GO')
    def test_negative_budget_gate(self):
        x=self._inputs(); x[0]['training_logs']['1']['PESCO-Full']['budget_contract']['optimizer_steps']=2
        out=build_gate(*x); self.assertFalse(out['p233_go']['environment_execution_budget_matched'])

    def test_ood_estimator_fixed_observed_arrays(self):
        rng=np.random.default_rng(4); t=rng.binomial(1,.5,64).astype(float); c=rng.normal(size=64); y=.2*t+.1*c+rng.normal(size=64)
        for family in (Tier1TabularEnvironment.FAMILY_HETEROGENEOUS_NOISE, Tier1TabularEnvironment.FAMILY_NONLINEAR_RESPONSE):
            a=Tier1TabularEnvironment.estimate_ood_repair(family,t,c,y); b=Tier1TabularEnvironment.estimate_ood_repair(family,t,c,y); self.assertEqual(a,b)

    def test_every_gate_fails_closed_on_empty_receipts(self):
        out=build_gate({}, {}, {}, {}, {}, {})
        self.assertEqual(out['status'],'NO_GO')
        for name, value in out['p233_go'].items():
            with self.subTest(gate=name):
                self.assertFalse(value)

if __name__=='__main__': unittest.main()
