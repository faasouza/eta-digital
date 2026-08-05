from eta_digital.optimization.optimizer import OptimizationResult
from eta_digital.supervision.fallback import FallbackPolicy
from eta_digital.supervision.fuzzy_rules import FuzzySupervisor,SupervisionConfig

def test_supervisor_accepts_and_falls_back():
    s=FuzzySupervisor(SupervisionConfig(),FallbackPolicy(10.95,3.7)); r=OptimizationResult(12,4,1,.97,True,10); d=s.evaluate(r,.95,1,0,(11,3.5)); assert d.status=="accepted"
    d=s.evaluate(OptimizationResult(12,4,1,.4,False,10),.2,.2,1,(11,3.5)); assert d.status=="fallback" and d.pac_mg_l==10.95
