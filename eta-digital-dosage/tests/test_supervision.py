from eta_digital.optimization import OptimizationResult
from eta_digital.supervision import FallbackDosage, FuzzySupervisor


def test_supervisor_falls_back_for_infeasible_result():
    supervisor = FuzzySupervisor(0.75, 0.5, 2.0, 0.75, FallbackDosage(10.95, 3.7))
    optimum = OptimizationResult(15, 3, 1, 0.6, False, 10)
    result = supervisor.evaluate(optimum, 0.9, 1.0, (10, 2))
    assert result.status == "fallback"
    assert result.pac_mg_l == 10.95
