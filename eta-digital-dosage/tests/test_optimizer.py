from eta_digital.optimization import (
    DosageBounds,
    ObjectiveWeights,
    QualityConstraints,
    ScenarioDosageOptimizer,
)
from eta_digital.scenarios import WeightedScenarioGenerator
from test_mixture import build_model


def test_optimizer_uses_joint_three_filter_constraint():
    model, frame = build_model()
    generator = WeightedScenarioGenerator(model, number_of_scenarios=60, random_seed=3)
    constraints = QualityConstraints(
        turbidity_outputs=(
            "filter_1_turbidity_ntu",
            "filter_2_turbidity_ntu",
            "filter_3_turbidity_ntu",
        ),
        ph_output="filtered_ph",
        maximum_turbidity_ntu=1.5,
        minimum_ph=5.8,
        maximum_ph=9.5,
        minimum_probability=0.90,
    )
    optimizer = ScenarioDosageOptimizer(
        generator,
        constraints,
        ObjectiveWeights(),
        DosageBounds(4, 22, 0.3, 4.5, pac_points=7, polymer_points=6),
    )
    state = frame.iloc[[210]][model.features]
    result = optimizer.solve(state, previous_dosage=(10.0, 2.0))
    assert 4 <= result.pac_mg_l <= 22
    assert 0.3 <= result.polymer_mg_l <= 4.5
    assert result.scenarios_evaluated == 42
    assert 0 <= result.compliance_probability <= 1
