import json
from cinderforge_e.dsl.spec import GraphSpec, LevelSpec
from cinderforge_e.validate.conformance import run_conformance

def main():
    spec = GraphSpec(id="quick", seed=7,
                     levels=[LevelSpec(units=2, neurons_per=16), LevelSpec(units=3, neurons_per=64), LevelSpec(units=2, neurons_per=16)],
                     max_vertical_skip=2, max_lateral_right=2, lateral_gate_after=1,
                     resources={"lateral_mode": "ws", "ws_k": 2, "ws_beta": 0.2})
    print(json.dumps(run_conformance(spec), indent=2))

if __name__ == '__main__':
    main()