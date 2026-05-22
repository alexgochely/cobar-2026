# Submission — COBAR Mini-project

Reactive controller for `MiniprojectSimulation` level 3. The fly walks toward the banana using olfaction while reacting to three hazards detected through other senses.

## Files

- `controller.py` — the `Controller` class.
- `run_controller.ipynb` — runner + live debug (raw vision, masks, trajectory, danger curves).

## Architecture

Three senses → three `[left, right]` drives, fused with fixed priority:

```
dragonfly (escape/freeze)  >  grass  >  olfaction (+ wind blend)
```

| Sense | Function | Behavior |
|---|---|---|
| Olfaction | `_olfactory_drive` | Baseline: bias toward the attractive odor, away from the aversive one. |
| Vision — grass | `_detect_grass` + `_visual_analysis` | Green dominant in the lower image → turn toward the clear side. |
| Vision — dragonfly | `_detect_dragonfly` + `_visual_analysis` | Red in the sky → full-throttle dash, then freeze. |
| Wind | `_wind_drive` | Antenna `qfrc_passive` → bias against the gust. |

## Dragonfly strategy

The dragonfly locks its trajectory at a fixed distance and stops correcting. Once red is confirmed for a few frames:

1. **Dash** straight ahead at full speed to build a lateral offset.
2. **Freeze** afterwards — stationary target + offset = strike misses.

## Key parameters

```python
CONTROL_DT = 0.02              # decision cadence (s)
grass_threshold = 0.035        # grass fraction in lower ROI
dragonfly_threshold = 5e-6     # a hint of red is enough
dragonfly_confirm_steps = 25   # frames before dashing
escape_speed = 4.0             # dash speed
stop_distance = 2.0            # mm to the banana
```

## Usage

```python
from flygym.compose import ActuatorType
from miniproject.simulation import MiniprojectSimulation
from submission.controller import Controller

sim = MiniprojectSimulation(level=3, seed=777)
controller = Controller(sim)

for _ in range(100000):
    joint_angles, adhesion = controller.step(sim)
    sim.set_actuator_inputs(sim.fly.name, ActuatorType.POSITION, joint_angles)
    sim.set_actuator_inputs(sim.fly.name, ActuatorType.ADHESION, adhesion)
    sim.step()
    sim.render_as_needed()
```
