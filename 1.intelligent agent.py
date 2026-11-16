import random
from dataclasses import dataclass, field


@dataclass
class Environment:
    temp: float = field(default_factory=lambda: random.uniform(18, 32))
    light: float = field(default_factory=lambda: random.uniform(0, 1000))
    motion: bool = field(default_factory=lambda: random.random() < 0.5)
    hour: int = field(default_factory=lambda: random.randint(0, 23))

    def step(self):
        self.temp += random.uniform(-1.5, 1.5)
        self.light = max(0, min(1000, self.light + random.uniform(-100, 100)))
        self.motion = random.random() < 0.5
        self.hour = (self.hour + 1) % 24


class SmartHomeAgent:
    def __init__(self):
        self.state = {"AC": False, "Heater": False, "Lights": False, "Door locked": True}

    def act(self, env):
        night = env.hour < 6 or env.hour >= 20
        want = {
            "AC": env.temp > 26,
            "Heater": env.temp < 20,
            "Lights": env.motion and (env.light < 300 or night),
        }
        if night and not env.motion:
            want["Door locked"] = True

        actions = []
        for device, on in want.items():
            if self.state[device] != on:
                self.state[device] = on
                actions.append(f"{device} -> {'ON' if on else 'OFF'}")
        return actions or ["No action needed"]


def run_simulation(steps=8):
    env, agent = Environment(), SmartHomeAgent()
    print("=" * 60, "\nSMART HOME CONTROLLER SIMULATION\n" + "=" * 60)
    for t in range(1, steps + 1):
        actions = agent.act(env)
        print(f"\n--- Step {t} ({env.hour:02d}:00) ---")
        print(f"Sensors: {env.temp:.1f}C | {env.light:.0f} lux | Motion: {env.motion}")
        print("Actions:", "; ".join(actions))
        print("State:  ", " | ".join(f"{k}: {'ON' if v else 'OFF'}" for k, v in agent.state.items()))
        env.step()


if __name__ == "__main__":
    run_simulation()
