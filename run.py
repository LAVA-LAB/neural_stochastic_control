
import gymnasium as gym
from models.pendulum import PendulumEnv

env = PendulumEnv(render_mode = "human")

observation, _ = env.reset()
x, y, angular_speed = observation
print(f'location {x,y}, speed {angular_speed}')

env.render()

for step in range(100):
    observation, reward, done, _, _ = env.step(env.action_space.sample())
    x, y, angular_speed = observation
    print(f'location {x, y}, speed {angular_speed}')