"""TagMind: mini self-play chase environment with two DQN agents.

Example:
    python Tagmind.py --rounds 100 --device auto
    python Tagmind.py --rounds 1000 --seed 42 --checkpoint-dir /content/drive/MyDrive/tagmind
"""

import argparse
import csv
import os
import random
import time
from collections import deque, namedtuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

SIZE = 10
MAX_STEPS = 50
ACTIONS = (
    "up", "down", "left", "right",
    "jump_up", "jump_down", "jump_left", "jump_right",
    "place_teleporter",
)

EMPTY, BLOCK, CHASER, RUNNER = ".", "#", "C", "R"
BOOST, ENER, RADAR, ENERCORE, TELEPORTER = "B", "E", "T", "X", "P"

BLOCK_COUNT = 18
BOOST_COUNT = 4
ENER_COUNT = 5
RADAR_COUNT = 1
ENERCORE_COUNT = 1
ENERGY_MAX = 6.0
ENER_PER_STEP = 0.5
ENER_PER_JUMP = 2.0
VISION = 4

REWARD_CATCH = 1.0
REWARD_DISTANCE_STEP = 0.05
REWARD_POWERUP = 0.20
REWARD_STEP_LIMIT = -0.02

LEARN_RATE = 1e-3
DISCOUNT = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.08
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
BUFFER_CAP = 20_000
SYNC_TARGET_EVERY = 100
MIN_BUFFER_BEFORE_LEARN = 500
GRAD_CLIP = 5.0

Transition = namedtuple("Transition", ("s", "a", "r", "ns", "d"))


def in_bounds(position):
    x, y = position
    return 0 <= x < SIZE and 0 <= y < SIZE


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_console():
    os.system("cls" if os.name == "nt" else "clear")


class ReplayBuffer:
    def __init__(self, capacity=BUFFER_CAP):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = np.stack([item.s for item in batch])
        next_states = np.stack([item.ns for item in batch])
        actions = np.asarray([item.a for item in batch], dtype=np.int64).reshape(-1, 1)
        rewards = np.asarray([item.r for item in batch], dtype=np.float32).reshape(-1, 1)
        dones = np.asarray([item.d for item in batch], dtype=np.float32).reshape(-1, 1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, observation_dim, device):
        self.device = torch.device(device)
        self.net = DQNNet(observation_dim, len(ACTIONS)).to(self.device)
        self.target = DQNNet(observation_dim, len(ACTIONS)).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.target.eval()
        self.optimizer = optim.Adam(self.net.parameters(), lr=LEARN_RATE)
        self.learn_steps = 0

    def act(self, observation, epsilon):
        if random.random() < epsilon:
            return random.randrange(len(ACTIONS))
        observation = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.net(observation).argmax(dim=1).item())

    def learn(self, replay_buffer):
        if len(replay_buffer) < MIN_BUFFER_BEFORE_LEARN:
            return None

        states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.net(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target(next_states).max(dim=1, keepdim=True).values
            targets = rewards + (1.0 - dones) * DISCOUNT * next_q_values

        loss = nn.functional.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), GRAD_CLIP)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % SYNC_TARGET_EVERY == 0:
            self.target.load_state_dict(self.net.state_dict())
        return float(loss.item())


class Agent:
    def __init__(self, role, device):
        self.role = role
        self.policy = DQNAgent(15, device)
        self.energy = ENERGY_MAX
        self.radar_active = 0
        self.enercore_active = 0
        self.enercore_steps = 0
        self.has_teleporter = True

    def state_vector(self, self_position, other_position, grid):
        dx = np.clip(other_position[0] - self_position[0], -VISION, VISION) / VISION
        dy = np.clip(other_position[1] - self_position[1], -VISION, VISION) / VISION
        distance = manhattan(self_position, other_position) / (SIZE * 2)

        x, y = self_position
        directions = (
            (x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y),
            (x - 1, y - 1), (x + 1, y - 1),
            (x - 1, y + 1), (x + 1, y + 1),
        )
        local = []
        for px, py in directions:
            if not (0 <= px < SIZE and 0 <= py < SIZE):
                local.append(1.0)
            elif grid[py][px] == BLOCK:
                local.append(1.0)
            elif grid[py][px] in (BOOST, ENER, RADAR, ENERCORE):
                local.append(0.5)
            else:
                local.append(0.0)

        return np.asarray([
            dx, dy, distance, self.energy / ENERGY_MAX,
            float(self.radar_active > 0), float(self.enercore_active > 0),
            float(self.has_teleporter),
            *local,
        ], dtype=np.float32)


def make_map():
    grid = [[EMPTY for _ in range(SIZE)] for _ in range(SIZE)]
    for _ in range(BLOCK_COUNT):
        while True:
            x, y = random.randrange(SIZE), random.randrange(SIZE)
            if grid[y][x] == EMPTY:
                grid[y][x] = BLOCK
                break
    return grid


def place_items(grid, symbol, count):
    positions = []
    for _ in range(count):
        while True:
            x, y = random.randrange(SIZE), random.randrange(SIZE)
            if grid[y][x] == EMPTY:
                grid[y][x] = symbol
                positions.append((x, y))
                break
    return positions


def move(position, action, grid, energy):
    dx = dy = 0
    jump = action.startswith("jump_")

    if "up" in action:
        dy = -1
    elif "down" in action:
        dy = 1
    elif "left" in action:
        dx = -1
    elif "right" in action:
        dx = 1

    if jump:
        if energy < ENER_PER_JUMP:
            step = 1
        else:
            step = 2
            energy -= ENER_PER_JUMP
    else:
        step = 1

    next_position = [position[0] + dx * step, position[1] + dy * step]
    if not in_bounds(next_position):
        return position, energy
    if grid[next_position[1]][next_position[0]] == BLOCK:
        return position, energy

    energy = max(0.0, energy - (ENER_PER_STEP if action != "place_teleporter" else 0.0))
    return next_position, energy


def collect_powerup(agent, position, grid):
    x, y = position
    symbol = grid[y][x]
    if symbol == EMPTY:
        return 0.0, None

    reward = REWARD_POWERUP
    effect = None
    if symbol == BOOST:
        agent.enercore_active = max(agent.enercore_active, 5)
        agent.enercore_steps = 5
        effect = "boost"
    elif symbol == ENER:
        agent.energy = min(ENERGY_MAX, agent.energy + 2.0)
        effect = "energy"
    elif symbol == RADAR:
        agent.radar_active = 8
        effect = "radar"
    elif symbol == ENERCORE:
        agent.enercore_active = 10
        agent.enercore_steps = 10
        effect = "enercore"

    if symbol in (BOOST, ENER, RADAR, ENERCORE):
        grid[y][x] = EMPTY
    return reward, effect


def tick_effects(agent):
    if agent.radar_active > 0:
        agent.radar_active -= 1
    if agent.enercore_active > 0:
        agent.enercore_active -= 1
    if agent.enercore_steps > 0:
        agent.enercore_steps -= 1


def print_map(grid, chaser_position, runner_position, chaser, runner,
              info="", logs=None, live=False):
    rendered = [row[:] for row in grid]
    rendered[chaser_position[1]][chaser_position[0]] = CHASER
    rendered[runner_position[1]][runner_position[0]] = RUNNER

    if live:
        clear_console()
    print(info)
    for row in rendered:
        print(" ".join(row))
    print("-" * 30)
    print(f"Chaser  energy={chaser.energy:.1f} radar={chaser.radar_active} core={chaser.enercore_active}")
    print(f"Runner  energy={runner.energy:.1f} radar={runner.radar_active} core={runner.enercore_active}")
    print("-" * 30)
    for line in (logs or [])[-8:]:
        print(line)


def run_round(chaser, runner, epsilon, grid_seed=None, visual=False, live=False, fps=4):
    if grid_seed is not None:
        random.seed(grid_seed)

    grid = make_map()
    place_items(grid, BOOST, BOOST_COUNT)
    place_items(grid, ENER, ENER_COUNT)
    place_items(grid, RADAR, RADAR_COUNT)
    place_items(grid, ENERCORE, ENERCORE_COUNT)

    while True:
        chaser_position = [random.randrange(SIZE), random.randrange(SIZE)]
        runner_position = [random.randrange(SIZE), random.randrange(SIZE)]
        if (grid[chaser_position[1]][chaser_position[0]] == EMPTY and
                grid[runner_position[1]][runner_position[0]] == EMPTY and
                manhattan(chaser_position, runner_position) > 3):
            break

    chaser.energy = ENERGY_MAX
    runner.energy = ENERGY_MAX
    chaser.radar_active = runner.radar_active = 0
    chaser.enercore_active = runner.enercore_active = 0
    chaser.has_teleporter = runner.has_teleporter = True
    teleporters = []

    chaser_replay = getattr(chaser, "replay", None)
    runner_replay = getattr(runner, "replay", None)

    total_chaser_reward = 0.0
    total_runner_reward = 0.0
    chaser_loss = runner_loss = None
    logs = []

    for step in range(MAX_STEPS):
        chaser_state = chaser.state_vector(chaser_position, runner_position, grid)
        runner_state = runner.state_vector(runner_position, chaser_position, grid)

        chaser_action = chaser.policy.act(chaser_state, epsilon)
        runner_action = runner.policy.act(runner_state, epsilon)
        chaser_action_name = ACTIONS[chaser_action]
        runner_action_name = ACTIONS[runner_action]

        if chaser_action_name == "place_teleporter" and chaser.has_teleporter:
            x, y = chaser_position
            if grid[y][x] == EMPTY:
                grid[y][x] = TELEPORTER
                teleporters.append(tuple(chaser_position))
                chaser.has_teleporter = False
                logs.append("Chaser placed teleporter")
            chaser_action_name = "none"

        if runner_action_name == "place_teleporter" and runner.has_teleporter:
            x, y = runner_position
            if grid[y][x] == EMPTY:
                grid[y][x] = TELEPORTER
                teleporters.append(tuple(runner_position))
                runner.has_teleporter = False
                logs.append("Runner placed teleporter")
            runner_action_name = "none"

        chaser_old_distance = manhattan(chaser_position, runner_position)
        runner_old_distance = chaser_old_distance

        if chaser_action_name == "none":
            next_chaser = chaser_position[:]
            chaser.energy = max(0.0, chaser.energy - ENER_PER_STEP)
        else:
            next_chaser, chaser.energy = move(chaser_position, chaser_action_name, grid, chaser.energy)

        if runner_action_name == "none":
            next_runner = runner_position[:]
            runner.energy = max(0.0, runner.energy - ENER_PER_STEP)
        else:
            next_runner, runner.energy = move(runner_position, runner_action_name, grid, runner.energy)

        if len(teleporters) == 2:
            for position, agent, name in (
                (next_chaser, chaser, "Chaser"),
                (next_runner, runner, "Runner"),
            ):
                if tuple(position) == teleporters[0]:
                    position[:] = list(teleporters[1])
                    logs.append(f"{name} used teleporter")
                elif tuple(position) == teleporters[1]:
                    position[:] = list(teleporters[0])
                    logs.append(f"{name} used teleporter")

        chaser_reward = REWARD_DISTANCE_STEP if manhattan(next_chaser, next_runner) < chaser_old_distance else -REWARD_DISTANCE_STEP
        runner_reward = REWARD_DISTANCE_STEP if manhattan(next_runner, next_chaser) > runner_old_distance else -REWARD_DISTANCE_STEP

        chaser_power_reward, chaser_effect = collect_powerup(chaser, next_chaser, grid)
        runner_power_reward, runner_effect = collect_powerup(runner, next_runner, grid)
        chaser_reward += chaser_power_reward
        runner_reward += runner_power_reward

        done = False
        if next_chaser == next_runner:
            chaser_reward += REWARD_CATCH
            runner_reward -= REWARD_CATCH
            done = True

        tick_effects(chaser)
        tick_effects(runner)

        chaser_next_state = chaser.state_vector(next_chaser, next_runner, grid)
        runner_next_state = runner.state_vector(next_runner, next_chaser, grid)

        if chaser_replay is not None:
            chaser_replay.push(chaser_state, chaser_action, chaser_reward, chaser_next_state, done)
        if runner_replay is not None:
            runner_replay.push(runner_state, runner_action, runner_reward, runner_next_state, done)

        chaser_loss = chaser.policy.learn(chaser_replay)
        runner_loss = runner.policy.learn(runner_replay)

        chaser_position, runner_position = next_chaser, next_runner
        total_chaser_reward += chaser_reward
        total_runner_reward += runner_reward

        if visual:
            print_map(
                grid, chaser_position, runner_position, chaser, runner,
                info=f"Step {step + 1}/{MAX_STEPS} | Chaser={total_chaser_reward:.2f} Runner={total_runner_reward:.2f}",
                logs=logs,
                live=live,
            )
            if live:
                time.sleep(1.0 / max(1, fps))

        if done:
            break

    timeout = not done
    if timeout:
        total_chaser_reward += REWARD_STEP_LIMIT
        total_runner_reward -= REWARD_STEP_LIMIT

    winner = "chaser" if done else "timeout"
    return {
        "winner": winner,
        "chaser_reward": total_chaser_reward,
        "runner_reward": total_runner_reward,
        "steps": step + 1,
        "chaser_loss": chaser_loss,
        "runner_loss": runner_loss,
    }


def save_checkpoint(path, chaser, runner, chaser_replay, runner_replay, episode, epsilon):
    torch.save({
        "episode": episode,
        "epsilon": epsilon,
        "version": 3,
        "size": SIZE,
        "actions": ACTIONS,
        "chaser_net": chaser.policy.net.state_dict(),
        "chaser_target": chaser.policy.target.state_dict(),
        "chaser_optimizer": chaser.policy.optimizer.state_dict(),
        "runner_net": runner.policy.net.state_dict(),
        "runner_target": runner.policy.target.state_dict(),
        "runner_optimizer": runner.policy.optimizer.state_dict(),
        "chaser_replay": list(chaser_replay.buffer),
        "runner_replay": list(runner_replay.buffer),
    }, path)


def load_checkpoint(path, chaser, runner, chaser_replay, runner_replay, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    chaser.policy.net.load_state_dict(checkpoint["chaser_net"])
    chaser.policy.target.load_state_dict(checkpoint["chaser_target"])
    runner.policy.net.load_state_dict(checkpoint["runner_net"])
    runner.policy.target.load_state_dict(checkpoint["runner_target"])

    try:
        chaser.policy.optimizer.load_state_dict(checkpoint["chaser_optimizer"])
        runner.policy.optimizer.load_state_dict(checkpoint["runner_optimizer"])
    except (KeyError, RuntimeError):
        pass

    chaser_replay.buffer.clear()
    runner_replay.buffer.clear()
    chaser_replay.buffer.extend(checkpoint.get("chaser_replay", []))
    runner_replay.buffer.extend(checkpoint.get("runner_replay", []))
    return int(checkpoint.get("episode", 0)), float(checkpoint.get("epsilon", EPSILON_START))


def run_experiment(args):
    set_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    chaser = Agent("chaser", device)
    runner = Agent("runner", device)
    chaser.replay = ReplayBuffer(BUFFER_CAP)
    runner.replay = ReplayBuffer(BUFFER_CAP)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = checkpoint_dir / "tagmind_metrics.csv"

    episode = 0
    epsilon = EPSILON_START
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        episode, epsilon = load_checkpoint(
            resume_path, chaser, runner,
            chaser.replay, runner.replay, device,
        )

    print(f"Device: {device} | starting episode: {episode + 1} | epsilon: {epsilon:.4f}")

    with metrics_path.open("a", newline="", encoding="utf-8") as csv_file:
        fieldnames = ("episode", "winner", "steps", "chaser_reward", "runner_reward",
                      "epsilon", "chaser_loss", "runner_loss", "chaser_buffer", "runner_buffer")
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if csv_file.tell() == 0:
            writer.writeheader()

        try:
            for episode in range(episode + 1, args.rounds + 1):
                result = run_round(
                    chaser, runner, epsilon,
                    visual=args.visual, live=args.live, fps=args.fps,
                )

                epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
                writer.writerow({
                    "episode": episode,
                    "winner": result["winner"],
                    "steps": result["steps"],
                    "chaser_reward": f'{result["chaser_reward"]:.6f}',
                    "runner_reward": f'{result["runner_reward"]:.6f}',
                    "epsilon": f'{epsilon:.6f}',
                    "chaser_loss": "" if result["chaser_loss"] is None else f'{result["chaser_loss"]:.6f}',
                    "runner_loss": "" if result["runner_loss"] is None else f'{result["runner_loss"]:.6f}',
                    "chaser_buffer": len(chaser.replay),
                    "runner_buffer": len(runner.replay),
                })

                if episode == 1 or episode % args.print_interval == 0 or episode == args.rounds:
                    print(
                        f"Episode {episode}/{args.rounds} | winner={result['winner']} "
                        f"| chaser_reward={result['chaser_reward']:.2f} "
                        f"| runner_reward={result['runner_reward']:.2f} "
                        f"| epsilon={epsilon:.4f}"
                    )

                if episode % args.save_interval == 0:
                    path = checkpoint_dir / f"tagmind_v3_ep{episode}.pth"
                    save_checkpoint(path, chaser, runner, chaser.replay, runner.replay, episode, epsilon)
                    print(f"Checkpoint saved: {path}")

                csv_file.flush()

        except KeyboardInterrupt:
            interrupt_path = checkpoint_dir / "tagmind_interrupt.pth"
            save_checkpoint(
                interrupt_path, chaser, runner,
                chaser.replay, runner.replay, episode, epsilon,
            )
            print(f"Interrupted. Checkpoint saved: {interrupt_path}")
            raise

    final_path = checkpoint_dir / "tagmind_final.pth"
    save_checkpoint(final_path, chaser, runner, chaser.replay, runner.replay, episode, epsilon)
    print(f"Training complete. Final checkpoint: {final_path}")
    print(f"Metrics: {metrics_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train TagMind Chaser/Runner DQN agents")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="tagmind_ckpt")
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--print-interval", type=int, default=50)
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--fps", type=int, default=4)

    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be at least 1")
    if args.save_interval < 1:
        parser.error("--save-interval must be at least 1")
    if args.print_interval < 1:
        parser.error("--print-interval must be at least 1")
    return args


if __name__ == "__main__":
    run_experiment(parse_args())
