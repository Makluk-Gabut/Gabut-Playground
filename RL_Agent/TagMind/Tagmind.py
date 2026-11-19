"""TagMind: DQN self-play sederhana untuk game kejar-kejaran."""

import argparse
import csv
import logging
import random
from collections import deque, namedtuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

SIZE = 10
MAX_STEPS = 50
BLOCK_COUNT = 18
ENERGY_ITEMS = 6

# Aksi jump bergerak dua petak jika energi mencukupi.
ACTIONS = ("up", "down", "left", "right",
           "jump_up", "jump_down", "jump_left", "jump_right")
EMPTY, BLOCK, ENERGY, CHASER, RUNNER = ".", "#", "E", "C", "R"

ENERGY_MAX = 6.0
ENERGY_COST_STEP = 0.25
ENERGY_COST_JUMP = 1.5

GAMMA = 0.95
LEARNING_RATE = 1e-3
EPSILON_START = 1.0
EPSILON_MIN = 0.08
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
REPLAY_CAPACITY = 20_000
WARMUP = 256
TARGET_SYNC_STEPS = 100
GRAD_CLIP = 5.0

# State: arah lawan (dx, dy), jarak, energi, dan 8 sel sekitar.
STATE_SIZE = 12
Transition = namedtuple("Transition", "state action reward next_state done")


def setup_logger(output_dir):
    logger = logging.getLogger("tagmind")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(output_dir / "training.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ReplayBuffer:
    def __init__(self, capacity=REPLAY_CAPACITY):
        self.data = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.data.append(Transition(
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self):
        batch = random.sample(self.data, min(BATCH_SIZE, len(self.data)))
        return (
            np.stack([t.state for t in batch]),
            np.asarray([t.action for t in batch], dtype=np.int64).reshape(-1, 1),
            np.asarray([t.reward for t in batch], dtype=np.float32).reshape(-1, 1),
            np.stack([t.next_state for t in batch]),
            np.asarray([t.done for t in batch], dtype=np.float32).reshape(-1, 1),
        )

    def __len__(self):
        return len(self.data)


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(STATE_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, len(ACTIONS)),
        )

    def forward(self, x):
        return self.layers(x)


class DQNAgent:
    def __init__(self, device):
        self.device = torch.device(device)
        self.online = QNetwork().to(self.device)
        self.target = QNetwork().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = optim.Adam(self.online.parameters(), lr=LEARNING_RATE)
        self.learn_steps = 0
        self.replay = ReplayBuffer()

    def act(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(len(ACTIONS))
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            return int(self.online(state_tensor).argmax(dim=1).item())

    def learn(self):
        if len(self.replay) < WARMUP:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample()
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        current_q = self.online(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target(next_states).max(dim=1, keepdim=True).values
            target_q = rewards + (1.0 - dones) * GAMMA * next_q

        loss = nn.functional.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), GRAD_CLIP)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % TARGET_SYNC_STEPS == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())


class Agent:
    def __init__(self, role, device):
        self.role = role
        self.brain = DQNAgent(device)
        self.energy = ENERGY_MAX


def make_map():
    grid = [[EMPTY for _ in range(SIZE)] for _ in range(SIZE)]

    for _ in range(BLOCK_COUNT):
        while True:
            x, y = random.randrange(SIZE), random.randrange(SIZE)
            if grid[y][x] == EMPTY:
                grid[y][x] = BLOCK
                break

    for _ in range(ENERGY_ITEMS):
        while True:
            x, y = random.randrange(SIZE), random.randrange(SIZE)
            if grid[y][x] == EMPTY:
                grid[y][x] = ENERGY
                break

    return grid


def get_state(agent, position, opponent, grid):
    dx = np.clip(opponent[0] - position[0], -SIZE, SIZE) / SIZE
    dy = np.clip(opponent[1] - position[1], -SIZE, SIZE) / SIZE
    distance = (
        abs(opponent[0] - position[0]) + abs(opponent[1] - position[1])
    ) / (2 * SIZE)

    x, y = position
    neighbor_positions = (
        (x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y),
        (x - 1, y - 1), (x + 1, y - 1),
        (x - 1, y + 1), (x + 1, y + 1),
    )

    nearby = []
    for nx, ny in neighbor_positions:
        if not (0 <= nx < SIZE and 0 <= ny < SIZE):
            nearby.append(1.0)
        elif grid[ny][nx] == BLOCK:
            nearby.append(1.0)
        elif grid[ny][nx] == ENERGY:
            nearby.append(0.5)
        else:
            nearby.append(0.0)

    return np.asarray(
        [dx, dy, distance, agent.energy / ENERGY_MAX, *nearby],
        dtype=np.float32,
    )


def take_action(position, action, grid, energy):
    name = ACTIONS[action]
    dx = dy = 0

    if "up" in name:
        dy = -1
    elif "down" in name:
        dy = 1
    elif "left" in name:
        dx = -1
    elif "right" in name:
        dx = 1

    is_jump = name.startswith("jump_")
    if is_jump and energy >= ENERGY_COST_JUMP:
        distance = 2
        energy -= ENERGY_COST_JUMP
    else:
        distance = 1

    candidate = [position[0] + dx * distance, position[1] + dy * distance]

    # Jump boleh melompati satu petak, tetapi tidak boleh mendarat di tembok/luar peta.
    if not (0 <= candidate[0] < SIZE and 0 <= candidate[1] < SIZE):
        candidate = position[:]
    elif grid[candidate[1]][candidate[0]] == BLOCK:
        candidate = position[:]

    if candidate != position:
        energy = max(0.0, energy - ENERGY_COST_STEP)

    return candidate, energy


def collect_energy(agent, position, grid):
    x, y = position
    if grid[y][x] == ENERGY:
        grid[y][x] = EMPTY
        agent.energy = min(ENERGY_MAX, agent.energy + 2.0)
        return 0.2
    return 0.0


def run_episode(chaser, runner, epsilon):
    grid = make_map()

    while True:
        chaser_pos = [random.randrange(SIZE), random.randrange(SIZE)]
        runner_pos = [random.randrange(SIZE), random.randrange(SIZE)]
        if (grid[chaser_pos[1]][chaser_pos[0]] == EMPTY
                and grid[runner_pos[1]][runner_pos[0]] == EMPTY
                and sum(abs(a - b) for a, b in zip(chaser_pos, runner_pos)) > 3):
            break

    chaser.energy = runner.energy = ENERGY_MAX
    chaser_path = [chaser_pos[:]]
    runner_path = [runner_pos[:]]
    chaser_total = runner_total = 0.0
    chaser_losses = []
    runner_losses = []
    caught = False

    for step in range(MAX_STEPS):
        chaser_state = get_state(chaser, chaser_pos, runner_pos, grid)
        runner_state = get_state(runner, runner_pos, chaser_pos, grid)

        chaser_action = chaser.brain.act(chaser_state, epsilon)
        runner_action = runner.brain.act(runner_state, epsilon)

        next_chaser, chaser.energy = take_action(
            chaser_pos, chaser_action, grid, chaser.energy
        )
        next_runner, runner.energy = take_action(
            runner_pos, runner_action, grid, runner.energy
        )

        old_distance = abs(chaser_pos[0] - runner_pos[0]) + abs(chaser_pos[1] - runner_pos[1])
        new_distance = abs(next_chaser[0] - next_runner[0]) + abs(next_chaser[1] - next_runner[1])

        chaser_reward = 0.05 if new_distance < old_distance else -0.05
        runner_reward = 0.05 if new_distance > old_distance else -0.05

        chaser_reward += collect_energy(chaser, next_chaser, grid)
        runner_reward += collect_energy(runner, next_runner, grid)

        caught = next_chaser == next_runner
        if caught:
            chaser_reward += 1.0
            runner_reward -= 1.0

        timed_out = step == MAX_STEPS - 1 and not caught
        if timed_out:
            # Runner memperoleh reward jika bertahan sampai batas episode.
            chaser_reward -= 0.2
            runner_reward += 0.2

        done = caught or timed_out
        next_chaser_state = get_state(chaser, next_chaser, next_runner, grid)
        next_runner_state = get_state(runner, next_runner, next_chaser, grid)

        # Buffer terpisah: pengalaman tiap agen hanya melatih agennya sendiri.
        chaser.brain.replay.push(
            chaser_state, chaser_action, chaser_reward, next_chaser_state, done
        )
        runner.brain.replay.push(
            runner_state, runner_action, runner_reward, next_runner_state, done
        )

        chaser_loss = chaser.brain.learn()
        runner_loss = runner.brain.learn()
        if chaser_loss is not None:
            chaser_losses.append(chaser_loss)
        if runner_loss is not None:
            runner_losses.append(runner_loss)

        chaser_pos, runner_pos = next_chaser, next_runner
        chaser_path.append(chaser_pos[:])
        runner_path.append(runner_pos[:])
        chaser_total += chaser_reward
        runner_total += runner_reward

        if done:
            break

    return {
        "winner": "chaser" if caught else "runner",
        "steps": step + 1,
        "chaser_reward": chaser_total,
        "runner_reward": runner_total,
        "chaser_loss": float(np.mean(chaser_losses)) if chaser_losses else "",
        "runner_loss": float(np.mean(runner_losses)) if runner_losses else "",
        "grid": grid,
        "chaser_path": chaser_path,
        "runner_path": runner_path,
    }


def save_gameplay_image(result, episode, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        return False

    colors = ["#f3f4f6", "#333333", "#38a169", "#e53e3e", "#3182ce"]
    cmap = ListedColormap(colors)
    values = {EMPTY: 0, BLOCK: 1, ENERGY: 2}

    image = np.array([
        [values.get(cell, 0) for cell in row]
        for row in result["grid"]
    ])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap=cmap, vmin=0, vmax=4)

    cp = np.asarray(result["chaser_path"])
    rp = np.asarray(result["runner_path"])
    ax.plot(cp[:, 0], cp[:, 1], color="#e53e3e", marker=".", label="Chaser path")
    ax.plot(rp[:, 0], rp[:, 1], color="#3182ce", marker=".", label="Runner path")
    ax.scatter(cp[-1, 0], cp[-1, 1], color="#e53e3e", s=100, marker="s")
    ax.scatter(rp[-1, 0], rp[-1, 1], color="#3182ce", s=100, marker="o")

    ax.set_xticks(range(SIZE))
    ax.set_yticks(range(SIZE))
    ax.set_xlim(-0.5, SIZE - 0.5)
    ax.set_ylim(SIZE - 0.5, -0.5)
    ax.grid(alpha=0.25)
    ax.set_title(
        f"Episode {episode} | winner={result['winner']} | steps={result['steps']}"
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def update_plot(csv_path, plot_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return False

    episodes = [int(row["episode"]) for row in rows]
    chaser_rewards = [float(row["chaser_reward"]) for row in rows]
    runner_rewards = [float(row["runner_reward"]) for row in rows]
    chaser_wins = [1.0 if row["winner"] == "chaser" else 0.0 for row in rows]

    window = min(20, len(rows))
    rolling_wins = np.convolve(
        chaser_wins, np.ones(window) / window, mode="valid"
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot(episodes, chaser_rewards, alpha=0.35, label="Chaser reward")
    axes[0].plot(episodes, runner_rewards, alpha=0.35, label="Runner reward")
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(episodes[window - 1:], rolling_wins, color="#e53e3e")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel(f"Chaser win rate ({window}-episode window)")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.25)

    fig.suptitle("TagMind training")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    return True


def save_checkpoint(path, chaser, runner, episode, epsilon):
    torch.save({
        "version": 1,
        "episode": episode,
        "epsilon": epsilon,
        "chaser_online": chaser.brain.online.state_dict(),
        "chaser_target": chaser.brain.target.state_dict(),
        "chaser_optimizer": chaser.brain.optimizer.state_dict(),
        "chaser_replay": list(chaser.brain.replay.data),
        "chaser_learn_steps": chaser.brain.learn_steps,
        "runner_online": runner.brain.online.state_dict(),
        "runner_target": runner.brain.target.state_dict(),
        "runner_optimizer": runner.brain.optimizer.state_dict(),
        "runner_replay": list(runner.brain.replay.data),
        "runner_learn_steps": runner.brain.learn_steps,
    }, path)


def load_checkpoint(path, chaser, runner, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # compatibility with older PyTorch versions
        checkpoint = torch.load(path, map_location=device)

    if checkpoint.get("version") != 1:
        raise ValueError("Checkpoint version is incompatible with this script.")

    for agent, prefix in ((chaser, "chaser"), (runner, "runner")):
        agent.brain.online.load_state_dict(checkpoint[f"{prefix}_online"])
        agent.brain.target.load_state_dict(checkpoint[f"{prefix}_target"])
        agent.brain.optimizer.load_state_dict(checkpoint[f"{prefix}_optimizer"])
        agent.brain.replay.data.extend(checkpoint.get(f"{prefix}_replay", []))
        agent.brain.learn_steps = int(checkpoint.get(f"{prefix}_learn_steps", 0))

    return int(checkpoint["episode"]), float(checkpoint["epsilon"])


def train(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)

    set_seed(args.seed)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diminta, tetapi GPU CUDA tidak tersedia.")

    chaser = Agent("chaser", device)
    runner = Agent("runner", device)
    start_episode = 0
    epsilon = EPSILON_START

    if args.resume:
        start_episode, epsilon = load_checkpoint(
            args.resume, chaser, runner, device
        )
        logger.info("Resume dari episode %d; epsilon %.4f", start_episode, epsilon)

    logger.info("Device: %s", device)
    logger.info("Folder hasil: %s", output_dir)

    csv_path = output_dir / "tagmind_metrics.csv"
    fields = [
        "episode", "winner", "steps", "chaser_reward", "runner_reward",
        "epsilon", "chaser_loss", "runner_loss", "chaser_buffer", "runner_buffer",
    ]
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        if write_header:
            writer.writeheader()

        try:
            for episode in range(start_episode + 1, args.rounds + 1):
                result = run_episode(chaser, runner, epsilon)
                epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

                writer.writerow({
                    "episode": episode,
                    "winner": result["winner"],
                    "steps": result["steps"],
                    "chaser_reward": f"{result['chaser_reward']:.6f}",
                    "runner_reward": f"{result['runner_reward']:.6f}",
                    "epsilon": f"{epsilon:.6f}",
                    "chaser_loss": result["chaser_loss"],
                    "runner_loss": result["runner_loss"],
                    "chaser_buffer": len(chaser.brain.replay),
                    "runner_buffer": len(runner.brain.replay),
                })
                csv_file.flush()

                if episode % args.image_interval == 0 or episode == args.rounds:
                    image_path = frames_dir / f"episode_{episode:06d}.png"
                    if save_gameplay_image(result, episode, image_path):
                        logger.info("Gambar gameplay: %s", image_path)
                    update_plot(csv_path, output_dir / "reward_plot.png")

                if episode == start_episode + 1 or episode % args.log_interval == 0:
                    logger.info(
                        "Episode %d/%d | winner=%s | steps=%d | "
                        "reward C=%.3f R=%.3f | epsilon=%.3f",
                        episode, args.rounds, result["winner"], result["steps"],
                        result["chaser_reward"], result["runner_reward"], epsilon,
                    )

                if episode % args.save_interval == 0:
                    checkpoint_path = output_dir / f"checkpoint_ep{episode}.pth"
                    save_checkpoint(checkpoint_path, chaser, runner, episode, epsilon)
                    logger.info("Checkpoint: %s", checkpoint_path)

        except KeyboardInterrupt:
            checkpoint_path = output_dir / "checkpoint_interrupted.pth"
            save_checkpoint(checkpoint_path, chaser, runner, episode, epsilon)
            logger.info("Training dihentikan. Checkpoint: %s", checkpoint_path)
            raise

    final_path = output_dir / "checkpoint_final.pth"
    save_checkpoint(final_path, chaser, runner, episode, epsilon)
    update_plot(csv_path, output_dir / "reward_plot.png")
    logger.info("Training selesai.")
    logger.info("CSV: %s", csv_path)
    logger.info("Grafik: %s", output_dir / "reward_plot.png")
    logger.info("Gambar: %s", frames_dir)
    logger.info("Checkpoint akhir: %s", final_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train TagMind DQN self-play agents")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", default="/content/drive/MyDrive/TagMind")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--image-interval", type=int, default=100)
    args = parser.parse_args()

    for name in ("rounds", "save_interval", "log_interval", "image_interval"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} harus >= 1")
    return args


if __name__ == "__main__":
    train(parse_args())
