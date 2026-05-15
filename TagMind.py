# TagMind v - Paling ambisiuss yang pernah kucoba 


import random
import math
import time
import argparse
import pathlib
import os
import json
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ====== PARAMETER GLOBAL ======
SIZE = 10
ROUNDS = 100
MAX_STEPS = 50
VISUAL = False        # kalo mau ada visua di tiap rondenga kasih true aja
PRINT_INTERVAL = 50

ACTIONS = [
    "up", "down", "left", "right",
    "jump_up", "jump_down", "jump_left", "jump_right",
    "place_teleporter"
]

EMPTY = "."
BLOCK = "#"
CHASER = "C"
RUNNER = "R"

BOOST = "B"
ENER = "E"
RADAR = "T"
ENERCORE = "X"
TELEPORTER = "P"

BLOCK_COUNT = 18
BOOST_COUNT = 4
ENER_COUNT = 5
ENERGY_MAX = 6
ENER_PER_STEP = 0.5
ENER_PER_USE = 2
VISION = 4

LEARN_RATE = 1e-3
DISCOUNT = 0.9
EPSILON = 0.08
BATCH_SIZE = 32
BUFFER_CAP = 2000
SYNC_TARGET_EVERY = 100
MIN_BUFFER_BEFORE_LEARN = 200
SAVE_EVERY_EPISODES = 2
GRAD_CLIP = 5.0

REWARD_CATCH = 1.0
REWARD_STEP_CHASER = 0.05
REWARD_STEP_RUNNER = 0.05
REWARD_POWERUP = 0.2

BUFF_REWARDS = {
    "energy_eff": +0.2, "energy_ineff": -0.5,
    "boost_eff": +0.2, "boost_ineff": -0.5,
    "radar_eff": +0.7, "radar_ineff": -0.9,
    "enercore_eff": +1, "enercore_ineff": -1.5
}

# ====== UTIL ======
def in_bounds(p):
    return 0 <= p[0] < SIZE and 0 <= p[1] < SIZE

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# Map helpers
def make_map():
    g = [[EMPTY for _ in range(SIZE)] for _ in range(SIZE)]
    for _ in range(BLOCK_COUNT):
        x, y = random.randint(0, SIZE-1), random.randint(0, SIZE-1)
        g[y][x] = BLOCK
    return g

def place_items(g, sym, count):
    placed = []
    for _ in range(count):
        while True:
            x, y = random.randint(0, SIZE-1), random.randint(0, SIZE-1)
            if g[y][x] == EMPTY:
                g[y][x] = sym
                placed.append((x, y))
                break
    return placed

def clear_console():
    # portable clear
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

def print_map(g, c, r, chaser_agent, runner_agent, info="", logs=None, live=False):
    """
    Print the map. Jika live=True, konsol akan dibersihkan sebelum mencetak untuk membuat animasi.
    """
    if logs is None:
        logs = []
    gm = [row[:] for row in g]
    if in_bounds(c):
        gm[c[1]][c[0]] = CHASER
    if in_bounds(r):
        gm[r[1]][r[0]] = RUNNER
    # print info safely
    if info is None:
        info = ""
    if live:
        clear_console()
    print(info)
    for row in gm:
        print(" ".join(row))
    print("-" * 30)
    print(f"Chaser | Radar:{getattr(chaser_agent, 'radar_active', 0)} | Enercore:{getattr(chaser_agent, 'enercore_active', 0)}")
    print(f"Runner | Radar:{getattr(runner_agent, 'radar_active', 0)} | Enercore:{getattr(runner_agent, 'enercore_active', 0)}")
    print("-" * 30)
    # print only last few logs to avoid clutter
    for line in logs[-8:]:
        print(line)
    print("=" * 30)

# ====== PyTorch Neural Network, Agent & Replay Buffer ======
Transition = namedtuple('Transition', ('s', 'a', 'r', 'ns', 'd'))

class ReplayBuffer:
    def __init__(self, capacity=BUFFER_CAP):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        # normalize states to numpy arrays float32 for consistent stacking later
        s, a, r, ns, d = args
        s = np.asarray(s, dtype=np.float32)
        ns = np.asarray(ns, dtype=np.float32)
        # a, r, d keep as scalars / ints / floats
        self.buffer.append(Transition(s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        # stack numpy arrays (fast) then convert to torch once
        states = np.stack([b.s for b in batch]).astype(np.float32)
        next_states = np.stack([b.ns for b in batch]).astype(np.float32)
        actions = np.array([b.a for b in batch], dtype=np.int64).reshape(-1, 1)
        rewards = np.array([b.r for b in batch], dtype=np.float32).reshape(-1, 1)
        dones = np.array([b.d for b in batch], dtype=np.float32).reshape(-1, 1)

        s = torch.from_numpy(states)
        a = torch.from_numpy(actions)
        r = torch.from_numpy(rewards)
        ns = torch.from_numpy(next_states)
        d = torch.from_numpy(dones)

        return s, a, r, ns, d

    def __len__(self):
        return len(self.buffer)

    def to_list(self):
        # convert to plain Python-friendly list for checkpointing (convert np arrays to lists)
        out = []
        for t in self.buffer:
            out.append([t.s.tolist(), int(t.a), float(t.r), t.ns.tolist(), float(t.d)])
        return out

    def load_list(self, L):
        # L should be list of [s_list, a, r, ns_list, d]
        self.buffer = deque(maxlen=self.buffer.maxlen)
        for x in L:
            try:
                s = np.asarray(x[0], dtype=np.float32)
                a = int(x[1])
                r = float(x[2])
                ns = np.asarray(x[3], dtype=np.float32)
                d = float(x[4])
                self.buffer.append(Transition(s, a, r, ns, d))
            except Exception:
                # ignore malformed entries
                continue

class DQNNet(nn.Module):
    def __init__(self, input_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, obs_dim, n_actions, lr=LEARN_RATE, gamma=DISCOUNT, device='cpu'):
        self.device = torch.device(device)
        self.net = DQNNet(obs_dim, n_actions).to(self.device)
        self.target = DQNNet(obs_dim, n_actions).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.n_actions = n_actions
        self.steps = 0

    def act(self, obs, eps=EPSILON):
        if random.random() < eps:
            return random.randrange(self.n_actions)
        obs_v = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        q = self.net(obs_v)
        return int(torch.argmax(q, dim=1).item())

    def learn(self, batch):
        s, a, r, ns, d = batch
        s = s.to(self.device); a = a.to(self.device); r = r.to(self.device); ns = ns.to(self.device); d = d.to(self.device)
        q_vals = self.net(s).gather(1, a)
        with torch.no_grad():
            next_q = self.target(ns).max(1)[0].unsqueeze(1)
            target = r + (1.0 - d) * self.gamma * next_q
        loss = nn.functional.mse_loss(q_vals, target)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), GRAD_CLIP)
        self.opt.step()
        return loss.item()

    def sync_target(self):
        self.target.load_state_dict(self.net.state_dict())

# ====== Agent wrapper that preserves original game features ======
class Agent:
    def __init__(self, vision=VISION, device='cpu'):
        self.vision = vision
        self.radar_active = 0
        self.enercore_active = 0
        self.enercore_steps = 0
        self.has_teleporter = True
        obs_dim = 8
        self.policy = DQNAgent(obs_dim, len(ACTIONS), device=device)

    def state_vec(self, selfpos, other, energy):
        dx = clamp(other[0] - selfpos[0], -self.vision, self.vision)
        dy = clamp(other[1] - selfpos[1], -self.vision, self.vision)
        dist = manhattan(selfpos, other) / (SIZE * 2)
        return np.array([dx / self.vision, dy / self.vision, dist,
                         energy / ENERGY_MAX,
                         1.0 if self.radar_active > 0 else 0.0,
                         1.0 if self.enercore_active > 0 else 0.0,
                         random.random(), 1.0], dtype=np.float32)

    def act(self, state, eps=EPSILON):
        a_idx = self.policy.act(state, eps=eps)
        return ACTIONS[a_idx], a_idx

    def learn(self, transition_batch):
        return self.policy.learn(transition_batch)

# ====== Movement ======
def move(pos, act, g, energy):
    dx, dy = 0, 0
    if "up" in act:
        dy = -1
    if "down" in act:
        dy = 1
    if "left" in act:
        dx = -1
    if "right" in act:
        dx = 1
    step = 2 if "jump" in act and energy >= ENER_PER_USE else 1
    new = [pos[0] + dx * step, pos[1] + dy * step]
    if not in_bounds(new) or g[new[1]][new[0]] == BLOCK:
        return pos
    return new

# ====== Simulation Round (integrates replay buffer and agent learning) ======
def run_round(chaser, runner, visual=False, replay=None, device='cpu', live=False, fps=4):
    g = make_map()
    boosts = place_items(g, BOOST, BOOST_COUNT)
    eners = place_items(g, ENER, ENER_COUNT)
    radar_pos = place_items(g, RADAR, 1)
    enercore_c = place_items(g, ENERCORE, 1)
    enercore_r = place_items(g, ENERCORE, 1)

    teleporter_pair = []

    while True:
        c = [random.randint(0, SIZE-1), random.randint(0, SIZE-1)]
        r = [random.randint(0, SIZE-1), random.randint(0, SIZE-1)]
        if g[c[1]][c[0]] == EMPTY and g[r[1]][r[0]] == EMPTY and manhattan(c, r) > 3:
            break

    ec, er = ENERGY_MAX, ENERGY_MAX
    total_c, total_r = 0.0, 0.0
    logs = []

    frame_delay = 1.0 / max(1, fps)

    for step in range(MAX_STEPS):
        dist_before = manhattan(c, r)
        sc = chaser.state_vec(c, r, ec)
        sr = runner.state_vec(r, c, er)

        ac_str, ac_idx = chaser.act(sc, eps=EPSILON)
        ar_str, ar_idx = runner.act(sr, eps=EPSILON)

        # handle teleporter placing
        if ac_str == "place_teleporter" and chaser.has_teleporter:
            if g[c[1]][c[0]] == EMPTY:
                g[c[1]][c[0]] = TELEPORTER
                teleporter_pair.append(tuple(c))
                logs.append(f"[CHASER] placed teleporter at {tuple(c)}")
                chaser.has_teleporter = False
        if ar_str == "place_teleporter" and runner.has_teleporter:
            if g[r[1]][r[0]] == EMPTY:
                g[r[1]][r[0]] = TELEPORTER
                teleporter_pair.append(tuple(r))
                logs.append(f"[RUNNER] placed teleporter at {tuple(r)}")
                runner.has_teleporter = False

        nc = move(c, ac_str, g, ec)
        nr = move(r, ar_str, g, er)

        # teleporter usage if pair ready
        if len(teleporter_pair) == 2:
            if tuple(nc) == teleporter_pair[0]:
                nc = list(teleporter_pair[1])
                logs.append(f"[CHASER] used teleporter -> {tuple(nc)}")
            elif tuple(nc) == teleporter_pair[1]:
                nc = list(teleporter_pair[0])
                logs.append(f"[CHASER] used teleporter -> {tuple(nc)}")
            if tuple(nr) == teleporter_pair[0]:
                nr = list(teleporter_pair[1])
                logs.append(f"[RUNNER] used teleporter -> {tuple(nr)}")
            elif tuple(nr) == teleporter_pair[1]:
                nr = list(teleporter_pair[0])
                logs.append(f"[RUNNER] used teleporter -> {tuple(nr)}")

        # rewards based on distance change
        reward_c = REWARD_STEP_CHASER if dist_before > manhattan(nc, nr) else -REWARD_STEP_CHASER
        reward_r = REWARD_STEP_RUNNER if dist_before < manhattan(nc, nr) else -REWARD_STEP_RUNNER
        done = False

        if nc == nr:
            reward_c += REWARD_CATCH
            reward_r -= REWARD_CATCH
            logs.append("[SYSTEM] Runner caught!")
            done = True

        sc2 = chaser.state_vec(nc, nr, ec)
        sr2 = runner.state_vec(nr, nc, er)

        # store transitions in replay
        if replay is not None:
            replay.push(sc, ac_idx, reward_c, sc2, float(done))
            replay.push(sr, ar_idx, reward_r, sr2, float(done))
            # if buffer large enough, do a learning step
            if len(replay) >= MIN_BUFFER_BEFORE_LEARN:
                batch = replay.sample(BATCH_SIZE)
                loss_c = chaser.learn(batch)
                loss_r = runner.learn(batch)
                # occasional target sync
                chaser.policy.steps += 1
                runner.policy.steps += 1
                if chaser.policy.steps % SYNC_TARGET_EVERY == 0:
                    chaser.policy.sync_target()
                if runner.policy.steps % SYNC_TARGET_EVERY == 0:
                    runner.policy.sync_target()

        c, r = nc, nr
        total_c += reward_c
        total_r += reward_r

        if visual:
            info = f"Step {step+1}/{MAX_STEPS} | C:{total_c:.2f} R:{total_r:.2f}"
            print_map(g, c, r, chaser, runner, info, logs, live=live)
            if live:
                # delay to control animation speed, but ensure minimal overhead
                time.sleep(frame_delay)

        if done:
            # final print to show the done state in live mode
            if visual and live:
                info = f"Step {step+1}/{MAX_STEPS} | (DONE) C:{total_c:.2f} R:{total_r:.2f}"
                print_map(g, c, r, chaser, runner, info, logs, live=live)
            break

    return c == r, total_c, total_r

# ====== Checkpoint helpers ======

def save_checkpoint(path, chaser, runner, replay, episode):
    data = {
        'episode': episode,
        'chaser_net': chaser.policy.net.state_dict(),
        'runner_net': runner.policy.net.state_dict(),
        'chaser_opt': chaser.policy.opt.state_dict(),
        'runner_opt': runner.policy.opt.state_dict(),
        'replay': replay.to_list()
    }
    torch.save(data, path)

def load_checkpoint(path, chaser, runner, replay):
    data = torch.load(path, map_location='cpu')
    chaser.policy.net.load_state_dict(data['chaser_net'])
    runner.policy.net.load_state_dict(data['runner_net'])
    # try load optimizer states if shapes match
    try:
        chaser.policy.opt.load_state_dict(data['chaser_opt'])
        runner.policy.opt.load_state_dict(data['runner_opt'])
    except Exception:
        pass
    try:
        replay.load_list(data.get('replay', []))
    except Exception:
        pass
    return data.get('episode', 0)

# ====== MAIN EXPERIMENT ======
def run_experiment(rounds=ROUNDS, visual=VISUAL, resume=None, device='cpu', live=False, fps=4):
    # device param left as-is (we default to cpu for most users)
    chaser = Agent(VISION, device=device)
    runner = Agent(VISION, device=device)
    replay = ReplayBuffer()

    # resume if provided
    start_episode = 1
    ckpt_dir = pathlib.Path('tagmind_ckpt')
    ckpt_dir.mkdir(exist_ok=True)
    if resume is not None and pathlib.Path(resume).exists():
        start_episode = load_checkpoint(resume, chaser, runner, replay) + 1
        print(f"Resumed from checkpoint episode {start_episode-1}")

    win_c = 0
    total_c = 0.0
    total_r = 0.0

    try:
        for i in range(start_episode, rounds+1):
            win, pc, pr = run_round(chaser, runner, visual=visual, replay=replay, device=device, live=live, fps=fps)
            if win:
                win_c += 1
            total_c += pc
            total_r += pr
            if i % PRINT_INTERVAL == 0 or i == 1 or i == rounds:
                print(f"Ronde {i}: Chaser menang {win_c} | Poin C:{total_c:.2f} R:{total_r:.2f}")

            # periodic autosave
            if i % SAVE_EVERY_EPISODES == 0:
                path = ckpt_dir / f"tagmind_v3_ep{i}.pth"
                save_checkpoint(path, chaser, runner, replay, i)
                print(f"Autosaved checkpoint -> {path}")

    except KeyboardInterrupt:
        print('Training interrupted by user — saving checkpoint...')
        path = ckpt_dir / f"tagmind_v3_ep_interrupt.pth"
        save_checkpoint(path, chaser, runner, replay, i)
        print(f"Saved interrupt checkpoint -> {path}")
        raise

    print(f"=== HASIL AKHIR ===")
    print(f"Chaser menang: {win_c} | Runner menang: {rounds-win_c}")
    print(f"Total poin: Chaser {total_c:.2f} | Runner {total_r:.2f}")

    # final checkpoint
    path = ckpt_dir / f"tagmind_pth"
    save_checkpoint(path, chaser, runner, replay, rounds)
    print('Checkpoint saved to', path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=ROUNDS)
    parser.add_argument('--visual', action='store_true', help='print map each step')
    parser.add_argument('--live', action='store_true', help='use live ASCII rendering (clears terminal each frame)')
    parser.add_argument('--fps', type=int, default=4, help='frames per second for live rendering')
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    # if user specified device 'auto', try to detect CUDA
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    run_experiment(rounds=args.rounds, visual=args.visual, resume=args.resume, device=device, live=args.live, fps=args.fps)



