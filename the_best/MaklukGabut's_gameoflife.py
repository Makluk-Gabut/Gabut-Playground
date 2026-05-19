"""
oke ini pertama kali aku bikin RL yang hampir sempurna
╔══════════════════════════════════════════════════════════════╗
║          MaklukGabut's Game of Life  —  ALL-IN-ONE          ║
║                        v4.0  (final)                        ║
╠══════════════════════════════════════════════════════════════╣
║  Satu file berisi segalanya:                                 ║
║    • Game engine  (agen + lingkungan 4-fase atomik)          ║
║    • RL engine    (12 NN, Double DQN, Replay Buffer)         ║
║    • Save / Load  (checkpoint otomatis)                      ║
║    • Statistik    (catat + grafik matplotlib)                ║
║    • Visualizer   (real-time pygame)                         ║
╠══════════════════════════════════════════════════════════════╣
║  Cara pakai:                                                 ║
║    python makluk_gabut.py train            ← mulai baru      ║
║    python makluk_gabut.py train --resume   ← lanjut          ║
║    python makluk_gabut.py train --visual   ← + visualisasi   ║
║    python makluk_gabut.py demo             ← tonton AI       ║
║    python makluk_gabut.py plot             ← lihat grafik     ║
║    python makluk_gabut.py list             ← daftar checkpoint║
╚══════════════════════════════════════════════════════════════╝
"""

import os, sys, json, random, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, Counter
from datetime import datetime

try:
    import pygame
    PYGAME_OK = True
except ImportError:
    PYGAME_OK = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MPL_OK = True
except ImportError:
    MPL_OK = False


# ══════════════════════════════════════════════════════════════
# SEKSI 1 — KONFIGURASI GLOBAL
# ══════════════════════════════════════════════════════════════
NUM_AGENTS  = 6
GRID_SIZE   = 12
ACTION_MAP  = [
    "IDLE",
    "MOVE_UP",  "MOVE_DOWN",  "MOVE_LEFT",  "MOVE_RIGHT",
    "BOOST_UP", "BOOST_DOWN", "BOOST_LEFT", "BOOST_RIGHT",
    "TAKE_FOOD", "EAT_FOOD", "GIVE_FOOD",
    "REVIVE", "CARRY_AGENT", "DROP_AGENT", "SHARE_INFO",
]
ACTION_SIZE = len(ACTION_MAP)   # 16
STATE_SIZE  = 8                 # x,y,energy,boost,inv,friend,sos,id

# Training
CFG = dict(
    episodes            = 500,
    max_steps           = 200,
    batch_size          = 64,
    gamma               = 0.99,
    lr                  = 1e-3,
    epsilon_start       = 1.0,
    epsilon_min         = 0.05,
    epsilon_decay       = 0.995,
    target_update_steps = 500,
    train_every_n_steps = 4,
    save_every_n_eps    = 50,
    plot_every_n_eps    = 100,
    save_dir            = "checkpoints",
    stats_plot_path     = "training_stats.png",
)

# Visualizer
VIZ = dict(
    cell        = 52,
    panel_w     = 290,
    default_fps = 10,
)


# ══════════════════════════════════════════════════════════════
# SEKSI 2 — KELAS AGEN
# ══════════════════════════════════════════════════════════════
class MaklukGabut:
    def __init__(self, agent_id, x, y):
        self.id            = agent_id
        self.x, self.y     = x, y
        self.energy        = 50
        self.steps_taken   = 0
        self.is_dormant    = False
        self.dormant_timer = 0
        self.is_dead       = False
        self.inventory        = []
        self.carrying         = None
        self.is_being_carried = False
        self.boost_charge  = 0
        self.boost_stacks  = 0
        self.food_memory   = None

    def release_carrying(self):
        if self.carrying is not None:
            self.carrying.is_being_carried = False
            self.carrying = None


# ══════════════════════════════════════════════════════════════
# SEKSI 3 — LINGKUNGAN (4-FASE ATOMIK)
# ══════════════════════════════════════════════════════════════
class MaklukGabutEnv:
    def __init__(self, grid_size=GRID_SIZE, num_agents=NUM_AGENTS):
        self.grid_size  = grid_size
        self.num_agents = num_agents
        self.step_count = 0
        self.agents = [
            MaklukGabut(i,
                        np.random.randint(0, grid_size),
                        np.random.randint(0, grid_size))
            for i in range(num_agents)
        ]
        self.food_grid = np.zeros((grid_size, grid_size), dtype=bool)
        self._spawn_food(15)

    def _spawn_food(self, n):
        done = attempts = 0
        while done < n and attempts < n * 10:
            x, y = np.random.randint(0, self.grid_size, size=2)
            if not self.food_grid[x, y]:
                self.food_grid[x, y] = True
                done += 1
            attempts += 1

    def _clamp(self, x, y):
        return (max(0, min(self.grid_size - 1, x)),
                max(0, min(self.grid_size - 1, y)))

    @staticmethod
    def _dir(action):
        if   "UP"    in action: return  0, -1
        elif "DOWN"  in action: return  0,  1
        elif "LEFT"  in action: return -1,  0
        elif "RIGHT" in action: return  1,  0
        return 0, 0

    def get_agent_at(self, x, y):
        for a in self.agents:
            if a.x == x and a.y == y and not a.is_dead and not a.is_being_carried:
                return a
        return None

    def get_neighbors(self, agent, radius=1):
        out = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == dy == 0: continue
                nx, ny = agent.x + dx, agent.y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    n = self.get_agent_at(nx, ny)
                    if n and n.id != agent.id:
                        out.append(n)
        return out

    def get_sos_signals(self, agent):
        return [(n.x, n.y) for n in self.get_neighbors(agent, radius=2) if n.is_dormant]

    # ── Step utama ───────────────────────────────────────────
    def step(self, actions_dict):
        self.step_count += 1
        if self.step_count % 20 == 0:
            self._spawn_food(3)

        alive_at_start   = {a.id for a in self.agents if not a.is_dead}
        pairing_penalty  = {aid: 0 for aid in alive_at_start}
        dead_this_step   = []

        # ── FASE 0: Dormant timer & Boost charge ─────────────
        for a in self.agents:
            if a.is_dead or a.is_being_carried: continue
            if a.is_dormant:
                a.dormant_timer += 1
                if a.dormant_timer >= 60:
                    a.release_carrying()
                    a.is_dead = True
                    dead_this_step.append(a.id)
                continue
            if a.boost_stacks < 5:
                a.boost_charge += 1
                if a.boost_charge >= 10:
                    a.boost_stacks += 1
                    a.boost_charge = 0

        active = [a for a in self.agents
                  if not a.is_dead and not a.is_dormant and not a.is_being_carried]

        # ── FASE 1: Deklarasi niat gerak (simultan) ───────────
        intents = {}
        for a in active:
            act = actions_dict.get(a.id, "IDLE")
            if act.startswith("MOVE_"):
                dx, dy = self._dir(act)
                intents[a.id] = dict(dest=self._clamp(a.x+dx, a.y+dy),
                                     wp=None, is_boost=False)
            elif act.startswith("BOOST_") and a.boost_stacks > 0:
                dx, dy = self._dir(act)
                wp   = self._clamp(a.x + dx,     a.y + dy)
                dest = self._clamp(a.x + dx * 2, a.y + dy * 2)
                intents[a.id] = dict(dest=dest, wp=wp, is_boost=True)
            else:
                intents[a.id] = dict(dest=(a.x, a.y), wp=None, is_boost=False)

        # ── FASE 2: Resolusi konflik gerakan ─────────────────
        moving_ids  = {aid for aid, it in intents.items()
                       if it["dest"] != (self.agents[aid].x, self.agents[aid].y)}
        static_cells = {(a.x, a.y) for a in active if a.id not in moving_ids}
        dest_claims  = Counter(it["dest"] for it in intents.values())

        final_pos = {}
        for a in active:
            it   = intents[a.id]
            dest = it["dest"]
            wp   = it["wp"]
            orig = (a.x, a.y)

            if dest == orig:
                final_pos[a.id] = orig; continue
            if dest_claims[dest] > 1:
                final_pos[a.id] = orig; continue

            if it["is_boost"]:
                if wp in static_cells:
                    final_pos[a.id] = orig          # petak-1 terblokir → diam
                elif dest in static_cells:
                    final_pos[a.id] = wp            # petak-2 terblokir → berhenti di petak-1
                else:
                    final_pos[a.id] = dest          # maju 2 petak
            else:
                final_pos[a.id] = orig if dest in static_cells else dest

        for a in active:
            new = final_pos[a.id]
            if new == (a.x, a.y): continue
            a.x, a.y = new
            if a.carrying:
                a.carrying.x, a.carrying.y = new
            if intents[a.id]["is_boost"]:
                a.boost_stacks -= 1
                a.energy       -= 2
            else:
                a.energy -= 2 if a.carrying else 1
            a.steps_taken += 1

        # ── FASE 3: Interaksi non-gerakan ─────────────────────
        for a in active:
            act       = actions_dict.get(a.id, "IDLE")
            neighbors = self.get_neighbors(a, radius=1)

            if act == "TAKE_FOOD" and len(a.inventory) < 3:
                if self.food_grid[a.x, a.y]:
                    a.inventory.append("food")
                    self.food_grid[a.x, a.y] = False
                    a.food_memory = (a.x, a.y)

            elif act == "EAT_FOOD" and "food" in a.inventory:
                a.inventory.remove("food")
                a.energy = min(50, a.energy + 20)

            if len(neighbors) > 1:
                valid = any(n.is_dormant for n in neighbors) or act == "SHARE_INFO"
                if not valid:
                    pairing_penalty[a.id] -= 5

            for nb in neighbors:
                if act == "GIVE_FOOD" and "food" in a.inventory and nb.energy < 20:
                    a.inventory.remove("food")
                    nb.energy = min(50, nb.energy + 20)
                    break
                elif act == "REVIVE" and nb.is_dormant:
                    nb.is_dormant  = False
                    nb.steps_taken = 0
                    sh = (a.energy + nb.energy) / 2
                    a.energy = nb.energy = sh
                    break
                elif act == "CARRY_AGENT" and a.carrying is None and nb.is_dormant:
                    a.carrying = nb
                    nb.is_being_carried = True
                    break
                elif act == "SHARE_INFO" and a.food_memory:
                    nb.food_memory = a.food_memory

            if act == "DROP_AGENT" and a.carrying is not None:
                dropped = False
                for ddx in [-1, 0, 1]:
                    for ddy in [-1, 0, 1]:
                        if ddx == ddy == 0: continue
                        nx, ny = self._clamp(a.x + ddx, a.y + ddy)
                        if self.get_agent_at(nx, ny) is None:
                            a.carrying.x, a.carrying.y = nx, ny
                            a.release_carrying()
                            dropped = True
                            break
                    if dropped: break

        # ── FASE 4: Cek kematian / dormancy ──────────────────
        for a in active:
            if a.is_dead: continue
            accompanied = len(self.get_neighbors(a, radius=1)) > 0
            if a.energy <= 0:
                a.release_carrying(); a.is_dead = True; dead_this_step.append(a.id)
            elif a.steps_taken >= 49:
                a.release_carrying(); a.is_dormant = True; a.dormant_timer = 0
            elif a.steps_taken >= 40 and not accompanied:
                a.release_carrying(); a.is_dormant = True; a.dormant_timer = 0

        # ── Reward ────────────────────────────────────────────
        total_alive = sum(1 for a in self.agents if not a.is_dead)
        n_deaths    = len(dead_this_step)
        rewards     = {}
        for aid in alive_at_start:
            base    = total_alive * 2
            penalty = pairing_penalty.get(aid, 0)
            rewards[aid] = (base + penalty - 10) if aid in dead_this_step \
                           else (base + penalty - 5 * n_deaths)

        return rewards, dead_this_step


# ══════════════════════════════════════════════════════════════
# SEKSI 4 — NEURAL NETWORK & REPLAY BUFFER
# ══════════════════════════════════════════════════════════════
class MaklukBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE, 128), nn.ReLU(),
            nn.Linear(128, 128),        nn.ReLU(),
            nn.Linear(128, ACTION_SIZE)
        )
    def forward(self, x): return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))

    def sample(self, k):
        s, a, r, ns, d = zip(*random.sample(self.buf, k))
        return (np.array(s, np.float32), np.array(a, np.int64),
                np.array(r, np.float32), np.array(ns, np.float32),
                np.array(d, bool))

    def __len__(self): return len(self.buf)


class AgentBrainManager:
    """
    Manajemen 12 NN terpisah (6 policy + 6 target).
    Setiap agen punya bobot, optimizer, dan memory sendiri.
    """
    def __init__(self, num_agents=NUM_AGENTS, lr=CFG["lr"]):
        self.num_agents  = num_agents
        self.policy_nets = {}
        self.target_nets = {}
        self.optimizers  = {}
        self.memories    = {}

        for i in range(num_agents):
            p = MaklukBrain(); t = MaklukBrain()
            t.load_state_dict(p.state_dict()); t.eval()
            self.policy_nets[i] = p
            self.target_nets[i] = t
            self.optimizers[i]  = optim.Adam(p.parameters(), lr=lr)
            self.memories[i]    = ReplayBuffer()

        params = sum(p.numel() for p in self.policy_nets[0].parameters())
        print(f"\n  NN: {num_agents*2} total  "
              f"({num_agents} policy + {num_agents} target)  "
              f"│  {params:,} param/NN  │  {params*num_agents*2:,} total param\n")

    def sync_all_targets(self):
        for i in range(self.num_agents):
            self.target_nets[i].load_state_dict(self.policy_nets[i].state_dict())


def get_state(agent_id, env):
    a = env.agents[agent_id]
    if a.is_dead:
        return np.zeros(STATE_SIZE, np.float32)
    nb  = env.get_neighbors(a, radius=1)
    sos = env.get_sos_signals(a)
    return np.array([
        a.x / env.grid_size,
        a.y / env.grid_size,
        a.energy / 50.0,
        a.boost_stacks / 5.0,
        len(a.inventory) / 3.0,
        1.0 if nb else 0.0,
        min(len(sos) / 5.0, 1.0),
        agent_id / max(1, env.num_agents - 1),
    ], np.float32)


# ══════════════════════════════════════════════════════════════
# SEKSI 5 — SAVE & LOAD
# ══════════════════════════════════════════════════════════════
def save_brain(brain, episode, epsilon, global_step, stats=None, name=None):
    os.makedirs(CFG["save_dir"], exist_ok=True)
    folder = os.path.join(CFG["save_dir"], name or f"ep{episode:04d}")
    os.makedirs(folder, exist_ok=True)

    for i in range(brain.num_agents):
        torch.save(brain.policy_nets[i].state_dict(), f"{folder}/policy_{i}.pth")
        torch.save(brain.target_nets[i].state_dict(), f"{folder}/target_{i}.pth")
        torch.save(brain.optimizers[i].state_dict(),  f"{folder}/optim_{i}.pth")

    meta = dict(episode=episode, epsilon=epsilon, global_step=global_step,
                num_agents=brain.num_agents,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    with open(f"{folder}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if stats: stats.save(f"{folder}/stats.json")
    print(f"  [SAVE ✓] ep {episode:>4} → {folder}")
    return folder


def load_brain(folder, brain):
    with open(f"{folder}/meta.json") as f:
        meta = json.load(f)
    for i in range(brain.num_agents):
        brain.policy_nets[i].load_state_dict(
            torch.load(f"{folder}/policy_{i}.pth", weights_only=True))
        brain.target_nets[i].load_state_dict(
            torch.load(f"{folder}/target_{i}.pth", weights_only=True))
        brain.optimizers[i].load_state_dict(
            torch.load(f"{folder}/optim_{i}.pth",  weights_only=True))
    print(f"  [LOAD ✓] ep {meta['episode']}  ε {meta['epsilon']:.3f}  "
          f"step {meta['global_step']}  {meta['timestamp']}")
    return brain, meta


def get_latest_checkpoint():
    sd = CFG["save_dir"]
    if not os.path.exists(sd): return None
    folders = [d for d in os.listdir(sd)
               if os.path.exists(os.path.join(sd, d, "meta.json"))]
    if not folders: return None
    folders.sort(key=lambda d: json.load(
        open(os.path.join(sd, d, "meta.json")))["episode"])
    return os.path.join(sd, folders[-1])


def list_checkpoints():
    sd = CFG["save_dir"]
    if not os.path.exists(sd):
        print("Belum ada checkpoint."); return
    rows = []
    for d in sorted(os.listdir(sd)):
        mp = os.path.join(sd, d, "meta.json")
        if os.path.exists(mp):
            m = json.load(open(mp))
            rows.append((d, m))
    if not rows: print("Belum ada checkpoint."); return
    print(f"\n  {'Folder':28s} {'Episode':>8}  {'Epsilon':>8}  Waktu")
    print(f"  {'─'*60}")
    for name, m in rows:
        print(f"  {name:28s} {m['episode']:>8}  {m['epsilon']:>8.3f}  {m['timestamp']}")
    print()


# ══════════════════════════════════════════════════════════════
# SEKSI 6 — STATS TRACKER
# ══════════════════════════════════════════════════════════════
class StatsTracker:
    _keys = ["episodes","rewards","alive","dormant","deaths","steps","epsilon"]

    def __init__(self):
        self._d = {k: [] for k in self._keys}

    def log(self, episode, reward, alive, dormant, deaths, steps, epsilon):
        for k, v in zip(self._keys,
                        [episode, reward, alive, dormant, deaths, steps, epsilon]):
            self._d[k].append(v)

    def __len__(self): return len(self._d["episodes"])

    def save(self, path):
        with open(path, "w") as f: json.dump(self._d, f)

    def load(self, path):
        with open(path) as f: self._d = json.load(f)

    @staticmethod
    def _roll(data, w=20):
        return [float(np.mean(data[max(0, i-w+1):i+1])) for i in range(len(data))]

    def summary(self, n=20):
        n = min(n, len(self))
        if n == 0: return
        r = self._d["rewards"][-n:]
        a = self._d["alive"][-n:]
        d = self._d["deaths"][-n:]
        print(f"\n  ┌─ RINGKASAN {n} EPISODE TERAKHIR ───────────────┐")
        print(f"  │  Reward rata-rata : {np.mean(r):>10.1f}              │")
        print(f"  │  Reward maks      : {np.max(r):>10.1f}              │")
        print(f"  │  Agen selamat avg : {np.mean(a):>10.2f} / {NUM_AGENTS}         │")
        print(f"  │  Kematian avg     : {np.mean(d):>10.2f}              │")
        print(f"  └────────────────────────────────────────────────┘\n")

    def plot(self, save_path=None, show=True, w=20):
        if not MPL_OK:
            print("[PLOT] matplotlib tidak terinstall."); return
        if len(self) == 0:
            print("[PLOT] Belum ada data."); return

        BG, CELL, GRID = "#0d0f1a", "#12152a", "#1a1e35"
        eps = self._d["episodes"]

        fig = plt.figure(figsize=(17, 10), facecolor=BG)
        fig.suptitle("MaklukGabut — Training Statistics",
                     color="#96c8ff", fontsize=15, fontweight="bold", y=0.99)
        gs = gridspec.GridSpec(2, 3, fig, hspace=0.48, wspace=0.33,
                               left=0.06, right=0.97, top=0.93, bottom=0.09)

    
