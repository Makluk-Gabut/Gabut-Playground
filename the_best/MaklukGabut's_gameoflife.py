"""
MaklukGabut's Game of Life  —  ALL-IN-ONE  + COMPANION SYSTEM
                  final+companion

  Satu file berisi segalanya:
    • Game engine   (agen + lingkungan 4-fase atomik)
    • Companion     (agen pendamping, NN ~4070 param, shared memory)
    • RL engine     (12+2 NN, Double DQN, Replay Buffer)
    • Save / Load   (checkpoint otomatis, termasuk companion)
    • Statistik     (catat + grafik matplotlib)
    • Visualizer    (real-time pygame, companion ditampilkan sebagai ♦)
    • Keribetan     (676767676767676767 out of 10)

  Cara pakai:
    python makluk_gabut.py train            ← mulai baru
    python makluk_gabut.py train --resume   ← lanjut
    python makluk_gabut.py train --visual   ← visualisasi
    python makluk_gabut.py demo             ← tonton AI
    python makluk_gabut.py plot             ← lihat grafik
    python makluk_gabut.py list             ← daftar checkpoint

  ═══════════ CATATAN COMPANION(fitur baru) ═══════════
    • Max 3 companion di map sekaligus
    • Muncul jika 3+ agen utama di 1 sel selama 40 step berturut-turut
    • Stamina 100, TIDAK ada fase dorman — langsung mati saat stamina ≤ 0
    • Inventory 5 slot, bisa carry 2 agen dorman sekaligus
    • Semua companion berbagi 1 policy NN + 1 shared replay buffer
    • Reward: +15 revive, +10 kasih makanan, +5 selamat sampai akhir
    • Penalti: -10 gagal revive, -10 mati sebelum selesai, -5 tidak kasih makan
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

VIZ = dict(
    cell        = 52,
    panel_w     = 300,
    default_fps = 10,
)

# ── Konstanta Companion ───────────────────────────────────────
COMPANION_STATE_SIZE  = 10    # dimensi state companion
COMPANION_ACTION_SIZE = 14    # jumlah aksi companion
MAX_COMPANIONS        = 3     # maks companion aktif di map
COMPANION_SPAWN_STEPS = 40    # step 3 agen di 1 sel → spawn companion

COMPANION_ACTION_MAP = [
    "IDLE",                                                     # 0
    "MOVE_UP",  "MOVE_DOWN",  "MOVE_LEFT",  "MOVE_RIGHT",      # 1-4
    "BOOST_UP", "BOOST_DOWN", "BOOST_LEFT", "BOOST_RIGHT",     # 5-8
    "TAKE_FOOD",   # 9  — ambil makanan di sel ini
    "GIVE_FOOD",   # 10 — kasih makanan ke agen utama radius 1
    "REVIVE",      # 11 — pulihkan agen dorman radius 1
    "CARRY",       # 12 — angkat agen dorman radius 1 (maks 2 sekaligus)
    "DROP",        # 13 — taruh agen yang sedang dibawa
]


# ══════════════════════════════════════════════════════════════
# SEKSI 2 — KELAS AGEN UTAMA
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
# SEKSI 3 — KELAS COMPANION
# ══════════════════════════════════════════════════════════════
class Companion:
    """
    Makluk pendamping — TIDAK ada fase dorman.
    Langsung mati saat stamina ≤ 0.
    Inventory 5 slot, bisa carry 2 agen utama dorman sekaligus.
    """
    def __init__(self, companion_id, x, y):
        self.id              = companion_id
        self.x, self.y      = x, y
        self.stamina        = 100        # hp, langsung mati kalau 0
        self.is_dead        = False
        self.inventory      = []         # ["food", ...] max 5
        self.carrying       = []         # [MaklukGabut, ...] max 2
        self.boost_charge   = 0
        self.boost_stacks   = 0
        self.steps_taken    = 0
        self.food_memory    = None
        # Statistik episode (untuk reward akhir)
        self.successful_revives  = 0
        self.food_delivered      = 0
        self.failed_revive_tries = 0

    def release_all(self):
        """Lepas semua agen yang sedang dibawa."""
        for carried in list(self.carrying):
            if hasattr(carried, "is_being_carried"):
                carried.is_being_carried = False
        self.carrying.clear()


# ══════════════════════════════════════════════════════════════
# SEKSI 4 — LINGKUNGAN (4-FASE ATOMIK + COMPANION SUPPORT)
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

        # ── State companion ───────────────────────────────────
        self.companions         = []          # list of active Companion
        self.next_companion_id  = 0
        self._cluster_timer     = {}          # {(x,y): step_count}

    # ── Helpers ──────────────────────────────────────────────
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

    # ── Spawn Companion ───────────────────────────────────────
    def _check_companion_spawn(self):
        """
        Spawn companion jika 3+ agen utama di 1 sel selama
        COMPANION_SPAWN_STEPS berturut-turut, dan jumlah companion < MAX.
        """
        alive = [a for a in self.agents
                 if not a.is_dead and not a.is_being_carried]
        cell_cnt    = Counter((a.x, a.y) for a in alive)
        cluster_now = {cell for cell, n in cell_cnt.items() if n >= 3}

        # Hapus sel yang sudah tidak cluster
        for cell in list(self._cluster_timer):
            if cell not in cluster_now:
                del self._cluster_timer[cell]

        # Update timer sel cluster aktif
        for cell in cluster_now:
            self._cluster_timer[cell] = self._cluster_timer.get(cell, 0) + 1

        # Spawn jika memenuhi syarat
        for cell, timer in list(self._cluster_timer.items()):
            if timer >= COMPANION_SPAWN_STEPS \
                    and len(self.companions) < MAX_COMPANIONS:
                cx = np.random.randint(0, self.grid_size)
                cy = np.random.randint(0, self.grid_size)
                c  = Companion(self.next_companion_id, cx, cy)
                self.next_companion_id += 1
                self.companions.append(c)
                self._cluster_timer[cell] = 0   # reset agar tidak spam spawn
                print(f"  ★  COMPANION #{c.id} muncul di ({cx},{cy})"
                      f"  │  total: {len(self.companions)}")

    # ── Step Companion ────────────────────────────────────────
    def step_companions(self, companion_actions):
        """
        Proses semua companion dalam 1 step.
        companion_actions = {companion_id: action_string}
        Returns: {companion_id: reward_langkah}
        """
        rewards      = {}
        active_comp  = [c for c in self.companions if not c.is_dead]
        if not active_comp:
            return rewards

        # ── Boost charge ──────────────────────────────────────
        for c in active_comp:
            if c.boost_stacks < 5:
                c.boost_charge += 1
                if c.boost_charge >= 10:
                    c.boost_stacks += 1
                    c.boost_charge = 0

        # ── FASE 1: Deklarasi niat gerak ──────────────────────
        intents = {}
        for c in active_comp:
            act = companion_actions.get(c.id, "IDLE")
            if act.startswith("MOVE_"):
                dx, dy = self._dir(act)
                intents[c.id] = dict(dest=self._clamp(c.x+dx, c.y+dy),
                                     is_boost=False)
            elif act.startswith("BOOST_") and c.boost_stacks > 0:
                dx, dy = self._dir(act)
                intents[c.id] = dict(dest=self._clamp(c.x+dx*2, c.y+dy*2),
                                     is_boost=True)
            else:
                intents[c.id] = dict(dest=(c.x, c.y), is_boost=False)

        # ── FASE 2: Resolusi konflik antar companion ──────────
        dest_claims = Counter(v["dest"] for v in intents.values())
        for c in active_comp:
            it   = intents[c.id]
            dest = it["dest"]
            orig = (c.x, c.y)
            if dest == orig:
                continue
            if dest_claims[dest] > 1:
                intents[c.id]["dest"] = orig   # konflik → diam
                continue
            # Gerak sah
            c.x, c.y = dest
            for carried in c.carrying:        # bawa ikut bergerak
                carried.x, carried.y = dest
            cost = (3 if it["is_boost"] else 1) + len(c.carrying)
            c.stamina -= cost
            if it["is_boost"]:
                c.boost_stacks -= 1
            c.steps_taken += 1

        # ── FASE 3: Interaksi ─────────────────────────────────
        for c in active_comp:
            if c.is_dead:
                continue
            act = companion_actions.get(c.id, "IDLE")
            r   = -0.1   # biaya kecil per step → dorong efisiensi

            # —— TAKE_FOOD ——————————————————————————————————
            if act == "TAKE_FOOD" and len(c.inventory) < 5:
                if self.food_grid[c.x, c.y]:
                    c.inventory.append("food")
                    self.food_grid[c.x, c.y] = False
                    c.food_memory = (c.x, c.y)

            # —— GIVE_FOOD ke agen utama ————————————————————
            elif act == "GIVE_FOOD" and "food" in c.inventory:
                targets = [
                    a for a in self.agents
                    if not a.is_dead
                    and abs(a.x - c.x) <= 1
                    and abs(a.y - c.y) <= 1
                ]
                if targets:
                    target = min(targets, key=lambda a: a.energy)
                    c.inventory.remove("food")
                    target.energy     = min(50, target.energy + 20)
                    c.food_delivered += 1
                    r += 10.0
                else:
                    r -= 2.0    # coba kasih tapi tidak ada target

            # —— REVIVE agen utama dorman ———————————————————
            elif act == "REVIVE":
                dormant_nb = [
                    a for a in self.agents
                    if a.is_dormant
                    and abs(a.x - c.x) <= 1
                    and abs(a.y - c.y) <= 1
                ]
                if dormant_nb:
                    target               = dormant_nb[0]
                    target.is_dormant    = False
                    target.dormant_timer = 0
                    target.steps_taken   = 0
                    c.successful_revives += 1
                    c.stamina            -= 15
                    r += 15.0
                else:
                    c.failed_revive_tries += 1
                    r -= 10.0   # penalti gagal revive

            # —— CARRY agen dorman (maks 2) ————————————————
            elif act == "CARRY" and len(c.carrying) < 2:
                candidates = [
                    a for a in self.agents
                    if a.is_dormant
                    and not a.is_being_carried
                    and abs(a.x - c.x) <= 1
                    and abs(a.y - c.y) <= 1
                ]
                if candidates:
                    target                   = candidates[0]
                    target.is_being_carried  = True
                    c.carrying.append(target)

            # —— DROP agen yang dibawa —————————————————————
            elif act == "DROP" and c.carrying:
                dropped = False
                for ddx in [-1, 0, 1]:
                    for ddy in [-1, 0, 1]:
                        if ddx == ddy == 0: continue
                        nx, ny = self._clamp(c.x + ddx, c.y + ddy)
                        if self.get_agent_at(nx, ny) is None:
                            target              = c.carrying.pop()
                            target.x, target.y  = nx, ny
                            target.is_being_carried = False
                            dropped = True
                            break
                    if dropped: break

            # Drain stamina pasif per step
            c.stamina -= 0.5
            rewards[c.id] = r

        # ── FASE 4: Cek kematian companion ───────────────────
        for c in active_comp:
            if not c.is_dead and c.stamina <= 0:
                c.release_all()
                c.is_dead     = True
                rewards[c.id] = rewards.get(c.id, 0) - 10.0

        return rewards

    def end_of_episode_companion_rewards(self):
        """
        Reward / penalti akhir episode untuk setiap companion.
        Dipanggil setelah loop episode selesai.
        """
        end_r = {}
        for c in self.companions:
            r = 0.0
            if not c.is_dead:
                r += 5.0    # selamat sampai akhir episode
            if c.food_delivered == 0:
                r -= 5.0    # tidak pernah kasih makanan ke agen utama
            end_r[c.id] = r
        return end_r

    # ── Step Utama (Agen Utama) ───────────────────────────────
    def step(self, actions_dict):
        self.step_count += 1
        if self.step_count % 20 == 0:
            self._spawn_food(3)

        alive_at_start  = {a.id for a in self.agents if not a.is_dead}
        pairing_penalty = {aid: 0 for aid in alive_at_start}
        dead_this_step  = []

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
        moving_ids   = {aid for aid, it in intents.items()
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
                    final_pos[a.id] = orig
                elif dest in static_cells:
                    final_pos[a.id] = wp
                else:
                    final_pos[a.id] = dest
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

        # Cek spawn companion setiap step
        self._check_companion_spawn()

        return rewards, dead_this_step


# ══════════════════════════════════════════════════════════════
# SEKSI 5 — NEURAL NETWORK AGEN UTAMA & REPLAY BUFFER
# ══════════════════════════════════════════════════════════════
class MaklukBrain(nn.Module):
    """NN agen utama — 3 layer, ~19.728 param."""
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
        return (np.array(s,  np.float32), np.array(a, np.int64),
                np.array(r,  np.float32), np.array(ns, np.float32),
                np.array(d,  bool))

    def __len__(self): return len(self.buf)


class AgentBrainManager:
    """
    6 policy + 6 target — masing-masing agen punya bobot & memori sendiri.
    Double DQN.
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
        print(f"\n  [Main Brain]  {num_agents*2} NN total"
              f"  │  {params:,} param/NN  │  {params*num_agents*2:,} total\n")

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
# SEKSI 6 — COMPANION NEURAL NETWORK (~4.070 param)
# ══════════════════════════════════════════════════════════════
class CompanionBrain(nn.Module):
    """
    3-layer MLP kecil — khusus companion.
    Linear(10→52) → ReLU → Linear(52→52) → ReLU → Linear(52→14)
    Total param:  572 + 2756 + 742 = 4.070
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(COMPANION_STATE_SIZE, 52), nn.ReLU(),
            nn.Linear(52, 52),                   nn.ReLU(),
            nn.Linear(52, COMPANION_ACTION_SIZE),
        )

    def forward(self, x):
        return self.net(x)


class CompanionBrainManager:
    """
    Semua companion berbagi 1 policy + 1 target + 1 replay buffer.
    ← ini yang membedakannya dari agen utama (yang masing-masing punya memori sendiri).
    Double DQN.
    """
    def __init__(self, lr=CFG["lr"]):
        self.policy_net = CompanionBrain()
        self.target_net = CompanionBrain()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer  = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory     = ReplayBuffer(capacity=20_000)   # shared memory

        n = sum(p.numel() for p in self.policy_net.parameters())
        print(f"  [Companion Brain]  1 policy + 1 target (shared)"
              f"  │  {n:,} param/NN  │  {n*2:,} total\n")

    def sync_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(COMPANION_ACTION_SIZE)
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0)
            return int(self.policy_net(t).argmax(dim=1).item())

    def train_step(self, batch_size, gamma):
        if len(self.memory) < batch_size:
            return None
        s, a, r, ns, d = self.memory.sample(batch_size)
        s   = torch.FloatTensor(s)
        a   = torch.LongTensor(a).unsqueeze(1)
        r   = torch.FloatTensor(r).unsqueeze(1)
        ns  = torch.FloatTensor(ns)
        d   = torch.BoolTensor(d).unsqueeze(1)

        q_curr = self.policy_net(s).gather(1, a)
        with torch.no_grad():
            best_a    = self.policy_net(ns).argmax(1, keepdim=True)
            q_next    = self.target_net(ns).gather(1, best_a)
            q_next[d] = 0.0
            q_target  = r + gamma * q_next
        loss = nn.functional.mse_loss(q_curr, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, folder):
        torch.save(self.policy_net.state_dict(), f"{folder}/c_policy.pth")
        torch.save(self.target_net.state_dict(), f"{folder}/c_target.pth")
        torch.save(self.optimizer.state_dict(),  f"{folder}/c_optim.pth")

    def load(self, folder):
        self.policy_net.load_state_dict(
            torch.load(f"{folder}/c_policy.pth", weights_only=True))
        self.target_net.load_state_dict(
            torch.load(f"{folder}/c_target.pth", weights_only=True))
        self.optimizer.load_state_dict(
            torch.load(f"{folder}/c_optim.pth",  weights_only=True))
        print("  [LOAD ✓] CompanionBrain loaded")


def get_companion_state(companion, env):
    """
    10-dim state vector companion.
    ─────────────────────────────────────
    0  x / grid_size
    1  y / grid_size
    2  stamina / 100
    3  isi inventory / 5
    4  jumlah yang dibawa / 2
    5  jarak L1 ke agen hidup terdekat (norm)
    6  jarak L1 ke agen dorman terdekat (norm)
    7  jarak L1 ke makanan terdekat (norm)
    8  punya makanan? (0/1)
    9  companion_id / (MAX_COMPANIONS-1)
    """
    c     = companion
    max_d = float(env.grid_size * 2)

    if c.is_dead:
        return np.zeros(COMPANION_STATE_SIZE, np.float32)

    alive_agents   = [a for a in env.agents if not a.is_dead]
    dormant_agents = [a for a in env.agents if a.is_dormant]

    def min_l1(entities):
        if not entities:
            return 1.0
        return min(abs(e.x - c.x) + abs(e.y - c.y) for e in entities) / max_d

    food_pos = np.argwhere(env.food_grid)
    if len(food_pos) > 0:
        food_d = float(
            np.abs(food_pos[:, 0] - c.x).min() +
            np.abs(food_pos[:, 1] - c.y).min()
        ) / max_d
    else:
        food_d = 1.0

    return np.array([
        c.x / env.grid_size,
        c.y / env.grid_size,
        c.stamina / 100.0,
        len(c.inventory) / 5.0,
        len(c.carrying) / 2.0,
        min(min_l1(alive_agents),   1.0),
        min(min_l1(dormant_agents), 1.0),
        min(food_d,                 1.0),
        1.0 if "food" in c.inventory else 0.0,
        c.id / max(1, MAX_COMPANIONS - 1),
    ], np.float32)


# ══════════════════════════════════════════════════════════════
# SEKSI 7 — SAVE & LOAD (termasuk companion)
# ══════════════════════════════════════════════════════════════
def save_brain(brain, companion_mgr, episode, epsilon, global_step,
               stats=None, name=None):
    os.makedirs(CFG["save_dir"], exist_ok=True)
    folder = os.path.join(CFG["save_dir"], name or f"ep{episode:04d}")
    os.makedirs(folder, exist_ok=True)

    for i in range(brain.num_agents):
        torch.save(brain.policy_nets[i].state_dict(), f"{folder}/policy_{i}.pth")
        torch.save(brain.target_nets[i].state_dict(), f"{folder}/target_{i}.pth")
        torch.save(brain.optimizers[i].state_dict(),  f"{folder}/optim_{i}.pth")

    companion_mgr.save(folder)

    meta = dict(episode=episode, epsilon=epsilon, global_step=global_step,
                num_agents=brain.num_agents,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    with open(f"{folder}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if stats:
        stats.save(f"{folder}/stats.json")
    print(f"  [SAVE ✓] ep {episode:>4} → {folder}")
    return folder


def load_brain(folder, brain, companion_mgr):
    with open(f"{folder}/meta.json") as f:
        meta = json.load(f)
    for i in range(brain.num_agents):
        brain.policy_nets[i].load_state_dict(
            torch.load(f"{folder}/policy_{i}.pth", weights_only=True))
        brain.target_nets[i].load_state_dict(
            torch.load(f"{folder}/target_{i}.pth", weights_only=True))
        brain.optimizers[i].load_state_dict(
            torch.load(f"{folder}/optim_{i}.pth",  weights_only=True))
    try:
        companion_mgr.load(folder)
    except FileNotFoundError:
        print("  [WARN] Companion checkpoint tidak ada — pakai brain baru.")
    print(f"  [LOAD ✓] ep {meta['episode']}  ε {meta['epsilon']:.3f}"
          f"  step {meta['global_step']}  {meta['timestamp']}")
    return brain, companion_mgr, meta


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
        print(f"  {name:28s} {m['episode']:>8}  {m['epsilon']:>8.3f}"
              f"  {m['timestamp']}")
    print()


# ══════════════════════════════════════════════════════════════
# SEKSI 8 — STATS TRACKER
# ══════════════════════════════════════════════════════════════
class StatsTracker:
    _keys = ["episodes", "rewards", "alive", "dormant",
             "deaths", "steps", "epsilon", "companions"]

    def __init__(self):
        self._d = {k: [] for k in self._keys}

    def log(self, episode, reward, alive, dormant, deaths, steps, epsilon, companions=0):
        vals = [episode, reward, alive, dormant, deaths, steps, epsilon, companions]
        for k, v in zip(self._keys, vals):
            self._d[k].append(v)

    def __len__(self): return len(self._d["episodes"])

    def save(self, path):
        with open(path, "w") as f: json.dump(self._d, f)

    def load(self, path):
        with open(path) as f: self._d = json.load(f)
        # Kompatibilitas dengan versi lama (tanpa 'companions')
        if "companions" not in self._d:
            self._d["companions"] = [0] * len(self._d["episodes"])

    @staticmethod
    def _roll(data, w=20):
        return [float(np.mean(data[max(0, i-w+1):i+1])) for i in range(len(data))]

    def summary(self, n=20):
        n = min(n, len(self))
        if n == 0: return
        r  = self._d["rewards"][-n:]
        a  = self._d["alive"][-n:]
        d  = self._d["deaths"][-n:]
        c  = self._d["companions"][-n:]
        print(f"\n  ┌─ RINGKASAN {n} EPISODE TERAKHIR ─────────────────┐")
        print(f"  │  Reward rata-rata   : {np.mean(r):>10.1f}               │")
        print(f"  │  Reward maks        : {np.max(r):>10.1f}               │")
        print(f"  │  Agen selamat avg   : {np.mean(a):>10.2f} / {NUM_AGENTS}          │")
        print(f"  │  Kematian avg       : {np.mean(d):>10.2f}               │")
        print(f"  │  Companion avg/ep   : {np.mean(c):>10.2f}               │")
        print(f"  └──────────────────────────────────────────────────┘\n")

    def plot(self, save_path=None, show=True, w=20):
        if not MPL_OK:
            print("[PLOT] matplotlib tidak terinstall."); return
        if len(self) == 0:
            print("[PLOT] Belum ada data."); return

        BG   = "#0d0f1a"
        CELL = "#12152a"
        GRID = "#1a1e35"
        eps  = self._d["episodes"]

        fig = plt.figure(figsize=(18, 11), facecolor=BG)
        fig.suptitle("MaklukGabut — Training Statistics",
                     color="#96c8ff", fontsize=15, fontweight="bold", y=0.99)
        gs = gridspec.GridSpec(2, 3, fig, hspace=0.48, wspace=0.33,
                               left=0.06, right=0.97, top=0.93, bottom=0.09)

        def make_ax(pos):
            ax = fig.add_subplot(gs[pos])
            ax.set_facecolor(CELL)
            ax.tick_params(colors="#aac4ff", labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID)
            return ax

        def plot_line(ax, data, label, color, alpha_raw=0.25):
            rolled = self._roll(data, w)
            ax.plot(eps, data,   color=color, alpha=alpha_raw, linewidth=0.8)
            ax.plot(eps, rolled, color=color, linewidth=1.8, label=label)
            ax.legend(fontsize=8, facecolor=CELL, labelcolor="#dde8ff")
            ax.set_xlabel("Episode", color="#7090cc", fontsize=8)
            ax.grid(color=GRID, linewidth=0.5, alpha=0.7)

        ax0 = make_ax((0, 0))
        ax0.set_title("Reward Total", color="#96c8ff", fontsize=10)
        plot_line(ax0, self._d["rewards"], "reward", "#4fc3f7")

        ax1 = make_ax((0, 1))
        ax1.set_title("Agen Selamat", color="#96c8ff", fontsize=10)
        plot_line(ax1, self._d["alive"], "alive", "#81c784")

        ax2 = make_ax((0, 2))
        ax2.set_title("Kematian per Episode", color="#96c8ff", fontsize=10)
        plot_line(ax2, self._d["deaths"], "deaths", "#e57373")

        ax3 = make_ax((1, 0))
        ax3.set_title("Agen Dorman", color="#96c8ff", fontsize=10)
        plot_line(ax3, self._d["dormant"], "dormant", "#ba68c8")

        ax4 = make_ax((1, 1))
        ax4.set_title("Companion Aktif / Episode", color="#96c8ff", fontsize=10)
        plot_line(ax4, self._d["companions"], "companion", "#ffd54f")

        ax5 = make_ax((1, 2))
        ax5.set_title("Epsilon (Eksplorasi)", color="#96c8ff", fontsize=10)
        plot_line(ax5, self._d["epsilon"], "ε", "#f06292")

        if save_path:
            fig.savefig(save_path, dpi=110, bbox_inches="tight", facecolor=BG)
            print(f"  [PLOT ✓] Tersimpan → {save_path}")
        if show:
            plt.show()
        plt.close(fig)


# ══════════════════════════════════════════════════════════════
# SEKSI 9 — VISUALIZER (Pygame, termasuk rendering companion)
# ══════════════════════════════════════════════════════════════
class Visualizer:
    # Warna per agen utama
    AGENT_COLORS = [
        (64, 160, 255), (255, 160, 64), (64, 255, 160),
        (255, 64, 160), (160, 255, 64), (160, 64, 255),
    ]
    COMPANION_COLOR = (255, 215, 50)   # emas
    DORMANT_COLOR   = (100, 100, 160)
    FOOD_COLOR      = (80,  200,  80)

    def __init__(self):
        pygame.init()
        cell    = VIZ["cell"]
        panel_w = VIZ["panel_w"]
        w = GRID_SIZE * cell + panel_w
        h = GRID_SIZE * cell
        self.screen  = pygame.display.set_mode((w, h))
        pygame.display.set_caption("MaklukGabut Simulator")
        self.clock   = pygame.time.Clock()
        self.fps     = VIZ["default_fps"]
        self.cell    = cell
        self.font_s  = pygame.font.SysFont("monospace", 11)
        self.font_m  = pygame.font.SysFont("monospace", 13, bold=True)

    def render(self, env, ep, step_n, epsilon, total_rew):
        cell   = self.cell
        gw     = GRID_SIZE * cell
        screen = self.screen

        screen.fill((13, 15, 26))

        # ── Grid lines ──────────────────────────────────────
        for i in range(GRID_SIZE + 1):
            pygame.draw.line(screen, (26, 30, 53), (i*cell, 0), (i*cell, gw))
            pygame.draw.line(screen, (26, 30, 53), (0, i*cell), (gw, i*cell))

        # ── Makanan ──────────────────────────────────────────
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if env.food_grid[x, y]:
                    r = pygame.Rect(x*cell+6, y*cell+6, cell-12, cell-12)
                    pygame.draw.rect(screen, self.FOOD_COLOR, r, border_radius=3)

        # ── Agen utama ────────────────────────────────────────
        for a in env.agents:
            if a.is_dead: continue
            col = self.DORMANT_COLOR if a.is_dormant \
                  else self.AGENT_COLORS[a.id % len(self.AGENT_COLORS)]
            cx = a.x * cell + cell // 2
            cy = a.y * cell + cell // 2
            pygame.draw.circle(screen, col, (cx, cy), cell // 2 - 4)
            lbl = self.font_s.render(str(a.id), True, (0, 0, 0))
            screen.blit(lbl, (cx - lbl.get_width()//2, cy - lbl.get_height()//2))
            # Energy bar (bawah)
            bx = a.x * cell + 4
            by = a.y * cell + cell - 7
            bw = cell - 8
            pygame.draw.rect(screen, (50, 50, 50), pygame.Rect(bx, by, bw, 4))
            fill = int(bw * a.energy / 50)
            if fill > 0:
                ec = (80, 200, 80) if a.energy > 20 else (220, 60, 60)
                pygame.draw.rect(screen, ec, pygame.Rect(bx, by, fill, 4))

        # ── Companion (ditampilkan sebagai ♦) ─────────────────
        for c in env.companions:
            if c.is_dead: continue
            cx = c.x * cell + cell // 2
            cy = c.y * cell + cell // 2
            half = cell // 2 - 4
            pts  = [(cx, cy - half), (cx + half, cy),
                    (cx, cy + half), (cx - half, cy)]
            pygame.draw.polygon(screen, self.COMPANION_COLOR, pts)
            pygame.draw.polygon(screen, (180, 140, 0), pts, 2)
            lbl = self.font_s.render(f"C{c.id}", True, (0, 0, 0))
            screen.blit(lbl, (cx - lbl.get_width()//2, cy - lbl.get_height()//2))
            # Stamina bar (atas)
            bx = c.x * cell + 4
            by = c.y * cell + 3
            bw = cell - 8
            pygame.draw.rect(screen, (50, 50, 50), pygame.Rect(bx, by, bw, 4))
            fill = int(bw * c.stamina / 100)
            if fill > 0:
                pygame.draw.rect(screen, self.COMPANION_COLOR,
                                 pygame.Rect(bx, by, fill, 4))

        # ── Panel kanan ───────────────────────────────────────
        px = gw
        pw = VIZ["panel_w"]
        ph = GRID_SIZE * cell
        pygame.draw.rect(screen, (18, 21, 42), pygame.Rect(px, 0, pw, ph))

        def txt(s, x, y, col=(200, 220, 255)):
            screen.blit(self.font_s.render(s, True, col), (x, y))

        y0 = 8
        txt(f"Episode   {ep}",         px+8, y0);               y0 += 16
        txt(f"Step      {step_n}",      px+8, y0);               y0 += 16
        txt(f"Epsilon   {epsilon:.3f}", px+8, y0);               y0 += 16
        txt(f"Reward    {total_rew:.1f}", px+8, y0);             y0 += 20
        txt("─── Agen Utama ───", px+8, y0, (100, 140, 220));   y0 += 16
        for a in env.agents:
            st  = "MATI" if a.is_dead else ("DORMAN" if a.is_dormant else "AKTIF")
            col = (140, 60, 60) if a.is_dead else \
                  ((120, 120, 180) if a.is_dormant else (200, 220, 255))
            txt(f"A{a.id} [{st}] E:{int(a.energy):>2} s:{a.steps_taken:>2}"
                f" inv:{len(a.inventory)}", px+8, y0, col)
            y0 += 15
        y0 += 4
        txt("─── Companion ───", px+8, y0, (255, 200, 50));     y0 += 16
        if env.companions:
            for c in env.companions:
                st  = "MATI" if c.is_dead else "AKTIF"
                col = (140, 60, 60) if c.is_dead else (255, 220, 80)
                txt(f"C{c.id} [{st}] ST:{int(c.stamina):>3}"
                    f" inv:{len(c.inventory)} carry:{len(c.carrying)}"
                    f" rev:{c.successful_revives}",
                    px+8, y0, col)
                y0 += 15
        else:
            txt("  (belum muncul)", px+8, y0, (100, 100, 120))
            y0 += 15

        # Hint kontrol
        txt("↑↓ FPS  Q keluar", px+8, ph - 20, (80, 80, 120))

        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    def handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_q:
                    return False
                elif ev.key == pygame.K_UP:
                    self.fps = min(60, self.fps + 5)
                elif ev.key == pygame.K_DOWN:
                    self.fps = max(1,  self.fps - 5)
        return True

    def close(self):
        pygame.quit()


# ══════════════════════════════════════════════════════════════
# SEKSI 10 — TRAINING UTILITIES
# ══════════════════════════════════════════════════════════════
def _select_action(policy_net, state, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        t = torch.FloatTensor(state).unsqueeze(0)
        return int(policy_net(t).argmax(dim=1).item())


def _train_agent_step(brain, agent_id, batch_size, gamma):
    mem = brain.memories[agent_id]
    if len(mem) < batch_size:
        return None
    s, a, r, ns, d = mem.sample(batch_size)
    s   = torch.FloatTensor(s)
    a   = torch.LongTensor(a).unsqueeze(1)
    r   = torch.FloatTensor(r).unsqueeze(1)
    ns  = torch.FloatTensor(ns)
    d   = torch.BoolTensor(d).unsqueeze(1)

    q_curr = brain.policy_nets[agent_id](s).gather(1, a)
    with torch.no_grad():
        best_a    = brain.policy_nets[agent_id](ns).argmax(1, keepdim=True)
        q_next    = brain.target_nets[agent_id](ns).gather(1, best_a)
        q_next[d] = 0.0
        q_target  = r + gamma * q_next
    loss = nn.functional.mse_loss(q_curr, q_target)
    brain.optimizers[agent_id].zero_grad()
    loss.backward()
    brain.optimizers[agent_id].step()
    return loss.item()


# ══════════════════════════════════════════════════════════════
# SEKSI 11 — TRAINING LOOP
# ══════════════════════════════════════════════════════════════
def train(resume=False, visual=False):
    brain         = AgentBrainManager()
    companion_mgr = CompanionBrainManager()
    stats         = StatsTracker()

    epsilon     = CFG["epsilon_start"]
    start_ep    = 0
    global_step = 0

    if resume:
        ckpt = get_latest_checkpoint()
        if ckpt:
            brain, companion_mgr, meta = load_brain(ckpt, brain, companion_mgr)
            epsilon     = meta["epsilon"]
            start_ep    = meta["episode"]
            global_step = meta["global_step"]
            sp = os.path.join(ckpt, "stats.json")
            if os.path.exists(sp):
                stats.load(sp)
        else:
            print("  Tidak ada checkpoint — mulai dari awal.")

    viz = None
    if visual:
        if PYGAME_OK:
            viz = Visualizer()
        else:
            print("  [WARN] pygame tidak terinstall — visual dinonaktifkan.")

    print(f"\n  Mulai training: {CFG['episodes']} episode"
          f"  │  max_step {CFG['max_steps']}"
          f"  │  start_ep {start_ep}"
          f"  │  ε {epsilon:.3f}\n")

    for ep in range(start_ep, CFG["episodes"]):
        env    = MaklukGabutEnv()
        states = {i: get_state(i, env) for i in range(NUM_AGENTS)}

        total_reward = 0.0
        ep_deaths    = 0

        for step_n in range(CFG["max_steps"]):
            global_step += 1

            # ── Aksi agen utama ───────────────────────────────
            actions_idx = {}
            actions_str = {}
            for i, a in enumerate(env.agents):
                if a.is_dead: continue
                idx = _select_action(brain.policy_nets[i], states[i],
                                     epsilon, ACTION_SIZE)
                actions_idx[i] = idx
                actions_str[i] = ACTION_MAP[idx]

            # ── Aksi companion ────────────────────────────────
            comp_states      = {c.id: get_companion_state(c, env)
                                for c in env.companions if not c.is_dead}
            comp_actions_idx = {}
            comp_actions_str = {}
            for cid, cs in comp_states.items():
                idx = companion_mgr.select_action(cs, epsilon)
                comp_actions_idx[cid] = idx
                comp_actions_str[cid] = COMPANION_ACTION_MAP[idx]

            # ── Step lingkungan ───────────────────────────────
            rewards, dead_this = env.step(actions_str)          # agen utama
            comp_rewards = env.step_companions(comp_actions_str) # companion
            # (env.step sudah memanggil _check_companion_spawn)

            ep_deaths += len(dead_this)

            # ── State baru ────────────────────────────────────
            new_states = {i: get_state(i, env) for i in range(NUM_AGENTS)}

            # ── Simpan pengalaman agen utama ──────────────────
            for i in actions_idx:
                done = env.agents[i].is_dead
                brain.memories[i].push(
                    states[i], actions_idx[i],
                    rewards.get(i, 0), new_states[i], done)

            # ── Simpan pengalaman companion (shared memory) ───
            for c in env.companions:
                if c.id not in comp_states:
                    continue
                ns   = get_companion_state(c, env)
                done = c.is_dead
                companion_mgr.memory.push(
                    comp_states[c.id],
                    comp_actions_idx.get(c.id, 0),
                    comp_rewards.get(c.id, 0),
                    ns, done)

            # ── Train ─────────────────────────────────────────
            if global_step % CFG["train_every_n_steps"] == 0:
                for i in range(NUM_AGENTS):
                    _train_agent_step(brain, i, CFG["batch_size"], CFG["gamma"])
                companion_mgr.train_step(CFG["batch_size"], CFG["gamma"])

            # ── Sync target networks ──────────────────────────
            if global_step % CFG["target_update_steps"] == 0:
                brain.sync_all_targets()
                companion_mgr.sync_target()

            total_reward += sum(rewards.values()) + sum(comp_rewards.values())
            states = new_states

            # ── Visualizer ────────────────────────────────────
            if viz:
                viz.render(env, ep + 1, step_n, epsilon, total_reward)
                if not viz.handle_events():
                    print("\n  Visualizer ditutup.")
                    viz.close()
                    return

            # Semua agen utama mati → selesai episode lebih awal
            if all(a.is_dead for a in env.agents):
                break

        # ── Reward akhir episode untuk companion ──────────────
        end_comp = env.end_of_episode_companion_rewards()
        for cid, er in end_comp.items():
            c = next((c for c in env.companions if c.id == cid), None)
            if c:
                s = get_companion_state(c, env)
                companion_mgr.memory.push(s, 0, er, s, True)
        total_reward += sum(end_comp.values())

        # ── Epsilon decay ─────────────────────────────────────
        epsilon = max(CFG["epsilon_min"], epsilon * CFG["epsilon_decay"])

        # ── Catat statistik ───────────────────────────────────
        alive   = sum(1 for a in env.agents if not a.is_dead)
        dormant = sum(1 for a in env.agents if a.is_dormant)
        n_comp  = sum(1 for c in env.companions if not c.is_dead)
        stats.log(ep + 1, total_reward, alive, dormant,
                  ep_deaths, step_n + 1, epsilon, n_comp)

        # ── Print progress setiap 10 ep ───────────────────────
        if (ep + 1) % 10 == 0:
            revives = sum(c.successful_revives for c in env.companions)
            print(f"  ep {ep+1:>4}  rew {total_reward:>9.1f}"
                  f"  alive {alive}/{NUM_AGENTS}"
                  f"  comp {n_comp}/{len(env.companions)}"
                  f"  rev {revives}"
                  f"  ε {epsilon:.3f}"
                  f"  step {global_step}")

        # ── Save checkpoint ───────────────────────────────────
        if (ep + 1) % CFG["save_every_n_eps"] == 0:
            save_brain(brain, companion_mgr, ep + 1, epsilon, global_step, stats)

        # ── Plot stats ────────────────────────────────────────
        if (ep + 1) % CFG["plot_every_n_eps"] == 0:
            stats.plot(save_path=CFG["stats_plot_path"], show=False)

    stats.summary()
    if viz:
        viz.close()
    print("  Training selesai.")


# ══════════════════════════════════════════════════════════════
# SEKSI 12 — DEMO MODE
# ══════════════════════════════════════════════════════════════
def demo():
    if not PYGAME_OK:
        print("pygame tidak terinstall — demo butuh pygame."); return

    brain         = AgentBrainManager()
    companion_mgr = CompanionBrainManager()
    ckpt          = get_latest_checkpoint()

    if not ckpt:
        print("Belum ada checkpoint — jalankan train dulu."); return

    brain, companion_mgr, meta = load_brain(ckpt, brain, companion_mgr)
    print(f"  Menonton ep {meta['episode']} | ε = 0 (greedy)")

    viz = Visualizer()
    ep  = 0

    while True:
        ep  += 1
        env  = MaklukGabutEnv()
        states = {i: get_state(i, env) for i in range(NUM_AGENTS)}

        for step_n in range(CFG["max_steps"]):
            # Agen utama — greedy
            actions_str = {}
            for i, a in enumerate(env.agents):
                if not a.is_dead:
                    idx = _select_action(brain.policy_nets[i], states[i], 0.0, ACTION_SIZE)
                    actions_str[i] = ACTION_MAP[idx]

            # Companion — greedy
            comp_actions_str = {}
            for c in env.companions:
                if not c.is_dead:
                    cs  = get_companion_state(c, env)
                    idx = companion_mgr.select_action(cs, 0.0)
                    comp_actions_str[c.id] = COMPANION_ACTION_MAP[idx]

            rewards, _ = env.step(actions_str)
            env.step_companions(comp_actions_str)
            states = {i: get_state(i, env) for i in range(NUM_AGENTS)}

            total_r = sum(rewards.values())
            viz.render(env, ep, step_n, 0.0, total_r)
            if not viz.handle_events():
                viz.close()
                return

            if all(a.is_dead for a in env.agents):
                time.sleep(1)
                break


# ══════════════════════════════════════════════════════════════
# SEKSI 13 — MAIN
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="MaklukGabut RL Simulator")
    parser.add_argument("mode", choices=["train", "demo", "plot", "list"])
    parser.add_argument("--resume", action="store_true",
                        help="Lanjutkan dari checkpoint terakhir")
    parser.add_argument("--visual", action="store_true",
                        help="Aktifkan visualisasi pygame saat training")
    args = parser.parse_args()

    if args.mode == "train":
        train(resume=args.resume, visual=args.visual)

    elif args.mode == "demo":
        demo()

    elif args.mode == "plot":
        ckpt = get_latest_checkpoint()
        if not ckpt:
            print("Belum ada checkpoint."); return
        stats = StatsTracker()
        sp = os.path.join(ckpt, "stats.json")
        if os.path.exists(sp):
            stats.load(sp)
            stats.plot(save_path=CFG["stats_plot_path"], show=True)
        else:
            print("Tidak ada stats.json di checkpoint terakhir.")

    elif args.mode == "list":
        list_checkpoints()


if __name__ == "__main__":
    main()
