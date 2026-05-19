import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, Counter

# ══════════════════════════════════════════════════════════════
#  TOTAL NN = 12 (6 Policy + 6 Target, satu pasang per agen)
#
#  CHANGELOG v3:
#  [NEW] Boost Mechanic: 2 energi, 2 petak, masih butuh stack
#  [FIX] Race condition → gerakan sekarang atomik (3 fase)
#  [FIX] Boost bug: agen tidak bisa gerak gratis jika stack habis
#  [KEEP] global_step % 4 dari kode user (bagus untuk stabilitas)
# ══════════════════════════════════════════════════════════════

NUM_AGENTS  = 6
ACTION_MAP  = [
    "IDLE", "MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT",
    "BOOST_UP", "BOOST_DOWN", "BOOST_LEFT", "BOOST_RIGHT",
    "TAKE_FOOD", "EAT_FOOD", "GIVE_FOOD", "REVIVE",
    "CARRY_AGENT", "DROP_AGENT", "SHARE_INFO"
]
ACTION_SIZE = len(ACTION_MAP)   # 16
STATE_SIZE  = 8                 # x, y, energy, boost, inv, has_friend, sos, agent_id


# ══════════════════════════════════════════════════════════════
# 1. KELAS AGEN
# ══════════════════════════════════════════════════════════════
class MaklukGabut:
    def __init__(self, agent_id, x, y):
        self.id = agent_id
        self.x, self.y   = x, y
        self.energy      = 50
        self.steps_taken = 0
        self.is_dormant  = False
        self.dormant_timer = 0
        self.is_dead       = False
        self.inventory        = []
        self.carrying         = None
        self.is_being_carried = False
        self.boost_charge = 0
        self.boost_stacks = 0
        self.food_memory  = None

    def release_carrying(self):
        if self.carrying is not None:
            self.carrying.is_being_carried = False
            self.carrying = None


# ══════════════════════════════════════════════════════════════
# 2. LINGKUNGAN (3-FASE ATOMIK)
# ══════════════════════════════════════════════════════════════
class MaklukGabutEnv:
    def __init__(self, grid_size=12, num_agents=NUM_AGENTS):
        self.grid_size  = grid_size
        self.num_agents = num_agents
        self.step_count = 0
        self.agents     = [
            MaklukGabut(i, np.random.randint(0, grid_size), np.random.randint(0, grid_size))
            for i in range(num_agents)
        ]
        self.food_grid = np.zeros((grid_size, grid_size), dtype=bool)
        self.spawn_food(15)

    def spawn_food(self, amount):
        spawned = attempts = 0
        while spawned < amount and attempts < amount * 10:
            x, y = np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)
            if not self.food_grid[x, y]:
                self.food_grid[x, y] = True
                spawned += 1
            attempts += 1

    def _clamp(self, x, y):
        return (max(0, min(self.grid_size - 1, x)),
                max(0, min(self.grid_size - 1, y)))

    @staticmethod
    def _get_direction(action: str):
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
        neighbors = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = agent.x + dx, agent.y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    n = self.get_agent_at(nx, ny)
                    if n and n.id != agent.id:
                        neighbors.append(n)
        return neighbors

    def get_sos_signals(self, agent):
        return [(n.x, n.y) for n in self.get_neighbors(agent, radius=2) if n.is_dormant]

    # ──────────────────────────────────────────────────────────
    def step(self, actions_dict):
        self.step_count += 1
        if self.step_count % 20 == 0:
            self.spawn_food(3)

        alive_at_start        = {a.id for a in self.agents if not a.is_dead}
        agent_pairing_penalty = {aid: 0 for aid in alive_at_start}
        dead_agents_this_step = []

        # ══════════════════════════════════════════════
        # FASE 0 — Dormant & Boost charge accumulation
        # ══════════════════════════════════════════════
        for agent in self.agents:
            if agent.is_dead or agent.is_being_carried:
                continue
            if agent.is_dormant:
                agent.dormant_timer += 1
                if agent.dormant_timer >= 60:
                    agent.release_carrying()
                    agent.is_dead = True
                    dead_agents_this_step.append(agent.id)
                continue
            if agent.boost_stacks < 5:
                agent.boost_charge += 1
                if agent.boost_charge >= 10:
                    agent.boost_stacks += 1
                    agent.boost_charge = 0

        active = [a for a in self.agents
                  if not a.is_dead and not a.is_dormant and not a.is_being_carried]

        # ══════════════════════════════════════════════
        # FASE 1 — Semua agen DEKLARASIKAN niat bergerak
        # [FIX RACE CONDITION] Tidak ada yang benar-benar
        # bergerak dulu sebelum seluruh niat terkumpul.
        #
        # Intent: {agent_id: {dest, wp, is_boost}}
        #   dest     = tujuan akhir (petak 2 untuk boost)
        #   wp       = waypoint petak 1 (hanya untuk boost)
        #   is_boost = True jika pakai boost
        # ══════════════════════════════════════════════
        intents = {}
        for agent in active:
            action = actions_dict.get(agent.id, "IDLE")

            if action.startswith("MOVE_"):
                dx, dy = self._get_direction(action)
                dest = self._clamp(agent.x + dx, agent.y + dy)
                intents[agent.id] = {'dest': dest, 'wp': None, 'is_boost': False}

            elif action.startswith("BOOST_") and agent.boost_stacks > 0:
                # ╔══════════════════════════════════════╗
                # ║  BOOST MECHANIC BARU                 ║
                # ║  - Butuh 1 boost stack               ║
                # ║  - Biaya: 2 energi saat berhasil     ║
                # ║  - Jarak: 2 petak ke depan           ║
                # ║  - Jika petak-1 terblokir → diam     ║
                # ║  - Jika petak-2 terblokir → ke pkt-1 ║
                # ╚══════════════════════════════════════╝
                dx, dy = self._get_direction(action)
                wp   = self._clamp(agent.x + dx,     agent.y + dy)      # petak 1
                dest = self._clamp(agent.x + dx * 2, agent.y + dy * 2)  # petak 2
                intents[agent.id] = {'dest': dest, 'wp': wp, 'is_boost': True}

            else:
                # IDLE, aksi non-gerak, atau BOOST gagal (stack habis) → DIAM
                intents[agent.id] = {'dest': (agent.x, agent.y), 'wp': None, 'is_boost': False}

        # ══════════════════════════════════════════════
        # FASE 2 — Resolusi konflik (atomik, deterministik)
        #
        # Aturan:
        # A. Agen statis (tidak bergerak) memblokir cell-nya
        # B. Jika >1 agen menginginkan cell yang sama → semua diam
        # C. Boost: cek waypoint dulu sebelum dest final
        # ══════════════════════════════════════════════
        moving_ids = {aid for aid, it in intents.items()
                      if it['dest'] != (self.agents[aid].x, self.agents[aid].y)}
        # Cell yang ditempati agen yang tidak bergerak = blokir permanen
        static_cells = {(a.x, a.y) for a in active if a.id not in moving_ids}

        # Hitung klaim per destinasi
        dest_claims = Counter(it['dest'] for it in intents.values())

        final_pos = {}
        for agent in active:
            it       = intents[agent.id]
            dest     = it['dest']
            wp       = it['wp']
            is_boost = it['is_boost']
            orig     = (agent.x, agent.y)

            if dest == orig:
                final_pos[agent.id] = orig
                continue

            # Aturan B: destinasi diperebutkan
            if dest_claims[dest] > 1:
                final_pos[agent.id] = orig
                continue

            if is_boost:
                if wp in static_cells:
                    # Petak-1 terblokir → diam total (tidak bisa boost sama sekali)
                    final_pos[agent.id] = orig
                elif dest in static_cells:
                    # Petak-1 bebas, petak-2 terblokir → berhenti di petak-1
                    final_pos[agent.id] = wp
                else:
                    # Keduanya bebas → maju 2 petak
                    final_pos[agent.id] = dest
            else:
                if dest in static_cells:
                    final_pos[agent.id] = orig   # terblokir agen statis
                else:
                    final_pos[agent.id] = dest   # bebas, bergerak

        # Apply gerakan final
        for agent in active:
            new_pos = final_pos[agent.id]
            if new_pos == (agent.x, agent.y):
                continue  # Tidak bergerak, skip

            agent.x, agent.y = new_pos
            if agent.carrying:
                agent.carrying.x, agent.carrying.y = new_pos

            if intents[agent.id]['is_boost']:
                agent.boost_stacks -= 1     # konsumsi 1 stack
                agent.energy       -= 2     # biaya energi boost
            else:
                agent.energy -= 2 if agent.carrying else 1

            agent.steps_taken += 1

        # ══════════════════════════════════════════════
        # FASE 3 — Interaksi non-gerakan
        # Semua agen sudah di posisi finalnya.
        # [FIX RACE CONDITION] Tidak ada lagi state yang
        # berubah di tengah jalan saat iterasi neighbor.
        # ══════════════════════════════════════════════
        for agent in active:
            action    = actions_dict.get(agent.id, "IDLE")
            neighbors = self.get_neighbors(agent, radius=1)

            # Food
            if action == "TAKE_FOOD" and len(agent.inventory) < 3:
                if self.food_grid[agent.x, agent.y]:
                    agent.inventory.append("food")
                    self.food_grid[agent.x, agent.y] = False
                    agent.food_memory = (agent.x, agent.y)

            elif action == "EAT_FOOD" and "food" in agent.inventory:
                agent.inventory.remove("food")
                agent.energy = min(50, agent.energy + 20)

            # Crowding penalty (per-agen, bukan global)
            if len(neighbors) > 1:
                valid = any(n.is_dormant for n in neighbors) or action == "SHARE_INFO"
                if not valid:
                    agent_pairing_penalty[agent.id] -= 5

            # Neighbor interactions
            for neighbor in neighbors:
                if action == "GIVE_FOOD" and "food" in agent.inventory and neighbor.energy < 20:
                    agent.inventory.remove("food")
                    neighbor.energy = min(50, neighbor.energy + 20)
                    break

                elif action == "REVIVE" and neighbor.is_dormant:
                    neighbor.is_dormant  = False
                    neighbor.steps_taken = 0
                    shared = (agent.energy + neighbor.energy) / 2
                    agent.energy = neighbor.energy = shared
                    break

                elif action == "CARRY_AGENT" and agent.carrying is None and neighbor.is_dormant:
                    agent.carrying            = neighbor
                    neighbor.is_being_carried = True
                    break

                elif action == "SHARE_INFO" and agent.food_memory:
                    neighbor.food_memory = agent.food_memory

            # Drop
            if action == "DROP_AGENT" and agent.carrying is not None:
                dropped = False
                for ddx in [-1, 0, 1]:
                    for ddy in [-1, 0, 1]:
                        if ddx == 0 and ddy == 0:
                            continue
                        nx, ny = self._clamp(agent.x + ddx, agent.y + ddy)
                        if self.get_agent_at(nx, ny) is None:
                            agent.carrying.x, agent.carrying.y = nx, ny
                            agent.release_carrying()
                            dropped = True
                            break
                    if dropped:
                        break

        # ══════════════════════════════════════════════
        # FASE 4 — Evaluasi kematian / dormancy
        # ══════════════════════════════════════════════
        for agent in active:
            if agent.is_dead:
                continue
            is_accompanied = len(self.get_neighbors(agent, radius=1)) > 0

            if agent.energy <= 0:
                agent.release_carrying()
                agent.is_dead = True
                dead_agents_this_step.append(agent.id)

            elif agent.steps_taken >= 49:
                agent.release_carrying()
                agent.is_dormant    = True
                agent.dormant_timer = 0

            elif agent.steps_taken >= 40 and not is_accompanied:
                agent.release_carrying()
                agent.is_dormant    = True
                agent.dormant_timer = 0

        # ── Reward ─────────────────────────────────────
        total_alive_after = sum(1 for a in self.agents if not a.is_dead)
        num_deaths        = len(dead_agents_this_step)

        rewards_dict = {}
        for aid in alive_at_start:
            base    = total_alive_after * 2
            pairing = agent_pairing_penalty.get(aid, 0)
            if aid in dead_agents_this_step:
                rewards_dict[aid] = base + pairing - 10
            else:
                rewards_dict[aid] = base + pairing + (-5 * num_deaths)

        return rewards_dict, dead_agents_this_step


# ══════════════════════════════════════════════════════════════
# 3. NEURAL NETWORK & REPLAY BUFFER
# ══════════════════════════════════════════════════════════════
class MaklukBrain(nn.Module):
    def __init__(self, input_size=STATE_SIZE, output_size=ACTION_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(),
            nn.Linear(128, 128),        nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        s, a, r, ns, d = zip(*random.sample(self.buffer, batch_size))
        return (np.array(s,  dtype=np.float32), np.array(a,  dtype=np.int64),
                np.array(r,  dtype=np.float32), np.array(ns, dtype=np.float32),
                np.array(d,  dtype=bool))

    def __len__(self):
        return len(self.buffer)


def get_agent_state(agent_id: int, env: MaklukGabutEnv) -> np.ndarray:
    agent = env.agents[agent_id]
    if agent.is_dead:
        return np.zeros(STATE_SIZE, dtype=np.float32)
    return np.array([
        agent.x / env.grid_size,
        agent.y / env.grid_size,
        agent.energy / 50.0,
        agent.boost_stacks / 5.0,
        len(agent.inventory) / 3.0,
        1.0 if env.get_neighbors(agent, radius=1) else 0.0,
        min(len(env.get_sos_signals(agent)) / 5.0, 1.0),
        agent_id / max(1, env.num_agents - 1)
    ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════
# 4. MULTI-AGENT BRAIN MANAGER
# ══════════════════════════════════════════════════════════════
class AgentBrainManager:
    """
    Setiap agen punya sepasang NN sendiri.
    Total NN = NUM_AGENTS * 2  →  untuk 6 agen: 12 NN
    """
    def __init__(self, num_agents: int, lr: float = 1e-3):
        self.num_agents  = num_agents
        self.policy_nets = {}
        self.target_nets = {}
        self.optimizers  = {}
        self.memories    = {}

        for i in range(num_agents):
            p = MaklukBrain()
            t = MaklukBrain()
            t.load_state_dict(p.state_dict())
            t.eval()
            self.policy_nets[i] = p
            self.target_nets[i] = t
            self.optimizers[i]  = optim.Adam(p.parameters(), lr=lr)
            self.memories[i]    = ReplayBuffer(capacity=10000)

        params = sum(p.numel() for p in self.policy_nets[0].parameters())
        total  = num_agents * 2
        print(f"\n{'═'*52}")
        print(f"  ARSITEKTUR NN")
        print(f"{'═'*52}")
        print(f"  Policy Nets : {num_agents}  │  Target Nets : {num_agents}")
        print(f"  TOTAL       : {total} Neural Networks")
        print(f"  Param/NN    : {params:,}  │  Total param  : {params*total:,}")
        print(f"{'═'*52}\n")

    def sync_all_targets(self):
        for i in range(self.num_agents):
            self.target_nets[i].load_state_dict(self.policy_nets[i].state_dict())


# ══════════════════════════════════════════════════════════════
# 5. TRAINING LOOP
# ══════════════════════════════════════════════════════════════
def train_makluk_gabut():
    BATCH_SIZE          = 64
    GAMMA               = 0.99
    LR                  = 1e-3
    EPISODES            = 500
    MAX_STEPS           = 200
    TARGET_UPDATE_STEPS = 500   # sync target net per N global step
    TRAIN_EVERY_N_STEPS = 4     # dari kode user — bagus untuk stabilitas!

    epsilon       = 1.0
    epsilon_min   = 0.05
    epsilon_decay = 0.995

    criterion   = nn.MSELoss()
    brain       = AgentBrainManager(NUM_AGENTS, lr=LR)
    global_step = 0

    print("=== MEMULAI PELATIHAN MAKLUKGABUT'S GAME OF LIFE ===\n")

    for episode in range(EPISODES):
        env   = MaklukGabutEnv()
        total = 0.0

        for _ in range(MAX_STEPS):
            # State semua agen yang masih hidup
            current_states = {
                i: get_agent_state(i, env)
                for i in range(NUM_AGENTS)
                if not env.agents[i].is_dead
            }
            if not current_states:
                break

            # Pilih aksi (ε-greedy, brain terpisah per agen)
            actions_dict   = {}
            action_indices = {}
            for i, state in current_states.items():
                if random.random() < epsilon:
                    idx = random.randrange(ACTION_SIZE)
                else:
                    with torch.no_grad():
                        idx = brain.policy_nets[i](
                            torch.FloatTensor(state).unsqueeze(0)
                        ).argmax().item()
                action_indices[i] = idx
                actions_dict[i]   = ACTION_MAP[idx]

            rewards_dict, _ = env.step(actions_dict)
            total      += sum(rewards_dict.values())
            global_step += 1
            game_done   = all(a.is_dead for a in env.agents)

            # Simpan transisi ke memory masing-masing agen
            for i in current_states:
                brain.memories[i].push(
                    current_states[i],
                    action_indices[i],
                    rewards_dict.get(i, -10.0),
                    get_agent_state(i, env),
                    game_done or env.agents[i].is_dead
                )

            # Update policy setiap N step (bukan tiap step → lebih stabil)
            if global_step % TRAIN_EVERY_N_STEPS == 0:
                for i in range(NUM_AGENTS):
                    if len(brain.memories[i]) < BATCH_SIZE:
                        continue

                    b_s, b_a, b_r, b_ns, b_d = brain.memories[i].sample(BATCH_SIZE)
                    b_s  = torch.FloatTensor(b_s)
                    b_
