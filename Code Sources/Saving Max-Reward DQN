import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
from collections import deque
# -----------------------
# Load bay files
# -----------------------
root = "BF"
bay_files = []

for dirpath, _, files in os.walk(root):
    for file in files:
        bay_files.append(os.path.join(dirpath, file))


def load_bay_file(index):
    path = bay_files[index]
    stacks = []
    width = None
    height = None

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if parts[0] == "Width":
                width = int(parts[2])
            elif parts[0] == "Height":
                height = int(parts[2])
            elif parts[0] == "Stack":
                stack = list(map(int, parts[3:])) if len(parts) > 3 else []
                stacks.append(stack)

    return stacks, height, width


# -----------------------
# Hyperparameters
# -----------------------
iteration_num = 300
alpha = 0.00005
gamma = 0.95
epsilon_start = 1
epsilon_end = 0.1
decay_rate = 0.995
n_steps = 3

replay = deque(maxlen=500000)
train_start = 1000
batch_size = 32
# -----------------------
# Environment
# -----------------------
grid, height, width = load_bay_file(1)
actions = [(x, y) for x in range(width) for y in range(width) if x != y]
action_number = len(actions)


# -----------------------
# Dueling N-step DQN
# -----------------------
class DuelingNstepDQN(nn.Module):
    def __init__(self, width, height, action_num):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(width * height * 32, 128)
        self.V = nn.Linear(128, 1)
        self.A = nn.Linear(128, action_num)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        V = self.V(x)
        A = self.A(x)
        return V + (A - A.mean(dim=1, keepdim=True))


dqn = DuelingNstepDQN(width, height, action_number)
optimizer = optim.Adam(dqn.parameters(), lr=alpha)
loss_fn = nn.MSELoss()

target_dqn = DuelingNstepDQN(width,height,action_number)
target_dqn.load_state_dict(dqn.state_dict())
target_dqn.eval()
# -----------------------
# Helpers
# -----------------------
def grid_to_tensor(grid):
    padded = [col + [0] * (height - len(col)) for col in grid]
    arr = np.array(padded, dtype=np.float32).T/10
    return torch.tensor(arr).unsqueeze(0).unsqueeze(0)


def greedy_select(state,epsilon):
    if random.random() < epsilon:
        return random.randrange(action_number)
    with torch.no_grad():
        return dqn(state).argmax(dim=1).item()


def compute_nstep(buffer, next_grid):
    G = 0
    for i, (_, _, r,_) in enumerate(buffer):
        G += (gamma ** i) * r
    if not buffer[-1][3]:
        with torch.no_grad():
            q_online = dqn(grid_to_tensor(next_grid))
            next_action = torch.argmax(q_online, dim=1)
            q_target = target_dqn(grid_to_tensor(next_grid))
            G += (gamma ** len(buffer)) * q_target.gather(1, next_action.unsqueeze(1)).item()

    return G


def is_sorted(col):
    if len(col) <= 1:
        return True
    return all(col[i] >= col[i + 1] for i in range(len(col) - 1))


def is_done(grid):
    return all(is_sorted(col) for col in grid)


def sorted_prefix_length(stack):
    cnt = 0
    for i in range(len(stack) - 1):
        if stack[i] >= stack[i + 1]:
            cnt += 1
        else:
            break
    return cnt

def relocation_cost(stack):
    cost = 0
    n = len(stack)
    for i in range(n - 1):
        if stack[i] < stack[i + 1]:  # violation
            cost += (n - 1 - i)
            break
    return cost

def potential(grid):
    return sum(sorted_prefix_length(col) for col in grid)


reward_out = -2
reward_goal = 200
reward_step = -0.01
reward_loop = -0.1

def simulate_action(grid, action_index):
    from_stack, to_stack = actions[action_index]
    new_grid = [col[:] for col in grid]
    reward = reward_step
    #punish making invalid moves
    if len(new_grid[from_stack]) ==0 or len(new_grid[to_stack]) >= height:
        return new_grid, reward_out, False

    if len(new_grid[from_stack]) == 1 and len(new_grid[to_stack]) == 0:
        reward -= 1.0
    #punish if you take from a sorted stack

    block = new_grid[from_stack].pop()
    new_grid[to_stack].append(block)
    
    phi_before = potential(grid)
    phi_after = potential(new_grid)
    reward += 0.02*(phi_after - phi_before)
    
    if phi_after > phi_before:
        reward += 0.02

    done = is_done(new_grid)
    if done:
        reward += reward_goal

    return new_grid, reward, done

def grid_key(grid):
    return tuple(tuple(col) for col in grid)

def train_from_replay():
    if len(replay) < batch_size:
        return
    batch = random.sample(replay,batch_size)
    states,actions,returns,next_states,dones = zip(*batch)
    
    states = torch.cat(states)
    actions = torch.tensor(actions)
    returns = torch.tensor(returns, dtype=torch.float32)
    next_states = torch.cat(next_states)
    dones = torch.tensor(dones,dtype=torch.float32)
    
    q_pred = dqn(states).gather(1,actions.unsqueeze(1)).squeeze(1)

    targets = returns
    
    loss = loss_fn(q_pred,targets)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(dqn.parameters(),5.0)
    optimizer.step()
    
# -----------------------
# Training (with visited)
# -----------------------
reward_history = []
best_reward = -float("inf")
for ep in range(iteration_num):
    grid, _, _ = load_bay_file(1)
    buffer = []
    visited = {grid_key(grid)}
    done = False
    steps = 0
    total_reward = 0
    
    while not done and steps < 200:
        steps += 1
        s = grid_to_tensor(grid)
        a = greedy_select(s,epsilon_start)
        new_grid, r, done = simulate_action(grid, a)

        key = grid_key(new_grid)
        if key in visited:
            r += reward_loop
        else:
            visited.add(key)

        total_reward+=r
        buffer.append((s, a, r,done))

        if len(buffer) >= n_steps:
            G = compute_nstep(buffer, new_grid)
            s0, a0, _, _ = buffer.pop(0)
            
            replay.append((s0,a0,G,grid_to_tensor(new_grid),done))
            train_from_replay()
        grid = new_grid
        epsilon_start = max(epsilon_end,epsilon_start*decay_rate)
        
    while len(buffer) > 0:
        G = compute_nstep(buffer, grid)
        s0, a0, _, _ = buffer.pop(0)
        replay.append((s0, a0, G, grid_to_tensor(grid), True))

    if ep<20 or ep%10==0:
        target_dqn.load_state_dict(dqn.state_dict())
    print(f"Episode {ep}, steps = {steps}, reward = {total_reward}")
    
    if total_reward > best_reward:
        best_reward = total_reward
        torch.save(dqn.state_dict(), "best_dqn.pth")
        print(f"Saved new best model at episode {ep}, reward = {total_reward}")


dqn.load_state_dict(torch.load("best_dqn.pth"))
dqn.eval()
# -----------------------
# Beam Search (with visited)
# -----------------------
k_size = 50
max_depth = 100

grid, _, _ = load_bay_file(1)
beam = [(grid, 0, [], False, {grid_key(grid)})]

for step in range(max_depth):
    candidates = []

    for g, total_r, traj, done, visited in beam:
        with torch.no_grad():
            qvals = dqn(grid_to_tensor(g))

        topk = torch.topk(qvals, min(k_size, action_number), dim=1).indices[0]

        for a in topk.tolist():
            ng, r, d = simulate_action(g, a)

            key = grid_key(ng)
            new_visited = visited.copy()
            if key in visited:
                r += reward_loop
            else:
                new_visited.add(key)

            candidates.append(
                (ng, total_r + r, traj + [(g, a, r)], d, new_visited)
            )

    if not candidates:
        break

    candidates.sort(key=lambda x: x[1], reverse=True)
    beam = candidates[:k_size]
    if any(d for _, _, _, d, _ in beam):
        break


# -----------------------
# Print best trajectory
# -----------------------
best_grid, best_reward, best_traj, best_done, _ = beam[0]

print("\n=========== BEST TRAJECTORY ===========")
print("Total reward:", best_reward)
print("Done:", best_done)
print("Steps:", len(best_traj))

for i, (g, a, r) in enumerate(best_traj):
    from_s, to_s = actions[a]
    print(f"\nStep {i+1}")
    print(f"Action: {from_s} -> {to_s}")
    print(f"Reward: {r}")
    print("Grid:")
    for col in g:
        print(col)
