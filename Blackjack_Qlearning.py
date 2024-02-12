from __future__ import annotations
from collections import defaultdict
import numpy as np
from matplotlib.patches import Patch
from tqdm import tqdm
import gym
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define a função para criar a grade de valores e políticas
def create_grids(agent, usable_ace=False):
    """Create value and policy grid given an agent."""
    # Convertendo state-action values para valores de estado
    # e construindo um dicionário de políticas que mapeia observações para ações
    state_value = defaultdict(float)
    policy = defaultdict(int)
    for obs, action_values in agent.q_values.items():
        state_value[obs] = float(np.max(action_values))
        policy[obs] = int(np.argmax(action_values))

    player_count, dealer_count = np.meshgrid(
        np.arange(12, 22),
        np.arange(1, 11),
    )

    # Criando a grade de valores para plotagem
    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    value_grid = player_count, dealer_count, value

    # Criando a grade de políticas para plotagem
    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return value_grid, policy_grid

# Define a função para criar os gráficos
def create_plots(value_grid, policy_grid, title: str):
    """Creates a plot using a value and policy grid."""
    # Criando uma nova figura com 2 subplots (esquerda: valores de estado, direita: política)
    player_count, dealer_count, value = value_grid
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle(title, fontsize=16)

    # Plotando os valores de estado
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        player_count,
        dealer_count,
        value,
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )
    plt.xticks(range(12, 22), range(12, 22))
    plt.yticks(range(1, 11), ["A"] + list(range(2, 11)))
    ax1.set_title(f"State values: {title}")
    ax1.set_xlabel("Player sum")
    ax1.set_ylabel("Dealer showing")
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel("Value", fontsize=14, rotation=90)
    ax1.view_init(20, 220)

    # Plotando a política
    fig.add_subplot(1, 2, 2)
    ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    ax2.set_title(f"Policy: {title}")
    ax2.set_xlabel("Player sum")
    ax2.set_ylabel("Dealer showing")
    ax2.set_xticklabels(range(12, 22))
    ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

    # Adicionando uma legenda
    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
    return fig

env = gym.make("Blackjack-v1", sab=True)

class BlackjackAgent(nn.Module):
    def __init__(
        self,
        learning_rate: float,
        num_actions: int,
        input_shape: int,
        hidden_layers: list[int],
        epsilon: float,
        epsilon_decay: float,
        min_epsilon: float
    ):
        super(BlackjackAgent, self).__init__()
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
                
        self.fc1 = nn.Linear(input_shape,hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], num_actions)
        
        # Inicializando o otimizador
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Função de perda (MSE)
        self.loss_function = nn.MSELoss()
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def calcular_target(self, Q, Q_proximo, reward, learning_rate):
        gamma = 0.9
        target_q = Q + learning_rate * (reward + gamma * torch.max(Q_proximo) - Q)
        return target_q

    def get_action(self, obs):
        if isinstance(obs, tuple) and len(obs) == 3:
            player_score, dealer_score, usable_ace = obs
        else:
            player_score, dealer_score, usable_ace = obs[0]
        
        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            obs_tensor = torch.Tensor([player_score, dealer_score, int(usable_ace)]).unsqueeze(0)
            q_values = self(obs_tensor).detach().numpy()
            return np.argmax(q_values)

    def update(self, obs, action, reward, next_obs, done):
        if isinstance(obs, tuple) and len(obs) == 3:
            player_score, dealer_score, usable_ace = obs
        else:
            player_score, dealer_score, usable_ace = obs[0]
                
        obs_tensor = torch.tensor([player_score, dealer_score, int(usable_ace)]).float()

        next_obs_tensor = torch.tensor([next_obs]).float()

        next_obs_tensor = torch.Tensor(next_obs).unsqueeze(0)

        q_values = self(obs_tensor)
        
        Q_proximo = self(next_obs_tensor)

        # Calculando o target Q-value usando a fórmula de Bellman
        Q = q_values.clone().detach()
        
        target_q = self.calcular_target(Q, Q_proximo, reward, learning_rate)


        # Calculando a perda e realizando uma etapa de otimização
        loss = self.loss_function(Q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# Parâmetros do agente DQN
learning_rate = 0.001
num_actions = 2  # hit or stick
input_shape = 3  # state: player sum, dealer showing, usable ace
hidden_layers = [128, 64]
epsilon = 1.0
epsilon_decay = 0.9999
min_epsilon = 0.1
n_episodes = 1000000
episode_reward = 0
episode_rewards = []

agent = BlackjackAgent(
    learning_rate=learning_rate,
    num_actions=num_actions,
    input_shape=input_shape,
    hidden_layers=hidden_layers,
    epsilon=epsilon,
    epsilon_decay=epsilon_decay,
    min_epsilon=min_epsilon
)


for episode in tqdm(range(n_episodes)):
    obs = env .reset()
    done = False

    # play one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, done,truncate, _ = env.step(action)

        # Atualizando o agente
        agent.update(obs, action, reward, next_obs, done)

        # Atualizando a observação atual
        obs = next_obs

        # Adicionando a recompensa do passo ao retorno do episódio
        episode_reward += reward

    agent.decay_epsilon()
    
     # Armazenando o retorno do episódio
    episode_rewards.append(episode_reward)
    
    
# Plotando o retorno acumulado ao longo dos episódios
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward per Episode")
plt.show()

# Criando e exibindo os gráficos de valores e políticas
value_grid, policy_grid = create_grids(agent, usable_ace=True)
fig1 = create_plots(value_grid, policy_grid, title="With usable ace")
plt.show()
