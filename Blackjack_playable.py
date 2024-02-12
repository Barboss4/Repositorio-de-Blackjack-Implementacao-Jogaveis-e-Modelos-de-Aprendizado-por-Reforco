import gym
from gym import spaces
import numpy as np

class CustomBlackjackEnv(gym.Env):
    def __init__(self,mode):
        self.action_space = spaces.Discrete(2)  # 0: Stand, 1: Hit
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),  # Pontuação do jogador
            spaces.Discrete(11),  # Carta visível do dealer
            spaces.Discrete(2)    # Ás no jogador (0 ou 1)
        ))
        self.mode = mode
        self.reset()
        
    def step(self, action):
        assert self.action_space.contains(action)

        if action:  # Hit
            drawn_card = self.draw_card()
            self.player.append(drawn_card)
            if self.is_bust(self.player):
                done = True
                reward = self.get_reward()
            else:
                done = False
                reward = 0
            return self._get_obs(drawn_card), reward, done, {}
        else:  # Stand
            done = True
            while self.sum_hand(self.dealer) < 17:
                self.dealer.append(self.draw_card())
                
            reward = self.get_reward()
            
            return self._get_obs(), reward, done, {}
        
    def reset(self):
        self.player = [self.draw_card(), self.draw_card()]
        self.dealer = [self.draw_card(), self.draw_card()]
        return self._get_obs()

    def _get_obs(self, drawn_card=None):
        if drawn_card is None:
            drawn_card = 0
        
        if self.mode == 'human':
            if self.is_bust(self.player) or self.is_bust(self.dealer):
                return ('Your hand:', self.sum_hand(self.player), 'Dealer hand:',[self.dealer[0], '?'])
            
            else:
                return ('Your hand:', self.player,'=', self.sum_hand(self.player), 'Dealer hand:', [self.dealer[0], '?'])
            
            
        else:
            return (self.sum_hand(self.player), self.dealer[0], self.has_usable_ace(self.player), drawn_card)
        
    def draw_card(self):
        card = min(np.random.randint(1, 14), 10)
        #print(card)
        return card  # Cartas de 1 a 10

    def sum_hand(self, hand):
        total = sum(hand)
        if total > 21 and 11 in hand:
            hand.remove(11)
            hand.append(1)
            total = sum(hand)
        return total

    def has_usable_ace(self, hand):
        return 1 in hand and self.sum_hand(hand) + 10 <= 21

    def is_bust(self, hand):
        return self.sum_hand(hand) > 21

    def get_reward(self):
        player_score = self.sum_hand(self.player)
        dealer_score = self.sum_hand(self.dealer)

        if self.is_bust(self.player):
            if self.mode == 'human':
                print('your hand:',player_score)
                return "You busted! You lose."
            else:
                return -1
        elif self.is_bust(self.dealer) or player_score > dealer_score:
            if self.mode == 'human':
                print('your hand:',player_score,'Dealers Hand:',dealer_score)
                return "You win!"
            else:
                return 1
        elif player_score < dealer_score:
            if self.mode == 'human':
                print('your hand:',player_score,'Dealers Hand:',dealer_score)
                return "You lose."
            else:
                return -1
        else:
            if self.mode == 'human':
                print('your hand:',player_score,'Dealers Hand:',dealer_score)
                return "It's a tie."
            else:
                return 0
        
# Exemplo de uso:
env = CustomBlackjackEnv(mode='human')

while True:
    obs = env.reset()
    print("Obs:", obs)
    done = False

    while not done:
        while True:
            try:
                action = int(input("Choose your action (0 to Stand, 1 to Hit): "))
                if action == 0 or action == 1:
                    break
                else:
                    print("Invalid input. Please enter 0 to Stand or 1 to Hit.")
            except ValueError:
                print("Invalid input. Please enter a number (0 or 1).")

        obs, reward, done, _ = env.step(action)
        print("Obs:", obs)
        
        if done:
            while True:
                print("Game Over!")
                print(reward)
                continue_playing = input("Do you want to continue playing? (yes/no): ").strip().lower()
                if continue_playing.startswith('y') or continue_playing == 'yes':
                    print("Great! Let's continue.")
                    break  # Break out of the inner loop to reset the game
                elif continue_playing.startswith('n') or continue_playing == 'no':
                    exit()  # Exit the program if the player doesn't want to continue
                else:
                    print("Invalid input. Please enter 'yes' or 'no'.")
                    