import tkinter as tk
from blackjack_playable import CustomBlackjackEnv

class BlackjackGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Blackjack")
        self.env = CustomBlackjackEnv(mode='human')
        self.label = tk.Label(self.root, text="")
        self.label.pack()
        self.button_stand = tk.Button(self.root, text="Stand", command=lambda: self.take_action(0))
        self.button_stand.pack()
        self.button_hit = tk.Button(self.root, text="Hit", command=lambda: self.take_action(1))
        self.button_hit.pack()
        self.restart_game()
        self.root.mainloop()

    def restart_game(self):
        self.obs = self.env.reset()
        self.label.config(text="Obs: {}".format(self.obs))

    def take_action(self, action):
        obs, reward, done, _ = self.env.step(action)
        self.label.config(text="Obs: {}, Reward: {}".format(obs, reward))
        if done:
            self.result_label.config(text="Game Over! Result: {}".format(reward))
            self.continue_entry.delete(0, tk.END)
            self.continue_entry.insert(0, "Do you want to continue playing? (yes/no)")
            self.continue_entry.bind("<Return>", self.handle_continue)

    def handle_continue(self, event):
        continue_playing = self.continue_entry.get().strip().lower()
        if continue_playing.startswith('y') or continue_playing == 'yes':
            self.result_label.config(text="")
            self.continue_entry.delete(0, tk.END)
            self.restart_game()
        elif continue_playing.startswith('n') or continue_playing == 'no':
            self.root.destroy()
        else:
            self.continue_entry.delete(0, tk.END)
            self.continue_entry.insert(0, "Invalid input. Please enter 'yes' or 'no'.")


if __name__ == "__main__":
    BlackjackGUI()