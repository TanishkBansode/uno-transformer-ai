from uno_full import UnoGame, Card
import jax.numpy as jnp
import pickle

# Load model (mocked for this test)
def get_ai_move(game, player_idx):
    return 0 # Simplified AI move

def play_game():
    game = UnoGame(["Human", "AI"])
    while True:
        print(f"\nTop: {game.discard_pile[-1]} (Color: {game.current_color})")
        print(f"Hand: {game.players[0].hand}")
        
        choice = input("Index to play (or 'd'): ")
        if choice == 'd':
            game.play_turn(0, None)
        else:
            idx = int(choice)
            card = game.players[0].hand[idx]
            color = None
            if card.color is None:
                color = input("Choose color (Red, Blue, Green, Yellow): ")
            game.play_turn(0, idx, chosen_color=color)
        
        if len(game.players[0].hand) == 0: break

if __name__ == "__main__":
    play_game()
