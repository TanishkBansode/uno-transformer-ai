from uno_full import UnoGame, Card
import jax.numpy as jnp
import pickle
import random
from unified_model import UnifiedUnoBrain

# Load model
model = UnifiedUnoBrain()
with open('uno_brain_10k_real.pkl', 'rb') as f:
    params = pickle.load(f)

def get_ai_move(game, player_idx):
    # AI logic: Find all valid moves, then use the model to pick the best one
    hand = game.players[player_idx].hand
    valid_indices = [i for i, card in enumerate(hand) if game.is_valid_move(card)]
    
    if not valid_indices:
        return None # Must draw
    
    # Use model to pick from valid_indices
    # For now, pick the first valid index to ensure it works
    return valid_indices[0]

def play_game():
    game = UnoGame(["Human", "AI"])
    
    while True:
        # Human turn
        print(f"\n--- Human's Turn ---")
        print(f"Top: {game.discard_pile[-1]} (Color: {game.current_color})")
        print(f"Hand: {game.players[0].hand}")
        
        choice = input("Enter index to play (or 'd' to draw): ")
        if choice.lower() == 'd':
            game.play_turn(0, None)
        else:
            try:
                idx = int(choice)
                if 0 <= idx < len(game.players[0].hand):
                    card = game.players[0].hand[idx]
                    color = None
                    if card.color is None:
                        color = input("Choose color (Red, Blue, Green, Yellow): ")
                    game.play_turn(0, idx, chosen_color=color)
                else:
                    print("Index out of range!")
            except ValueError:
                print("Invalid input!")
        
        if len(game.players[0].hand) == 0:
            print("Human wins!")
            break
            
        # AI turn
        print(f"\n--- AI's Turn ---")
        ai_move = get_ai_move(game, 1)
        if ai_move is not None:
            card = game.players[1].hand[ai_move]
            color = None
            if card.color is None:
                color = random.choice(Card.COLORS)
            game.play_turn(1, ai_move, chosen_color=color)
        else:
            game.play_turn(1, None)
            
        if len(game.players[1].hand) == 0:
            print("AI wins!")
            break

if __name__ == "__main__":
    play_game()
