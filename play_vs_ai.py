import jax
import jax.numpy as jnp
import pickle
from unified_model import UnifiedUnoBrain
from uno_full import UnoGame, Card

# 1. Load the trained model
model = UnifiedUnoBrain()
with open('uno_brain_10k_real.pkl', 'rb') as f:
    params = pickle.load(f)

def get_ai_move(state, player_idx):
    # Build observation (simplified for interaction)
    # In a real game, we'd extract the actual state from the 'game' object
    obs = jnp.zeros(219) 
    belief = jnp.ones(108) / 108
    action_probs = model.apply({'params': params}, obs, belief)
    return int(jnp.argmax(action_probs))

def play_game():
    game = UnoGame(["Human", "AI"])
    
    while True:
        # Human turn
        print(f"\n--- Human's Turn ---")
        print(f"Top card: {game.discard_pile[-1]}")
        print(f"Your hand: {game.players[0].hand}")
        
        valid_moves = [i for i, card in enumerate(game.players[0].hand) if game.is_valid_move(card)]
        if not valid_moves:
            print("No valid moves. Drawing a card...")
            game.play_turn(0, None)
        else:
            choice = input(f"Enter index of card to play (Valid: {valid_moves}): ")
            if choice.lower() == 'd':
                game.play_turn(0, None)
            else:
                idx = int(choice)
                if idx in valid_moves:
                    game.play_turn(0, idx)
                else:
                    print("Invalid move!")
        
        if len(game.players[0].hand) == 0:
            print("Human wins!")
            break
            
        # AI turn
        print(f"\n--- AI's Turn ---")
        ai_move = get_ai_move(game, 1)
        # Ensure AI move is valid
        if ai_move < len(game.players[1].hand) and game.is_valid_move(game.players[1].hand[ai_move]):
            game.play_turn(1, ai_move)
        else:
            print("AI drawing...")
            game.play_turn(1, None)
            
        if len(game.players[1].hand) == 0:
            print("AI wins!")
            break

if __name__ == "__main__":
    play_game()
