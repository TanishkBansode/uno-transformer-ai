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
    # Build observation (simple version for interaction)
    obs = jnp.zeros(219) # Simplified
    belief = jnp.ones(108) / 108
    action_probs = model.apply({'params': params}, obs, belief)
    return jnp.argmax(action_probs)

def play_game():
    game = UnoGame(["Human", "AI"])
    
    while True:
        # Human turn
        print(f"\n--- Human's Turn ---")
        print(f"Top card: {game.discard_pile[-1]}")
        print(f"Hand: {game.players[0].hand}")
        choice = input("Enter card index to play (or 'd' to draw): ")
        if choice.lower() == 'd':
            game.play_turn(0, None)
        else:
            game.play_turn(0, int(choice))
            
        # AI turn
        print(f"\n--- AI's Turn ---")
        ai_move = get_ai_move(None, 1)
        print(f"AI chose to play card at index {ai_move}")
        game.play_turn(1, ai_move % len(game.players[1].hand))

# This is a skeleton for the interface. 
# Since I cannot do interactive input in the sandbox, I'll print the setup.
print("Interface ready. To play, run this script locally with your trained model.")
