import jax
import jax.numpy as jnp
import optax
import pickle
import numpy as np
from collections import Counter
from uno_full import UnoGame, Card
from unified_model import UnifiedUnoBrain

# Constants
NUM_EPISODES = 1000
LEARNING_RATE = 1e-3
GAMMA = 0.99
MAX_STEPS = 500

ALL_CARDS = [Card(c, v) for c in Card.COLORS for v in Card.VALUES] + [Card(None, v) for v in Card.WILD_VALUES for _ in range(4)]
CARD_INDEX = {(c.color, c.value): i for i, c in enumerate(ALL_CARDS)}

def get_observation(game, player_idx):
    obs = jnp.zeros(219)
    
    # 0-3: One-hot current color
    if game.current_color in Card.COLORS:
        idx = Card.COLORS.index(game.current_color)
        obs = obs.at[idx].set(1.0)
        
    # 4-18: One-hot current value
    all_values = Card.VALUES + Card.WILD_VALUES
    if game.current_value in all_values:
        idx = 4 + all_values.index(game.current_value)
        obs = obs.at[idx].set(1.0)
        
    # 19-22: Hand sizes
    for i in range(4):
        if i < len(game.players):
            obs = obs.at[19 + i].set(len(game.players[i].hand) / 108.0)
            
    # 23-130: Multi-hot current player hand
    for card in game.players[player_idx].hand:
        idx = CARD_INDEX.get((card.color, card.value))
        if idx is not None:
            obs = obs.at[23 + idx].set(1.0)
        
    return obs

def collect_episode(model, params):
    game = UnoGame(["AI", "P2", "P3", "P4"])
    episode_data = []
    done = False
    steps = 0
    
    while not done and steps < MAX_STEPS:
        steps += 1
        player_idx = game.current_player_idx
        obs = get_observation(game, player_idx)
        belief = jnp.zeros(108)
        
        # Get action distribution
        logits = model.apply({'params': params}, obs, belief)
        probs = jax.nn.softmax(logits)
        action_idx = np.random.choice(len(probs[0]), p=np.array(probs[0]))
        
        try:
            if action_idx == 108:
                result = game.play_turn(player_idx, None)
            else:
                hand = game.players[player_idx].hand
                card_idx = action_idx % len(hand)
                card = hand[card_idx]
                
                chosen_color = None
                if card.color is None:
                    # Pick the color most represented in hand
                    colors = [c.color for c in hand if c.color]
                    chosen_color = Counter(colors).most_common(1)[0][0] if colors else 'Red'

                result = game.play_turn(player_idx, card_idx, chosen_color)
            
            reward = -0.01
            if "winner" in result:
                reward = 1.0 if result["winner"] == "AI" else -1.0
                done = True
        except (ValueError, IndexError) as e:
            reward = -0.1
            done = True
            
        episode_data.append((obs, belief, action_idx, reward))
    return episode_data

def train():
    model = UnifiedUnoBrain()
    key = jax.random.PRNGKey(0)
    dummy_obs = jnp.ones((219,))
    dummy_belief = jnp.ones((108,))
    params = model.init(key, dummy_obs, dummy_belief)['params']
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)

    for ep in range(NUM_EPISODES):
        data = collect_episode(model, params)
        if not data: continue
        obs_list, belief_list, action_list, reward_list = zip(*data)
        
        returns = []
        running_add = 0
        for r in reversed(reward_list):
            running_add = r + GAMMA * running_add
            returns.insert(0, running_add)
        returns = jnp.array(returns)
        
        def loss_fn(params):
            log_probs = []
            for i in range(len(obs_list)):
                logits = model.apply({'params': params}, obs_list[i], belief_list[i])
                log_probs.append(jax.nn.log_softmax(logits[0])[action_list[i]])
            return -jnp.mean(jnp.array(log_probs) * returns)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        if ep % 100 == 0:
            print(f"Episode {ep}, Loss: {loss}")

    with open("uno_brain_final.pkl", "wb") as f:
        pickle.dump(params, f)

if __name__ == "__main__":
    train()
