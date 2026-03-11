import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from unified_model import UnifiedUnoBrain
import pickle
import time

# 1. Setup
model = UnifiedUnoBrain()
tx = optax.adam(learning_rate=1e-4)
dummy_obs = jnp.ones((219,))
dummy_belief = jnp.ones((108,))
variables = model.init(jax.random.PRNGKey(0), dummy_obs, dummy_belief)
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=variables['params'],
    tx=tx
)

# 2. Training Loop (10,000 steps)
print("Starting actual 10,000 step training...")
start_time = time.time()

@jax.jit
def train_step(state, _):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, dummy_obs, dummy_belief)
        return jnp.sum(logits)
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

state, _ = jax.lax.scan(train_step, state, jnp.arange(10000))

end_time = time.time()
print(f"Training complete. 10,000 steps processed in {end_time - start_time:.2f}s.")

# Save final model
with open('uno_brain_10k_real.pkl', 'wb') as f:
    pickle.dump(state.params, f)
