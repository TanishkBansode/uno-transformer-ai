import flax.linen as nn
import jax.numpy as jnp
import jax

class UnifiedUnoBrain(nn.Module):
    num_actions: int = 20
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2

    @nn.compact
    def __call__(self, obs, belief_state):
        # 1. Combine observation and belief state
        x = jnp.concatenate([obs, belief_state], axis=-1)
        
        # 2. Embed and reshape to (batch, sequence_length, embed_dim)
        # We treat the input as a single sequence of length 1
        x = nn.Dense(self.embed_dim)(x)
        x = x[jnp.newaxis, jnp.newaxis, :] 
        
        # 3. Transformer Encoder Layers
        for _ in range(self.num_layers):
            attn = nn.SelfAttention(num_heads=self.num_heads)(x)
            x = nn.LayerNorm()(x + attn)
            ff = nn.Dense(self.embed_dim * 4)(x)
            ff = nn.relu(ff)
            ff = nn.Dense(self.embed_dim)(ff)
            x = nn.LayerNorm()(x + ff)
        
        # 4. Flatten and output head
        x = x.reshape((x.shape[0], -1))
        logits = nn.Dense(self.num_actions)(x)
        return nn.softmax(logits)

# Test the unified model
model = UnifiedUnoBrain()
dummy_obs = jnp.ones((219,))
dummy_belief = jnp.ones((108,))
variables = model.init(jax.random.PRNGKey(0), dummy_obs, dummy_belief)
output = model.apply(variables, dummy_obs, dummy_belief)

print("Unified Brain initialized.")
print("Output shape:", output.shape)
