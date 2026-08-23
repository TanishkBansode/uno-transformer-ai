import flax.linen as nn
import jax.numpy as jnp


class TransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x):
        attn = nn.SelfAttention(num_heads=self.num_heads, qkv_features=self.embed_dim)(x)
        x    = nn.LayerNorm()(x + attn)
        ff   = nn.Dense(self.embed_dim * 4)(x)
        ff   = nn.relu(ff)
        ff   = nn.Dense(self.embed_dim)(ff)
        x    = nn.LayerNorm()(x + ff)
        return x


class UnifiedUnoBrain(nn.Module):
    """
    Actor-Critic Transformer for UNO (PPO).

    Inputs
    ------
    obs          (77,)  — see layout below
    belief_state (54,)  — fraction of each card type seen in the discard pile

    Observation layout:
        obs[ 0: 4]   current colour one-hot                (4)
        obs[ 4:19]   current value  one-hot                (15)
        obs[19:23]   all hand sizes / 108                  (4)
        obs[23:77]   own hand, card-type multi-hot         (54)

    Tokenisation  ->  5 tokens, each 27-wide:
        Token 0     : game context  obs[0:23] zero-padded to 27   (1, 27)
        Tokens 1-2  : hand          obs[23:77] -> 2 x 27           (2, 27)
        Tokens 3-4  : belief        belief[0:54] -> 2 x 27         (2, 27)

    Outputs
    -------
    logits  (61,)   action logits:
                        0-51  play coloured card type (colour*13+value)
                       52-55  play Wild choosing R/G/B/Y
                       56-59  play WildDrawFour choosing R/G/B/Y
                           60 draw
                    Wild colour choice is learned by the policy.
    value   ()      scalar state-value estimate for the critic
    """
    num_actions : int = 61   # 52 coloured types + 8 wild-colour + 1 draw
    embed_dim   : int = 96
    num_heads   : int = 6
    num_layers  : int = 3
    num_tokens  : int = 5

    @nn.compact
    def __call__(self, obs, belief_state):
        # 1. Tokenise
        context_token = jnp.concatenate([obs[:23], jnp.zeros(4)])[jnp.newaxis, :]  # (1, 27)
        hand_tokens   = obs[23:77].reshape(2, 27)                                  # (2, 27)
        belief_tokens = belief_state.reshape(2, 27)                                # (2, 27)
        tokens = jnp.concatenate(
            [context_token, hand_tokens, belief_tokens], axis=0
        )  # (5, 27)

        # 2. Token projection + positional embeddings
        x = nn.Dense(self.embed_dim)(tokens)                                       # (5, embed_dim)
        pos_emb = self.param('pos_emb', nn.initializers.normal(0.02),
                             (self.num_tokens, self.embed_dim))
        x = x + pos_emb

        # 3. Transformer encoder
        x = x[jnp.newaxis, :, :]                                                   # (1, 5, embed_dim)
        for _ in range(self.num_layers):
            x = TransformerBlock(self.embed_dim, self.num_heads)(x)
        x = x[0]                                                                   # (5, embed_dim)

        # 4. Shared trunk (mean-pool over tokens)
        trunk = nn.relu(nn.Dense(self.embed_dim)(x.mean(axis=0)))                  # (embed_dim,)

        # 5. Policy head
        logits = nn.Dense(self.num_actions)(trunk)                                 # (61,)

        # 6. Value head
        value = nn.relu(nn.Dense(self.embed_dim)(trunk))
        value = jnp.squeeze(nn.Dense(1)(value), axis=-1)                           # scalar

        return logits, value
