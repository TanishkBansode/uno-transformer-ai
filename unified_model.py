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
    obs          (153,) — see layout below
    belief_state (54,)  — fraction of each card type seen in the discard pile

    Observation layout:
        obs[ 0: 4]    current colour one-hot                (4)
        obs[ 4:19]    current value  one-hot                (15)
        obs[19:23]    all hand sizes / 108                  (4)
        obs[23:77]    own hand, count 0.5 per copy          (54)
        obs[77:153]   last 4 plays, each 4+15 one-hot       (76)

    Tokenisation  ->  8 tokens, each 27-wide:
        Token 0     : game context  obs[0:23] zero-padded to 27   (1, 27)
        Tokens 1-2  : hand          obs[23:77] -> 2 x 27           (2, 27)
        Tokens 3-5  : history       obs[77:153] (+5 pad) -> 3 x 27 (3, 27)
        Tokens 6-7  : belief        belief[0:54] -> 2 x 27         (2, 27)

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
    num_tokens  : int = 8   # 1 context +2 hand +3 history +2 belief

    @nn.compact
    def __call__(self, obs, belief_state):
        # 1. Tokenise — obs is 153 (23 ctx +54 hand +76 hist), belief is 54
        context_token = jnp.concatenate([obs[:23], jnp.zeros(4)])[jnp.newaxis, :]  # (1, 27)
        hand_tokens   = obs[23:77].reshape(2, 27)                                  # (2, 27)
        hist_padded   = jnp.concatenate([obs[77:153], jnp.zeros(5)])               # 76+5=81
        history_tokens = hist_padded.reshape(3, 27)                                # (3, 27)
        belief_tokens = belief_state.reshape(2, 27)                                # (2, 27)
        tokens = jnp.concatenate(
            [context_token, hand_tokens, history_tokens, belief_tokens], axis=0
        )  # (8, 27)

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
