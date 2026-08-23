"""
PPO self-play trainer for the UNO transformer ("UnifiedUnoBrain").

Upgrades over the previous A2C version:
  * PPO (clipped surrogate) with GAE(lambda) instead of vanilla A2C
  * Policy learns wild-card colour choice (61 actions, no hardcoded heuristic)
  * Legal-move action masking everywhere
  * Snapshot opponent pool (league-style self-play) + uniform-random opponents
  * Periodic evaluation vs random agents and vs frozen snapshots
  * Resumable checkpoints (params + optimiser state + episode counter)
  * Time-limit aware main loop designed for GitHub Actions runners
"""
import argparse
import os
import pickle
import random
import time

import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
import optax

from uno_full import UnoGame, Card
from unified_model import UnifiedUnoBrain

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
LEARNING_RATE       = 3e-4
GAMMA               = 0.99
LAMBDA              = 0.95          # GAE decay
CLIP_EPS            = 0.2           # PPO clip
ENTROPY_COEF        = 0.01          # moderate entropy reg (helps card-game PPO)
VALUE_COEF          = 0.5
GRAD_CLIP_NORM      = 0.5
PPO_EPOCHS          = 3
MINIBATCHES         = 4

EPISODES_PER_UPDATE = 8             # self-play games collected per PPO round
MAX_STEPS           = 750          # safety valve; longer games are stalls (penalised)

WIN_REWARD          = 1.0
LOSS_REWARD         = -1.0
TIMEOUT_REWARD      = -0.5
PLAY_REWARD         = 0.01
DRAW_REWARD         = -0.02

EVAL_INTERVAL       = 200           # episodes between evaluations
EVAL_GAMES_RANDOM   = 50
EVAL_GAMES_SNAPSHOT = 25
SAVE_INTERVAL       = 1000          # episodes between rolling checkpoints
SNAP_INTERVAL       = 200           # episodes between league snapshots
SNAPSHOT_POOL       = 8             # frozen opponents kept in memory

OPP_CURRENT         = 0.60          # prob an opponent seat uses the learning policy
OPP_SNAPSHOT        = 0.25          # prob it uses a frozen snapshot (rest: random agent)

PRINT_INTERVAL      = 50
PLOT_INTERVAL       = 10
WIN_WINDOW          = 100

# ---------------------------------------------------------------------------
# Action space (61)
#   0-51 : play coloured card type  color_idx*13 + value_idx
#  52-55 : play Wild choosing Red/Green/Blue/Yellow
#  56-59 : play WildDrawFour choosing Red/Green/Blue/Yellow
#     60 : draw
# ---------------------------------------------------------------------------
NUM_COLORED_ACTIONS = len(Card.COLORS) * len(Card.VALUES)      # 52
ACT_WILD_BASE       = NUM_COLORED_ACTIONS                      # 52
ACT_WDF_BASE        = ACT_WILD_BASE + len(Card.COLORS)         # 56
ACT_DRAW            = ACT_WDF_BASE + len(Card.COLORS)          # 60
NUM_ACTIONS         = ACT_DRAW + 1                             # 61

ALL_VALUES          = Card.VALUES + Card.WILD_VALUES           # 15
NUM_TYPES           = len(Card.COLORS) * len(Card.VALUES) + len(Card.WILD_VALUES)
OBS_SIZE            = 4 + 15 + 4 + NUM_TYPES                   # 77

COLORED_INDEX       = {(c, v): ci * len(Card.VALUES) + vi
                       for ci, c in enumerate(Card.COLORS)
                       for vi, v in enumerate(Card.VALUES)}
WILD_HAND_INDEX     = {v: NUM_COLORED_ACTIONS + i
                       for i, v in enumerate(Card.WILD_VALUES)}

BATCH_CAP           = EPISODES_PER_UPDATE * 800                # covers MAX_STEPS worst case
MB_SIZE             = BATCH_CAP // MINIBATCHES


# ---------------------------------------------------------------------------
# Observation / belief / mask helpers
# ---------------------------------------------------------------------------
def get_observation(game, player_idx):
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    if game.current_color in Card.COLORS:
        obs[Card.COLORS.index(game.current_color)] = 1.0
    if game.current_value in ALL_VALUES:
        obs[4 + ALL_VALUES.index(game.current_value)] = 1.0
    for i in range(4):
        obs[19 + i] = len(game.players[i].hand) / 108.0
    for card in game.players[player_idx].hand:
        idx = COLORED_INDEX.get((card.color, card.value))
        if idx is None:
            idx = WILD_HAND_INDEX.get(card.value)
        if idx is not None:
            obs[23 + idx] += 1.0
    return obs


def get_belief_state(game):
    """Fraction of each card type that has appeared in the discard pile."""
    belief = np.zeros(NUM_TYPES, dtype=np.float32)
    for card in game.discard_pile:
        idx = COLORED_INDEX.get((card.color, card.value))
        if idx is None:
            idx = WILD_HAND_INDEX.get(card.value)
        if idx is not None:
            belief[idx] += 1.0
    return belief / 108.0


def get_action_mask(game, player_idx):
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for card in game.players[player_idx].hand:
        if not game.is_valid_move(card):
            continue
        if card.color is not None:
            mask[COLORED_INDEX[(card.color, card.value)]] = True
        elif card.value == 'Wild':
            mask[ACT_WILD_BASE:ACT_WDF_BASE] = True
        else:
            mask[ACT_WDF_BASE:ACT_DRAW] = True
    mask[ACT_DRAW] = True
    return mask


def execute_action(game, pid, action_idx):
    """Map an action id onto the game engine. Returns (result, executed_action)."""
    hand = game.players[pid].hand
    if action_idx >= ACT_DRAW:
        return game.play_turn(pid, None), ACT_DRAW
    if action_idx < NUM_COLORED_ACTIONS:
        ci, vi = divmod(action_idx, len(Card.VALUES))
        target_color, target_value = Card.COLORS[ci], Card.VALUES[vi]
        chosen_color = None
    else:
        base = ACT_WILD_BASE if action_idx < ACT_WDF_BASE else ACT_WDF_BASE
        target_color = None
        target_value = 'Wild' if base == ACT_WILD_BASE else 'WildDrawFour'
        chosen_color = Card.COLORS[action_idx - base]
    for i, card in enumerate(hand):
        if card.color == target_color and card.value == target_value:
            return game.play_turn(pid, i, chosen_color), action_idx
    return game.play_turn(pid, None), ACT_DRAW     # unreachable when masked properly


# ---------------------------------------------------------------------------
# Model + JIT helpers
# ---------------------------------------------------------------------------
model = UnifiedUnoBrain()


@jax.jit
def forward(params, obs, belief):
    return model.apply({'params': params}, obs, belief)


@jax.jit
def _batch_apply(params, obs_b, belief_b):
    return jax.vmap(lambda o, b: model.apply({'params': params}, o, b))(obs_b, belief_b)


@jax.jit
def compute_loss(params, obs_b, bel_b, act_b, mask_b,
                 adv_b, ret_b, old_lp_b, w_b):
    """Weighted PPO loss — w_b zeroes out padding rows so batch shape is fixed."""
    logits_b, values_b = _batch_apply(params, obs_b, bel_b)

    masked_logits = jnp.where(mask_b, logits_b, -1e9)
    log_probs     = jax.nn.log_softmax(masked_logits)
    probs         = jax.nn.softmax(masked_logits)

    n       = log_probs.shape[0]
    lp_a    = log_probs[jnp.arange(n), act_b]
    ratio   = jnp.exp(lp_a - old_lp_b)
    adv     = jax.lax.stop_gradient(adv_b)
    clipped = jnp.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
    w_sum   = jnp.sum(w_b) + 1e-8

    pol_loss = -jnp.sum(w_b * jnp.minimum(ratio * adv, clipped * adv)) / w_sum
    val_loss = jnp.sum(w_b * jnp.square(values_b - ret_b)) / w_sum
    entropy  = -jnp.sum(w_b * jnp.sum(probs * log_probs, axis=-1)) / w_sum

    return pol_loss + VALUE_COEF * val_loss - ENTROPY_COEF * entropy


loss_and_grad = jax.jit(jax.value_and_grad(compute_loss))


# ---------------------------------------------------------------------------
# Acting
# ---------------------------------------------------------------------------
def sample_action(params, game, pid):
    """Sample from the policy. Returns (action, obs, bel, mask, logp, value)."""
    obs  = get_observation(game, pid)
    bel  = get_belief_state(game)
    mask = get_action_mask(game, pid)
    logits, value = forward(params, jnp.asarray(obs), jnp.asarray(bel))
    logits = np.array(logits, dtype=np.float64)   # copy: asarray() would be read-only
    logits[~mask] = -1e9
    shifted   = logits - logits.max()
    exp_l     = np.exp(shifted)
    z         = exp_l.sum()
    probs     = exp_l / z
    action    = int(np.random.choice(NUM_ACTIONS, p=probs))
    return action, obs, bel, mask, float(shifted[action] - np.log(z)), float(value)


def random_action(game, pid):
    legal = np.flatnonzero(get_action_mask(game, pid))
    return int(np.random.choice(legal))


# ---------------------------------------------------------------------------
# Episode collection — league self-play
# ---------------------------------------------------------------------------
def collect_episode(learn_params, seat_params):
    """
    seat_params: list of 4 entries. Entry 0 is ignored (seat 0 always uses
                 learn_params); other entries are params or None (= random agent).
    """
    game = UnoGame(["P0", "P1", "P2", "P3"])
    bufs = {i: {'obs': [], 'bel': [], 'act': [], 'mask': [],
                'logp': [], 'val': [], 'rew': []} for i in range(4)}
    done, steps, draws = False, 0, 0
    winner_pid = None

    while not done and steps < MAX_STEPS:
        steps += 1
        pid    = game.current_player_idx
        params = learn_params if pid == 0 else seat_params[pid]

        if params is None:
            action = random_action(game, pid)
            obs  = get_observation(game, pid)
            bel  = get_belief_state(game)
            mask = get_action_mask(game, pid)
            logp, val = 0.0, 0.0
        else:
            action, obs, bel, mask, logp, val = sample_action(params, game, pid)

        result, action = execute_action(game, pid, action)
        step_reward = DRAW_REWARD if action == ACT_DRAW else PLAY_REWARD
        draws += int(action == ACT_DRAW)

        if params is not None:
            # Only transitions taken by a learned policy enter the PPO buffer.
            # (Random-agent moves have no valid old-log-prob.)
            b = bufs[pid]
            b['obs'].append(obs);    b['bel'].append(bel)
            b['act'].append(action); b['mask'].append(mask)
            b['logp'].append(logp);  b['val'].append(val)
            b['rew'].append(step_reward)

        if "winner" in result:
            done = True
            winner_pid = pid
            if bufs[pid]['rew']:                     # random agents keep no buffer
                bufs[pid]['rew'][-1] = WIN_REWARD
            for i in range(4):
                if i != pid and bufs[i]['rew']:
                    bufs[i]['rew'][-1] += LOSS_REWARD

    if winner_pid is None:                       # timeout: everyone penalised
        for i in range(4):
            if bufs[i]['rew']:
                bufs[i]['rew'][-1] += TIMEOUT_REWARD

    # Per-player GAE, then flatten
    all_obs, all_bel, all_act, all_mask = [], [], [], []
    all_adv, all_ret, all_logp = [], [], []
    for i in range(4):
        b = bufs[i]
        n = len(b['rew'])
        if n == 0:
            continue
        vals = np.array(b['val'], dtype=np.float32)
        rews = np.array(b['rew'], dtype=np.float32)
        adv  = np.zeros(n, dtype=np.float32)
        gae  = 0.0
        for t in range(n - 1, -1, -1):
            next_val = vals[t + 1] if t + 1 < n else 0.0
            delta = rews[t] + GAMMA * next_val - vals[t]
            gae   = delta + GAMMA * LAMBDA * gae
            adv[t] = gae
        ret = adv + vals

        all_obs.extend(b['obs']);      all_bel.extend(b['bel'])
        all_act.extend(b['act']);      all_mask.extend(b['mask'])
        all_adv.extend(adv.tolist());  all_ret.extend(ret.tolist())
        all_logp.extend(b['logp'])

    won = 1 if winner_pid == 0 else 0
    return (all_obs, all_bel, all_act, all_mask,
            all_adv, all_ret, all_logp, steps, draws, won)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_vs_random(params, n_games=EVAL_GAMES_RANDOM):
    wins = 0
    for _ in range(n_games):
        game = UnoGame(["P0", "P1", "P2", "P3"])
        done, steps = False, 0
        while not done and steps < MAX_STEPS:
            steps += 1
            pid = game.current_player_idx
            if pid == 0:
                action, _, _, _, _, _ = sample_action(params, game, 0)
            else:
                action = random_action(game, pid)
            result, _ = execute_action(game, pid, action)
            if "winner" in result:
                done = True
                if result["winner"] == "P0":
                    wins += 1
    return wins / max(n_games, 1)


def evaluate_vs_snapshot(params, snap_params, n_games=EVAL_GAMES_SNAPSHOT):
    wins = 0
    for _ in range(n_games):
        game = UnoGame(["P0", "P1", "P2", "P3"])
        done, steps = False, 0
        while not done and steps < MAX_STEPS:
            steps += 1
            pid = game.current_player_idx
            if pid == 0:
                action, _, _, _, _, _ = sample_action(params, game, 0)
            else:
                action, _, _, _, _, _ = sample_action(snap_params, game, pid)
            result, _ = execute_action(game, pid, action)
            if "winner" in result:
                done = True
                if result["winner"] == "P0":
                    wins += 1
    return wins / max(n_games, 1)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def to_numpy_tree(tree):
    return jax.tree_util.tree_map(np.asarray, tree)


def save_checkpoint(path, params, opt_state, episode, best_wr):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({
            'version': 3,
            'params': to_numpy_tree(params),
            'opt_state': to_numpy_tree(opt_state),
            'episode': episode,
            'best_wr': best_wr,
        }, f)
    os.replace(tmp, path)


def load_checkpoint(path):
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    if ckpt.get('version') != 3:
        raise ValueError(f"{path}: incompatible checkpoint (needs version 3)")
    params    = jax.tree_util.tree_map(jnp.asarray, ckpt['params'])
    opt_state = jax.tree_util.tree_map(jnp.asarray, ckpt['opt_state'])
    print(f"Resumed from {path} (episode {ckpt['episode']}, "
          f"best vs-random WR {ckpt['best_wr']:.1%})", flush=True)
    return params, opt_state, ckpt['episode'], ckpt['best_wr']


def init_params(key):
    return model.init(key, jnp.ones((OBS_SIZE,)), jnp.ones((NUM_TYPES,)))['params']


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_plot():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("UNO PPO Training (league self-play)", fontsize=14, fontweight='bold')
    return fig, axes


def update_plot(fig, axes, hist):
    ax_loss, ax_steps, ax_wins, ax_draws = axes.flat
    for ax in axes.flat:
        ax.cla()

    if hist['loss']:
        ax_loss.plot(hist['loss_ep'], hist['loss'], color='royalblue', linewidth=0.8)
    ax_loss.set_title("PPO loss")
    ax_loss.set_xlabel("Episode")
    ax_loss.axhline(0, color='gray', linestyle='--', linewidth=0.5)

    ax_steps.plot(hist['ep'], hist['steps'], color='orange', linewidth=0.8)
    ax_steps.set_title("Episode length")
    ax_steps.set_xlabel("Episode")

    wins = np.array(hist['wins'], dtype=float)
    rolling = [wins[max(0, i - WIN_WINDOW):i + 1].mean() for i in range(len(wins))]
    ax_wins.plot(hist['ep'], rolling, color='green', linewidth=1.0, alpha=0.6,
                 label="Self-play WR (rolling)")
    if hist['eval_rand']:
        ax_wins.plot(hist['eval_ep'], hist['eval_rand'], color='darkgreen',
                     linewidth=1.8, marker='o', markersize=4, label="vs Random WR")
    if hist['eval_snap']:
        ax_wins.plot(hist['eval_ep'], hist['eval_snap'], color='teal',
                     linewidth=1.4, marker='s', markersize=3,
                     label="vs Snapshot WR")
    ax_wins.set_ylim(0, 1)
    ax_wins.axhline(0.25, color='red', linestyle='--', linewidth=0.8,
                    label='Random baseline (25%)')
    ax_wins.set_title("Win rate")
    ax_wins.legend(fontsize=7)

    ax_draws.plot(hist['ep'], hist['draws'], color='purple', linewidth=0.8)
    ax_draws.set_title("Draw-action rate")
    ax_draws.set_xlabel("Episode")
    ax_draws.set_ylim(0, 1)


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------
def pad_to_cap(a, fill_value, dtype=None):
    n = len(a)
    arr = np.asarray(a, dtype=dtype or (a[0].dtype if hasattr(a[0], 'dtype') else None))
    out = np.full((BATCH_CAP,) + arr.shape[1:], fill_value, dtype=arr.dtype)
    out[:n] = arr
    return out


def run_ppo_update(params, opt_state, optimizer, data):
    obs_l, bel_l, act_l, mask_l, adv_l, ret_l, logp_l = data
    n = len(obs_l)
    if n == 0:
        return params, opt_state, float('nan')
    if n > BATCH_CAP:                       # keep the most recent experience
        obs_l, bel_l = obs_l[-BATCH_CAP:], bel_l[-BATCH_CAP:]
        act_l, mask_l = act_l[-BATCH_CAP:], mask_l[-BATCH_CAP:]
        adv_l, ret_l, logp_l = adv_l[-BATCH_CAP:], ret_l[-BATCH_CAP:], logp_l[-BATCH_CAP:]
        n = BATCH_CAP

    adv = np.array(adv_l, dtype=np.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    obs_p  = pad_to_cap(obs_l,  0.0, np.float32)
    bel_p  = pad_to_cap(bel_l,  0.0, np.float32)
    act_p  = pad_to_cap(act_l,  0,   np.int32)
    mask_p = pad_to_cap(mask_l, False, bool)
    adv_p  = np.zeros(BATCH_CAP, dtype=np.float32); adv_p[:n] = adv
    ret_p  = pad_to_cap(ret_l,  0.0, np.float32)
    lp_p   = pad_to_cap(logp_l, 0.0, np.float32)

    idx = np.random.permutation(n)
    last_loss = float('nan')
    for _ in range(PPO_EPOCHS):
        np.random.shuffle(idx)
        for s in range(MINIBATCHES):
            start = s * MB_SIZE
            if start >= n:
                break
            take  = min(MB_SIZE, n - start)
            mb_i  = np.zeros(MB_SIZE, dtype=np.int64)
            w_mb  = np.zeros(MB_SIZE, dtype=np.float32)
            mb_i[:take]  = idx[start:start + take]
            w_mb[:take]  = 1.0

            loss_val, grads = loss_and_grad(
                params,
                obs_p[mb_i], bel_p[mb_i], act_p[mb_i], mask_p[mb_i],
                adv_p[mb_i], ret_p[mb_i], lp_p[mb_i], w_mb)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            last_loss = float(loss_val)
    return params, opt_state, last_loss


def sample_seat_params(learn_params, snapshots):
    r = random.random()
    if r < OPP_CURRENT or not snapshots:
        return learn_params
    if r < OPP_CURRENT + OPP_SNAPSHOT:
        return snapshots[random.randrange(len(snapshots))]
    return None                                            # uniform random agent


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    key     = jax.random.PRNGKey(args.seed)
    params  = init_params(key)
    optimizer = optax.chain(optax.clip_by_global_norm(GRAD_CLIP_NORM),
                            optax.adam(LEARNING_RATE))
    opt_state = optimizer.init(params)
    ep_done   = 0
    best_wr   = -1.0

    if args.resume and os.path.exists(args.checkpoint_in):
        try:
            params, opt_state, ep_done, best_wr = load_checkpoint(args.checkpoint_in)
        except (ValueError, EOFError, pickle.UnpicklingError) as e:
            print(f"WARNING: could not resume ({e}); starting fresh", flush=True)
            ep_done, best_wr = 0, -1.0

    snapshots = []
    hist = {'ep': [], 'steps': [], 'wins': [], 'draws': [],
            'loss': [], 'loss_ep': [], 'eval_ep': [],
            'eval_rand': [], 'eval_snap': []}

    interactive = 'agg' not in matplotlib.get_backend().lower()
    fig = axes = None
    if not args.no_plot:
        fig, axes = make_plot()
        if interactive:
            plt.ion()
            plt.show()

    deadline = time.time() + args.time_limit_minutes * 60 if args.time_limit_minutes else None
    t_start  = time.time()
    buf = {'obs': [], 'bel': [], 'act': [], 'mask': [],
           'adv': [], 'ret': [], 'logp': []}

    try:
        while ep_done < args.episodes:
            if deadline and time.time() > deadline:
                print("Time limit reached — saving checkpoint.", flush=True)
                break

            seat_params = [None] * 4
            for i in range(1, 4):
                seat_params[i] = sample_seat_params(params, snapshots)

            rollout = collect_episode(params, seat_params)
            (obs_l, bel_l, act_l, mask_l, adv_l, ret_l,
             logp_l, steps, draws, won) = rollout

            buf['obs'].extend(obs_l);   buf['bel'].extend(bel_l)
            buf['act'].extend(act_l);   buf['mask'].extend(mask_l)
            buf['adv'].extend(adv_l);   buf['ret'].extend(ret_l)
            buf['logp'].extend(logp_l)

            ep_done += 1
            hist['ep'].append(ep_done)
            hist['steps'].append(steps)
            hist['wins'].append(won)
            hist['draws'].append(draws / max(steps, 1))

            if ep_done % EPISODES_PER_UPDATE == 0:
                params, opt_state, loss_val = run_ppo_update(
                params, opt_state, optimizer,
                (buf['obs'], buf['bel'], buf['act'], buf['mask'],
                 buf['adv'], buf['ret'], buf['logp']))
                buf = {k: [] for k in buf}
                hist['loss'].append(loss_val)
                hist['loss_ep'].append(ep_done)

            if ep_done % SNAP_INTERVAL == 0:
                snapshots.append(to_numpy_tree(params))
                if len(snapshots) > SNAPSHOT_POOL:
                    snapshots.pop(0)

            if ep_done % EVAL_INTERVAL == 0:
                wr_rand = evaluate_vs_random(params)
                wr_snap = (evaluate_vs_snapshot(params, snapshots[-1])
                           if snapshots else float('nan'))
                hist['eval_ep'].append(ep_done)
                hist['eval_rand'].append(wr_rand)
                hist['eval_snap'].append(wr_snap)

                self_wr = np.mean(hist['wins'][-WIN_WINDOW:])
                rate    = ep_done / max(time.time() - t_start, 1e-9)
                snap_s  = f"{wr_snap:.1%}" if not np.isnan(wr_snap) else "  n/a"
                last_loss = hist['loss'][-1] if hist['loss'] else float('nan')
                print(f"Ep {ep_done:6d} | Loss {last_loss:+.4f} "
                      f"| Self-WR {self_wr:.1%} | vs Random {wr_rand:.1%} "
                      f"| vs Snap {snap_s} | Steps {steps:3d} "
                      f"| {rate:.0f} ep/s", flush=True)

                if wr_rand > best_wr:
                    best_wr = wr_rand
                    save_checkpoint(args.best_out, params, opt_state,
                                    ep_done, best_wr)
                    print(f"  New best vs-random WR {best_wr:.1%} "
                          f"-> saved {args.best_out}", flush=True)

            if not args.no_plot and ep_done % PLOT_INTERVAL == 0 and hist['ep']:
                update_plot(fig, axes, hist)
                if ep_done % EVAL_INTERVAL == 0:
                    fig.savefig(args.curves, dpi=150)
                if interactive:
                    plt.pause(0.001)

            if ep_done % SAVE_INTERVAL == 0:
                save_checkpoint(args.checkpoint_out, params, opt_state,
                                ep_done, best_wr)
    except KeyboardInterrupt:
        print("\nInterrupted — saving checkpoint.", flush=True)
    finally:
        save_checkpoint(args.checkpoint_out, params, opt_state, ep_done, best_wr)
        if fig is not None:
            update_plot(fig, axes, hist)
            fig.savefig(args.curves, dpi=150)
        if not args.no_plot and interactive:
            plt.ioff()
            plt.show()

    print(f"Done. Episodes this session: total={ep_done}, "
          f"best vs-random WR={best_wr:.1%}. Saved {args.checkpoint_out}",
          flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="PPO self-play UNO trainer")
    p.add_argument('--episodes', type=int, default=100_000_000,
                   help='total episode cap (across resumes)')
    p.add_argument('--time-limit-minutes', type=float, default=None,
                   help='stop training gracefully after this many minutes')
    p.add_argument('--resume', action='store_true',
                   help='resume from --checkpoint-in if it exists')
    p.add_argument('--checkpoint-in', default='uno_brain_final.pkl')
    p.add_argument('--checkpoint-out', default='uno_brain_final.pkl')
    p.add_argument('--best-out', default='uno_brain_best.pkl')
    p.add_argument('--curves', default='training_curves.png')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval-interval', type=int, default=None,
                   help='override EVAL_INTERVAL')
    p.add_argument('--eval-games', type=int, default=None,
                   help='override number of evaluation games')
    p.add_argument('--save-interval', type=int, default=None,
                   help='override SAVE_INTERVAL')
    p.add_argument('--no-plot', action='store_true')
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.eval_interval:
        EVAL_INTERVAL = SNAP_INTERVAL = args.eval_interval
    if args.eval_games:
        EVAL_GAMES_RANDOM = args.eval_games
        EVAL_GAMES_SNAPSHOT = max(args.eval_games // 2, 10)
    if args.save_interval:
        SAVE_INTERVAL = args.save_interval
    train(args)
