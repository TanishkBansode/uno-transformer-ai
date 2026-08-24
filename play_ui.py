"""
UNO vs Transformer — play in your browser.

    python play_ui.py [--model uno_brain_best.pkl] [--port 5000]
    open http://localhost:5000

Elimination mode: keep playing until 3 players empty their hands.
The last player holding cards loses — everyone else is ranked.
Bots play greedily; their moves are animated slowly so it feels human.
"""
import argparse
import os
import pickle
import random
import threading

import numpy as np
from flask import Flask, jsonify, render_template_string, request

import jax
import jax.numpy as jnp

from uno_full import UnoGame, Card
from train import (ACT_DRAW, ACT_WDF_BASE, ACT_WILD_BASE,
                   execute_action, forward,
                   get_action_mask, get_belief_state, get_observation)

DEFAULT_MODEL_PATHS = ["uno_brain_best.pkl", "uno_brain_final.pkl"]


def load_params(path):
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    if ckpt.get("version") != 3:
        raise ValueError(f"{path}: not a v3 checkpoint")
    return jax.tree_util.tree_map(jnp.asarray, ckpt["params"])


def find_model(cli_path):
    candidates = ([cli_path] if cli_path else []) + DEFAULT_MODEL_PATHS
    for p in candidates:
        if p and os.path.exists(p):
            print(f"Loaded model: {p}")
            return load_params(p)
    raise SystemExit(
        "\nNo model checkpoint found.\n"
        "Download uno_brain_best.pkl from the GitHub Actions run\n"
        "(Artifacts) into this folder, or pass --model PATH."
    )


class Session:
    def __init__(self, params):
        self.params = params
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.game = UnoGame(["You", "Bot West", "Bot North", "Bot East"],
                            elimination_mode=True)
        self.events = ["New game — first to empty wins a podium spot, last holding cards loses. Good luck!"]
        self.move_count = 0

    # ---------------- helpers
    @staticmethod
    def _card_dict(card, idx=None):
        return {"color": card.color, "value": card.value, "idx": idx}

    def _describe(self, pid, action_id, drawn_cards=None):
        name = self.game.players[pid].name
        if action_id >= ACT_DRAW:
            n = drawn_cards or 1
            return f"{name} draws {n} card{'s' if n != 1 else ''}"
        if action_id < ACT_WILD_BASE:
            ci, vi = divmod(action_id, len(Card.VALUES))
            return f"{name} plays {Card.COLORS[ci]} {Card.VALUES[vi]}"
        base = ACT_WILD_BASE if action_id < ACT_WDF_BASE else ACT_WDF_BASE
        val = "Wild" if base == ACT_WILD_BASE else "Wild Draw Four"
        col = Card.COLORS[action_id - base]
        return f"{name} plays {val} → {col}"

    def _collect_bot_moves(self, max_batch=10):
        """Run bots in a small chunk — until next human turn or one elimination.
        Keeps animations short; frontend polls /api/tick while spectating."""
        moves = []
        guard = 0
        while guard < 60 and len(moves) < max_batch:
            guard += 1
            if len(self.game.ranking) >= 3:
                break
            if self.game.current_player_idx == 0:
                if 0 in self.game.finished:
                    self.game.current_player_idx = self.game._next_active(0, 1)
                    continue
                break
            pid = self.game.current_player_idx
            if pid in self.game.finished:
                self.game.current_player_idx = self.game._next_active(pid, 1)
                continue
            before_counts = [len(p.hand) for p in self.game.players]
            logits, _ = forward(self.params,
                                get_observation(self.game, pid),
                                get_belief_state(self.game))
            logits = np.array(logits)
            mask = get_action_mask(self.game, pid)
            logits[~mask] = -1e9
            action = int(np.argmax(logits))
            result, executed = execute_action(self.game, pid, action)
            after_counts = [len(p.hand) for p in self.game.players]
            drew = executed >= ACT_DRAW
            drawn = 0
            if drew:
                drawn = after_counts[pid] - before_counts[pid]
                if drawn <= 0:
                    drawn = 1
            victim_draw = 0
            victim_idx = None
            if not drew:
                for i in range(len(after_counts)):
                    if i != pid and after_counts[i] > before_counts[i]:
                        victim_draw = after_counts[i] - before_counts[i]
                        victim_idx = i
                        break
            moves.append({
                "pid": pid,
                "name": self.game.players[pid].name,
                "action": int(executed),
                "drew": bool(drew),
                "drawn": int(drawn),
                "victim": victim_idx,
                "victim_draw": int(victim_draw),
                "text": self._describe(pid, executed, drawn if drew else victim_draw),
                "counts": after_counts,
                "top": self._card_dict(self.game.discard_pile[-1]),
                "current_color": self.game.current_color,
                "direction": self.game.direction,
                "ranking": list(self.game.ranking),
                "finished": list(self.game.finished),
            })
            if result.get("finished"):
                moves[-1]["finished_game"] = True
                moves[-1]["ranking"] = result["ranking"]
                moves[-1]["loser"] = result["loser"]
                break
            if result.get("eliminated"):
                moves[-1]["eliminated"] = result["eliminated"]
                # stop chunk here so UI can celebrate the podium before continuing
                break
        return moves

    def state_json(self):
        g = self.game
        # human may be finished — still show empty hand
        me = g.players[0]
        is_finished = len(g.ranking) >= 3
        loser = None
        if is_finished:
            # loser is the one not in ranking
            all_names = [p.name for p in g.players]
            loser = next((n for n in all_names if n not in g.ranking), None)
        return {
            "your_turn": (not is_finished and g.current_player_idx == 0
                          and 0 not in g.finished),
            "spectating": 0 in g.finished,
            "hand": [self._card_dict(c, i) for i, c in enumerate(me.hand)],
            "legal": [g.is_valid_move(c) for c in me.hand],
            "top": self._card_dict(g.discard_pile[-1]),
            "current_color": g.current_color,
            "direction": g.direction,
            "deck_count": len(g.deck.cards),
            "discard_count": len(g.discard_pile),
            "opponents": [
                {"name": g.players[i].name,
                 "count": len(g.players[i].hand),
                 "finished": i in g.finished,
                 "rank": (g.ranking.index(g.players[i].name) + 1)
                         if g.players[i].name in g.ranking else None}
                for i in (1, 2, 3)
            ],
            "you": {"name": me.name, "count": len(me.hand),
                    "finished": 0 in g.finished,
                    "rank": (g.ranking.index(me.name) + 1)
                            if me.name in g.ranking else None},
            "ranking": list(g.ranking),
            "loser": loser,
            "finished": is_finished,
            "active_idx": g.current_player_idx,
            "events": self.events[-10:],
        }

    # ---------------- API
    def api_new(self):
        with self.lock:
            self.reset()
            s = self.state_json()
            s["bot_moves"] = []
            return s

    def api_play(self, idx, color):
        with self.lock:
            if len(self.game.ranking) >= 3:
                return {"error": "Game over — start a new game.", **self.state_json()}
            if self.game.current_player_idx != 0:
                return {"error": "Not your turn.", **self.state_json()}
            if 0 in self.game.finished:
                return {"error": "You're already finished — spectating.", **self.state_json()}
            hand = self.game.players[0].hand
            if not isinstance(idx, int) or idx < 0 or idx >= len(hand):
                return {"error": "Bad card index.", **self.state_json()}
            card = hand[idx]
            if not self.game.is_valid_move(card):
                return {"error": "That card doesn't match.", **self.state_json()}
            chosen = color if card.color is None else None
            if card.color is None and chosen not in Card.COLORS:
                return {"error": "Pick a colour for the wild card.", **self.state_json()}
            result = self.game.play_turn(0, idx, chosen)
            shown = f"{card.color} {card.value}" if card.color else f"{card.value} → {chosen}"
            self.events.append(f"You played {shown}")
            self.move_count += 1
            bot_moves = []
            if result.get("finished"):
                self.events.append(f"Game over — {result['loser']} loses!")
            elif result.get("eliminated"):
                self.events.append(f"{result['eliminated']} finished #{len(self.game.ranking)}!")
                bot_moves = self._collect_bot_moves()
            else:
                bot_moves = self._collect_bot_moves()
            self.move_count += len(bot_moves)
            # collect bot texts into events too (for log fallback)
            for m in bot_moves:
                self.events.append(m["text"])
                if m.get("eliminated"):
                    self.events.append(f"{m['eliminated']} finished!")
                if m.get("finished_game"):
                    self.events.append(f"Game over — {m['loser']} loses!")
            s = self.state_json()
            s["bot_moves"] = bot_moves
            s["human_result"] = {k: v for k, v in result.items()
                                 if k in ("eliminated", "finished", "ranking", "loser")}
            return s

    def api_draw(self):
        with self.lock:
            if len(self.game.ranking) >= 3:
                return {"error": "Game over.", **self.state_json()}
            if self.game.current_player_idx != 0:
                return {"error": "Not your turn.", **self.state_json()}
            if 0 in self.game.finished:
                return {"error": "You're finished.", **self.state_json()}
            before = len(self.game.players[0].hand)
            self.game.play_turn(0, None)
            after = len(self.game.players[0].hand)
            drew = max(after - before, 1)
            self.events.append(f"You draw {drew} card{'s' if drew != 1 else ''} (turn passes)")
            self.move_count += 1
            bot_moves = self._collect_bot_moves()
            self.move_count += len(bot_moves)
            for m in bot_moves:
                self.events.append(m["text"])
                if m.get("eliminated"):
                    self.events.append(f"{m['eliminated']} finished!")
                if m.get("finished_game"):
                    self.events.append(f"Game over — {m['loser']} loses!")
            s = self.state_json()
            s["bot_moves"] = bot_moves
            return s

    def api_tick(self):
        """Advance bots while human is spectating. Called repeatedly by frontend."""
        with self.lock:
            if len(self.game.ranking) >= 3:
                return self.state_json()
            if 0 not in self.game.finished:
                return {"error": "Not spectating.", **self.state_json()}
            bot_moves = self._collect_bot_moves(max_batch=30)
            self.move_count += len(bot_moves)
            for m in bot_moves:
                self.events.append(m["text"])
                if m.get("eliminated"):
                    self.events.append(f"{m['eliminated']} finished!")
                if m.get("finished_game"):
                    self.events.append(f"Game over — {m['loser']} loses!")
            # safety: weak random policy can stall for hundreds of moves — force finish
            if not self.state_json()["finished"] and (self.move_count > 700 or len(self.events) > 700):
                counts = [(len(p.hand), p.name, i) for i, p in enumerate(self.game.players) if i not in self.game.finished]
                counts.sort()
                for _, name, idx in counts[:-1]:
                    if idx not in self.game.finished:
                        self.game.finished.append(idx)
                        self.game.ranking.append(name)
                self.events.append("Stall protection — ranking by fewest cards.")
                s = self.state_json()
                s["bot_moves"] = bot_moves
                return s
            s = self.state_json()
            s["bot_moves"] = bot_moves
            return s


session = None
app = Flask(__name__)


@app.post("/api/new")
def route_new():
    return jsonify(session.api_new())


@app.get("/api/state")
def route_state():
    return jsonify(session.state_json())


@app.post("/api/play")
def route_play():
    data = request.get_json(force=True) or {}
    out = session.api_play(data.get("idx"), (data.get("color") or "").title() or None)
    return jsonify(out)


@app.post("/api/draw")
def route_draw():
    return jsonify(session.api_draw())


@app.post("/api/tick")
def route_tick():
    return jsonify(session.api_tick())


PAGE_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>UNO · Transformer</title>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@600;800&family=Space+Grotesk:wght@700&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0}
:root{--red:#ef4444;--blue:#3b82f6;--green:#22c55e;--yellow:#eab308;--ink:#0f172a}
body{font-family:'Outfit',system-ui,sans-serif;background:#060d1a;color:#e2e8f0;min-height:100vh;overflow-x:hidden}
.bg{position:fixed;inset:0;z-index:-2;background:
  radial-gradient(900px 600px at 20% -10%,#1e3a8a55,transparent),
  radial-gradient(800px 500px at 95% 10%,#7c3aed33,transparent),
  radial-gradient(700px 400px at 50% 115%,#0e749055,transparent),
  linear-gradient(180deg,#070d1d,#0a1430 45%,#060d1a)}
.grid{position:fixed;inset:0;z-index:-1;opacity:.07;background-image:
  linear-gradient(rgba(255,255,255,.6) 1px,transparent 1px),
  linear-gradient(90deg,rgba(255,255,255,.6) 1px,transparent 1px);
  background-size:42px 42px;mask:radial-gradient(800px 500px at 50% 40%,black,transparent 80%)}
header{max-width:1100px;margin:14px auto 10px;padding:0 14px;display:flex;align-items:center;justify-content:space-between;gap:12px}
.brand{display:flex;align-items:center;gap:10px}
.logo{width:38px;height:38px;border-radius:12px;background:linear-gradient(135deg,#60a5fa,#a78bfa);display:grid;place-items:center;font-weight:800;color:#0b1020;box-shadow:0 8px 18px rgba(96,165,250,.35)}
.brand h1{font-family:'Space Grotesk',sans-serif;font-size:18px;line-height:1}
.brand p{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:#93a7c9}
.hdr-actions{display:flex;gap:10px}
.btn{border:none;border-radius:999px;padding:10px 16px;font-weight:700;font-size:13px;cursor:pointer;transition:.15s;letter-spacing:.02em}
.btn-primary{background:linear-gradient(135deg,#38bdf8,#6366f1);color:#06101f;box-shadow:0 8px 18px rgba(56,189,248,.35)}
.btn-ghost{background:rgba(255,255,255,.08);color:#e2e8f0;border:1px solid rgba(255,255,255,.12);backdrop-filter:blur(8px)}
.btn:disabled{opacity:.38;cursor:not-allowed;transform:none!important}
.btn:active{transform:translateY(1px)}
.wrap{max-width:1100px;margin:0 auto;padding:0 14px 22px;display:flex;flex-direction:column;align-items:center;gap:14px}
/* opponents */
.opp-row{display:flex;gap:14px;justify-content:center;flex-wrap:wrap}
.seat{position:relative;min-width:148px;display:flex;align-items:center;gap:10px;
  background:linear-gradient(180deg,rgba(255,255,255,.09),rgba(255,255,255,.04));
  border:1px solid rgba(255,255,255,.12);border-radius:16px;padding:10px 12px;
  backdrop-filter:blur(10px);box-shadow:0 10px 24px rgba(0,0,0,.25);transition:.25s}
.seat.active{border-color:rgba(56,189,248,.9);box-shadow:0 0 0 2px rgba(56,189,248,.35),0 16px 32px rgba(0,0,0,.35)}
.seat.finished{opacity:.55;filter:saturate(.6)}
.avatar{width:40px;height:40px;border-radius:50%;display:grid;place-items:center;font-weight:800;color:#fff;flex-shrink:0;
  background:linear-gradient(135deg,#334155,#1e293b);border:2px solid rgba(255,255,255,.12);font-size:13px}
.seat.active .avatar{border-color:#38bdf8;box-shadow:0 0 14px rgba(56,189,248,.6)}
.meta .name{font-weight:700;font-size:13px}
.meta .sub{font-size:12px;color:#94a3b8;display:flex;align-items:center;gap:6px}
.badge{font-size:11px;font-weight:800;padding:3px 7px;border-radius:999px;background:rgba(255,255,255,.12)}
.badge.rank1{background:linear-gradient(135deg,#facc15,#f59e0b);color:#422006}
.badge.loser{background:#ef4444;color:#fff}
.card-backs{display:flex;margin-left:2px}
.card-backs i{width:18px;height:26px;border-radius:4px;border:1px solid rgba(255,255,255,.7);
  background:repeating-linear-gradient(45deg,#1e3a8a,#1e3a8a 4px,#172554 4px,#172554 8px);
  margin-left:-8px;box-shadow:0 2px 6px rgba(0,0,0,.35)}
/* table */
.table{position:relative;width:min(680px,100%);height:300px;display:grid;place-items:center;margin:6px 0}
.felt{position:absolute;inset:0;border-radius:36px;
  background:
    radial-gradient(700px 260px at 50% 38%,rgba(34,197,94,.18),transparent 60%),
    radial-gradient(500px 220px at 50% 50%,rgba(255,255,255,.06),transparent 70%),
    linear-gradient(180deg,#0f4a2a,#0b331d 55%,#082616);
  border:1px solid rgba(34,197,94,.22);box-shadow:inset 0 0 0 1px rgba(255,255,255,.06),inset 0 30px 60px rgba(0,0,0,.35),0 18px 40px rgba(0,0,0,.45)}
.wood{position:absolute;inset:-10px;border-radius:42px;z-index:-1;
  background:linear-gradient(180deg,#2a1a0e,#1a120a);border:1px solid rgba(255,255,255,.08)}
.center{display:flex;align-items:center;gap:26px;position:relative;z-index:1}
.pile{width:92px;height:134px;border-radius:16px;display:grid;place-items:center;
  font-weight:800;font-size:28px;color:#fff;text-shadow:0 2px 8px rgba(0,0,0,.55);
  border:2.5px solid rgba(255,255,255,.88);box-shadow:0 10px 24px rgba(0,0,0,.45);position:relative;overflow:hidden}
.pile small{position:absolute;top:7px;left:9px;font-size:11px;letter-spacing:.06em;opacity:.9}
.deck{cursor:pointer;background:
  repeating-linear-gradient(45deg,#1e40af 0 8px,#1e3a8a 8px 16px);
  transition:.15s}
.deck:hover{transform:translateY(-2px) rotate(-1deg)}
.deck-count{position:absolute;bottom:-18px;left:50%;transform:translateX(-50%);font-size:11px;color:#9fb2c8;white-space:nowrap}
.meta-col{display:flex;flex-direction:column;align-items:center;gap:10px;min-width:110px}
.color-orb{width:54px;height:54px;border-radius:50%;border:3px solid #fff;box-shadow:0 0 18px rgba(255,255,255,.25),0 8px 18px rgba(0,0,0,.4);transition:.4s}
.color-label{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:#cbd5e1}
.dir{width:42px;height:42px;border-radius:50%;display:grid;place-items:center;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.12);font-size:18px;transition:transform .6s}
.dir.rev{transform:rotate(180deg)}
.discard-hint{font-size:11px;color:#7ea0b8}
/* log */
.log{width:min(680px,100%);min-height:56px;max-height:84px;overflow:auto;
  background:rgba(8,14,30,.72);border:1px solid rgba(255,255,255,.08);border-radius:14px;
  padding:10px 12px;font-size:13px;line-height:1.55;color:#b8c7dd;backdrop-filter:blur(8px)}
.log::-webkit-scrollbar{height:6px;width:6px}
/* hand */
.hand-wrap{width:min(1020px,100%);background:linear-gradient(180deg,rgba(255,255,255,.06),rgba(255,255,255,.02));
  border:1px solid rgba(255,255,255,.08);border-radius:22px;padding:16px 12px 14px;backdrop-filter:blur(10px)}
.hand-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;padding:0 4px}
.hand-head h3{font-size:13px;letter-spacing:.14em;text-transform:uppercase;color:#93a7c9}
.turn-chip{font-size:12px;font-weight:800;padding:6px 10px;border-radius:999px;background:rgba(34,197,94,.15);color:#86efac;border:1px solid rgba(34,197,94,.25)}
.turn-chip.wait{background:rgba(148,163,184,.12);color:#cbd5e1;border-color:rgba(255,255,255,.1)}
#hand{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;align-items:flex-end;min-height:132px}
.card{width:84px;height:124px;border-radius:16px;border:2.5px solid rgba(255,255,255,.9);
  display:flex;align-items:center;justify-content:center;flex-direction:column;
  font-weight:800;font-size:26px;color:#fff;text-shadow:0 2px 7px rgba(0,0,0,.6);
  box-shadow:0 8px 18px rgba(0,0,0,.35),inset 0 1px 0 rgba(255,255,255,.25);
  position:relative;overflow:hidden;transition:transform .18s, box-shadow .18s, filter .18s;user-select:none}
.card small{position:absolute;top:7px;left:9px;font-size:11px;font-weight:800;letter-spacing:.02em}
.card .oval{position:absolute;inset:10px 8px;background:rgba(255,255,255,.92);border-radius:50% / 42%;
  transform:rotate(-12deg);opacity:.92;z-index:0}
.card .val{position:relative;z-index:1}
.card.playable{cursor:pointer}
.card.playable:hover{transform:translateY(-14px) rotate(-1deg) scale(1.04);box-shadow:0 18px 28px rgba(0,0,0,.5)}
.card.dead{filter:saturate(.55) brightness(.85);opacity:.5}
.card.selected{transform:translateY(-14px) scale(1.04);box-shadow:0 0 0 3px #38bdf8,0 18px 28px rgba(0,0,0,.5)}
.wildbg{background:conic-gradient(from 0deg,var(--red) 0 25%,var(--yellow) 25% 50%,var(--green) 50% 75%,var(--blue) 75% 100%)!important}
.fly{position:fixed;z-index:30;pointer-events:none;transition:all .62s cubic-bezier(.22,1,.36,1)}
/* modal */
.modal{position:fixed;inset:0;display:none;place-items:center;background:rgba(3,7,18,.66);backdrop-filter:blur(8px);z-index:20}
.modal.open{display:grid}
.modal-card{background:linear-gradient(180deg,#141f3a,#0e1730);border:1px solid rgba(255,255,255,.12);
  border-radius:20px;padding:20px 18px;min-width:320px;box-shadow:0 24px 60px rgba(0,0,0,.55);text-align:center}
.modal-card h3{margin-bottom:14px}
.crow{display:flex;gap:12px;justify-content:center}
.csel{width:56px;height:56px;border-radius:50%;cursor:pointer;border:3px solid #fff;box-shadow:0 8px 18px rgba(0,0,0,.35);transition:transform .12s}
.csel:hover{transform:scale(1.08)}
/* ranking overlay */
.overlay{position:fixed;inset:0;display:none;place-items:center;background:rgba(3,7,18,.78);backdrop-filter:blur(10px);z-index:25}
.overlay.open{display:grid}
.podium{background:linear-gradient(180deg,#131f3d,#0c142a);border:1px solid rgba(255,255,255,.12);border-radius:22px;
  padding:22px 18px;min-width:min(520px,92vw);box-shadow:0 24px 60px rgba(0,0,0,.6);text-align:center}
.rank-row{display:flex;align-items:center;gap:12px;padding:10px 12px;border-radius:14px;background:rgba(255,255,255,.05);margin:8px 0}
.rank-num{width:36px;height:36px;border-radius:50%;display:grid;place-items:center;font-weight:800;color:#fff;flex-shrink:0}
.r1{background:linear-gradient(135deg,#facc15,#eab308);color:#422006}
.r2{background:linear-gradient(135deg,#e5e7eb,#94a3b8);color:#1e293b}
.r3{background:linear-gradient(135deg,#f97316,#fb923c);color:#431407}
.rL{background:#ef4444}
.hidden{display:none!important}
@media(max-width:700px){
  .table{height:260px}.pile{width:78px;height:114px;font-size:24px}.card{width:72px;height:108px;font-size:22px}
  .seat{min-width:124px}
}
</style></head><body>
<div class="bg"></div><div class="grid"></div>
<header>
  <div class="brand">
    <div class="logo">U</div>
    <div><h1>UNO · Transformer</h1><p>Elimination · last standing loses</p></div>
  </div>
  <div class="hdr-actions">
    <button class="btn btn-ghost" onclick="apiNew()">↻ New game</button>
    <button class="btn btn-primary" id="drawBtn" onclick="doDraw()">Draw card</button>
  </div>
</header>

<div class="wrap">
  <div class="opp-row" id="opps"></div>

  <div class="table">
    <div class="wood"></div><div class="felt"></div>
    <div class="center">
      <div style="position:relative">
        <div class="pile deck" id="deck" onclick="doDraw()" title="Draw a card"></div>
        <div class="deck-count" id="deckN"></div>
      </div>
      <div class="meta-col">
        <div class="color-label">Current colour</div>
        <div class="color-orb" id="orb"></div>
        <div class="dir" id="dir">↻</div>
        <div class="discard-hint" id="discN"></div>
      </div>
      <div class="pile" id="topCard"></div>
    </div>
  </div>

  <div class="log" id="log"></div>

  <div class="hand-wrap">
    <div class="hand-head">
      <h3>Your hand</h3>
      <div class="turn-chip" id="turnChip">Your turn</div>
    </div>
    <div id="hand"></div>
  </div>
</div>

<div class="modal" id="colorModal"><div class="modal-card">
  <h3>Choose a colour</h3>
  <div class="crow">
    <div class="csel" style="background:var(--red)" onclick="pickColor('Red')" title="Red"></div>
    <div class="csel" style="background:var(--blue)" onclick="pickColor('Blue')" title="Blue"></div>
    <div class="csel" style="background:var(--green)" onclick="pickColor('Green')" title="Green"></div>
    <div class="csel" style="background:var(--yellow)" onclick="pickColor('Yellow')" title="Yellow"></div>
  </div>
  <p style="margin-top:12px;font-size:12px;color:#94a3b8">Wild cards let the bot learn colour choice — now it's your turn to choose.</p>
</div></div>

<div class="overlay" id="over"><div class="podium">
  <h2 id="overTitle" style="font-family:'Space Grotesk',sans-serif;font-size:22px;margin-bottom:6px"></h2>
  <p id="overSub" style="color:#94a3b8;font-size:13px;margin-bottom:14px"></p>
  <div id="podium"></div>
  <button class="btn btn-primary" style="margin-top:14px" onclick="closeOver()">Play again</button>
</div></div>

<script>
let S=null, pendingIdx=null, animating=false;
const COLORS={Red:'#ef4444',Blue:'#3b82f6',Green:'#22c55e',Yellow:'#eab308'};
const VAL={Skip:'⊘',Reverse:'⇄',DrawTwo:'+2',Wild:'W',WildDrawFour:'+4'};
const AVATAR={You:'🧑', 'Bot West':'🤖', 'Bot North':'🦊', 'Bot East':'🐯'};

function sleep(ms){return new Promise(r=>setTimeout(r,ms));}

async function apiNew(){ if(animating) return;
  const j=await (await fetch('/api/new',{method:'POST'})).json();
  if(j.error) alert(j.error);
  S=j; render(); if(j.bot_moves) await animateBots(j.bot_moves);
}
async function doDraw(){ if(animating) return;
  const j=await (await fetch('/api/draw',{method:'POST',
    headers:{'Content-Type':'application/json'}})).json();
  if(j.error){ alert(j.error); return; }
  S=j; render();
  // animate your draw (one card slides from deck)
  if(!j.finished) { await flyFromDeckToHand(); }
  if(j.bot_moves && j.bot_moves.length) await animateBots(j.bot_moves);
  else render();
}
async function doPlay(idx,color){
  if(animating) return;
  const j=await (await fetch('/api/play',{method:'POST',
    headers:{'Content-Type':'application/json'}, body:JSON.stringify({idx,color})})).json();
  if(j.error){ alert(j.error); return; }
  // animate human play
  await flyFromHandToCenter(idx);
  S=j; render();
  if(j.bot_moves && j.bot_moves.length) await animateBots(j.bot_moves);
}

function cardFace(c){
  const bg=c.color?COLORS[c.color]: '';
  const cls=c.color?'':'wildbg';
  const label=VAL[c.value]||c.value;
  return {bg, cls, label};
}

function render(){
  if(!S) return;
  // opponents
  document.getElementById('opps').innerHTML = S.opponents.map(o=>{
    const active = !S.finished && S.active_idx=== (o.name==='Bot West'?1:o.name==='Bot North'?2:3);
    const fin = o.finished;
    const rankBadge = fin ? `<span class="badge rank1">#${o.rank}</span>`
                    : `<span class="badge">${o.count} cards</span>`;
    const backs = fin ? '' : `<span class="card-backs">${'<i></i>'.repeat(Math.min(o.count,6))}</span>`;
    return `<div class="seat ${active?'active':''} ${fin?'finished':''}">
      <div class="avatar">${AVATAR[o.name]||'🤖'}</div>
      <div class="meta"><div class="name">${o.name} ${fin?'✓':''}</div>
        <div class="sub">${rankBadge} ${backs}</div></div>
    </div>`;
  }).join('');

  // your seat handled in hand header
  const youRank = S.you && S.you.rank ? `Finished #${S.you.rank}` : `${S.hand.length} cards`;
  const chip=document.getElementById('turnChip');
  if(S.finished){ chip.textContent = S.loser==='You' ? 'You lose — last standing' : `Finished #${S.you.rank||'?'}`; chip.className='turn-chip wait'; }
  else if(S.spectating){ chip.textContent='Spectating — you finished!'; chip.className='turn-chip wait'; }
  else chip.textContent = S.your_turn ? '✦ Your turn' : 'Bots thinking…';
  if(!S.finished) chip.className = S.your_turn ? 'turn-chip' : 'turn-chip wait';

  // table
  document.getElementById('orb').style.background = COLORS[S.current_color]||'#475569';
  document.getElementById('orb').style.boxShadow = `0 0 18px ${COLORS[S.current_color]||'#475569'}88`;
  document.getElementById('dir').textContent = S.direction===1 ? '↻' : '↺';
  document.getElementById('dir').classList.toggle('rev', S.direction===-1);
  document.getElementById('deckN').textContent = `draw · ${S.deck_count}`;
  document.getElementById('discN').textContent = `discard · ${S.discard_count}`;
  const t=S.top;
  const topEl=document.getElementById('topCard');
  const tf=cardFace(t);
  topEl.style.background = tf.bg || '';
  topEl.className = 'pile ' + tf.cls;
  topEl.innerHTML = `<small>${t.color||'WILD'}</small><span style="position:relative;z-index:1">${tf.label}</span><span class="oval" style="${t.color?'display:none':''}"></span>`;

  // hand
  const handEl=document.getElementById('hand');
  if(S.hand.length===0 && S.you && S.you.finished){
    handEl.innerHTML = `<div style="color:#86efac;font-weight:700;padding:18px">You finished #${S.you.rank}! Watching the rest…</div>`;
  } else {
    handEl.innerHTML = S.hand.map((c,i)=>{
      const f=cardFace(c);
      const playable = S.your_turn && S.legal[i] && !S.finished && !S.spectating;
      return `<div class="card ${playable?'playable':''} ${S.legal[i]?'':'dead'} ${f.cls}" data-i="${i}"
        style="${f.bg?`background:${f.bg}`:''}">
        <span class="oval" style="${c.color?'display:none':''}"></span>
        <small>${c.color||''}</small><span class="val">${f.label}</span></div>`;
    }).join('');
    handEl.querySelectorAll('.card.playable').forEach(el=>{
      el.onclick=()=>{
        const i=+el.dataset.i; const c=S.hand[i];
        if(!c.color){ pendingIdx=i; document.getElementById('colorModal').classList.add('open'); }
        else doPlay(i);
      };
    });
  }
  document.getElementById('drawBtn').disabled = !S.your_turn || S.finished || S.spectating || animating;

  // log
  document.getElementById('log').innerHTML = S.events.map(e=>'· '+e).join('<br>');
  document.getElementById('log').scrollTop = 9999;

  // ranking overlay
  if(S.finished){
    const title = S.loser==='You' ? 'You lose 😅' : 'Game over!';
    const sub = S.loser ? `${S.loser} is last holding cards — everyone else is safe.` : '';
    document.getElementById('overTitle').textContent = title;
    document.getElementById('overSub').textContent = sub;
    const all = [...S.ranking.map((name,idx)=>({name,rank:idx+1})), {name:S.loser, rank:4, loser:true}];
    document.getElementById('podium').innerHTML = all.map(r=>{
      const cls = r.loser ? 'rL' : r.rank===1 ? 'r1' : r.rank===2 ? 'r2' : 'r3';
      const medal = r.rank===1?'🥇':r.rank===2?'🥈':r.rank===3?'🥉':'💀';
      return `<div class="rank-row"><div class="rank-num ${cls}">${r.loser?'✕':r.rank}</div>
        <div style="font-weight:700">${medal} ${r.name}</div>
        <div style="margin-left:auto;font-size:12px;color:#94a3b8">${r.loser?'Loser':`#${r.rank} safe`}</div></div>`;
    }).join('');
    document.getElementById('over').classList.add('open');
  }
}

function pickColor(col){
  document.getElementById('colorModal').classList.remove('open');
  const idx=pendingIdx; pendingIdx=null;
  doPlay(idx,col);
}
function closeOver(){
  document.getElementById('over').classList.remove('open');
  apiNew();
}
document.getElementById('colorModal').addEventListener('click',e=>{
  if(e.target.id==='colorModal') e.currentTarget.classList.remove('open');
});

async function flyFromHandToCenter(idx){
  const cardEl=document.querySelector(`.card[data-i="${idx}"]`);
  const target=document.getElementById('topCard');
  if(!cardEl||!target) return;
  const r1=cardEl.getBoundingClientRect(), r2=target.getBoundingClientRect();
  const fly=document.createElement('div');
  fly.className='card fly'; fly.style.cssText=`left:${r1.left}px;top:${r1.top}px;width:${r1.width}px;height:${r1.height}px;background:${cardEl.style.background||'#111'};border-radius:16px`;
  fly.innerHTML=cardEl.innerHTML; document.body.appendChild(fly);
  cardEl.style.opacity='0';
  await sleep(20);
  fly.style.left=r2.left+'px'; fly.style.top=r2.top+'px';
  fly.style.transform='rotate(6deg) scale(.98)';
  await sleep(520); fly.remove();
}
async function flyFromDeckToHand(){
  const deck=document.getElementById('deck'), hand=document.getElementById('hand');
  if(!deck||!hand) return;
  const r1=deck.getBoundingClientRect(), r2=hand.getBoundingClientRect();
  const fly=document.createElement('div');
  fly.className='card fly'; fly.style.cssText=`left:${r1.left}px;top:${r1.top}px;width:86px;height:126px;background:repeating-linear-gradient(45deg,#1e40af 0 8px,#1e3a8a 8px 16px);border-radius:16px;border:2px solid #fff`;
  document.body.appendChild(fly);
  await sleep(20);
  fly.style.left=(r2.left+r2.width/2-42)+'px'; fly.style.top=(r2.top+10)+'px';
  fly.style.transform='rotate(-4deg)';
  await sleep(520); fly.remove();
}
async function animateBots(moves){
  if(!moves||!moves.length) return;
  animating=true; render();
  const fast = S && S.spectating;
  for(const m of moves){
    // thinking pause — faster when you're spectating so the wait isn't endless
    await sleep(fast ? 180 + Math.random()*220 : 420 + Math.random()*520);
    // show toast-like highlight in log already; now animate card movement
    if(m.drew){
      // deck -> bot seat
      const deck=document.getElementById('deck');
      const oppRow=document.getElementById('opps');
      if(deck && oppRow){
        const r1=deck.getBoundingClientRect();
        for(let k=0;k<Math.min(m.drawn,3);k++){
          const fly=document.createElement('div');
          fly.className='fly'; fly.style.cssText=`left:${r1.left}px;top:${r1.top}px;width:22px;height:32px;border-radius:5px;background:repeating-linear-gradient(45deg,#1e40af 0 6px,#1e3a8a 6px 12px);border:1px solid #fff`;
          document.body.appendChild(fly);
          const tgt = oppRow.children[m.pid-1] ? oppRow.children[m.pid-1].getBoundingClientRect()
                    : oppRow.getBoundingClientRect();
          await sleep(40);
          fly.style.left=(tgt.left+tgt.width/2-11)+'px';
          fly.style.top=(tgt.top+10)+'px';
          await sleep(220); fly.remove();
        }
        if(m.drawn>3) await sleep(120);
      }
    } else {
      // bot play -> center
      const oppRow=document.getElementById('opps');
      const tgt=document.getElementById('topCard');
      if(oppRow && tgt){
        const src = oppRow.children[m.pid-1] || oppRow;
        const r1=src.getBoundingClientRect(), r2=tgt.getBoundingClientRect();
        const fly=document.createElement('div');
        const col = (m.current_color && COLORS[m.current_color]) || '#334155';
        fly.className='card fly'; fly.style.cssText=`left:${r1.left+r1.width/2-36}px;top:${r1.top+10}px;width:72px;height:108px;background:${col};border-radius:14px;border:2px solid #fff;display:grid;place-items:center;color:#fff;font-weight:800`;
        fly.textContent = (m.text.split('plays')[1]||'').trim().slice(0,12);
        document.body.appendChild(fly);
        await sleep(20);
        fly.style.left=r2.left+'px'; fly.style.top=r2.top+'px';
        await sleep(560); fly.remove();
      }
    }
    // patch state to this move's snapshot for smooth count updates
    if(m.counts){
      // update counts live before full render
      S.opponents.forEach((o,i)=>{ o.count = m.counts[i+1]; if(m.finished&&m.finished.includes(i+1)) o.finished=true; });
      S.top = m.top; S.current_color = m.current_color; S.direction = m.direction;
      if(m.ranking) S.ranking = m.ranking;
      render();
    }
    // brief pause between bots
    await sleep(fast ? 110 : 260);
  }
  animating=false;
  render();
  // if human is spectating, keep ticking bots until everyone is ranked
  if(S && S.spectating && !S.finished){
    await sleep(fast ? 400 : 700);
    const j=await (await fetch('/api/tick',{method:'POST',
      headers:{'Content-Type':'application/json'}})).json();
    if(!j.error){
      S=j; render();
      if(j.bot_moves && j.bot_moves.length) await animateBots(j.bot_moves);
      else if(!S.finished) { await sleep(500); const k=await (await fetch('/api/tick',{method:'POST'})).json(); if(!k.error){S=k; render(); if(k.bot_moves) await animateBots(k.bot_moves);} }
    }
  }
}

// boot
(async()=>{ const j=await (await fetch('/api/state')).json(); S=j; render(); if(S.spectating && !S.finished){ await sleep(900); const t=await (await fetch('/api/tick',{method:'POST'})).json(); if(!t.error){S=t; render(); if(t.bot_moves) await animateBots(t.bot_moves);} } })();
</script></body></html>"""


@app.get("/")
def route_index():
    return render_template_string(PAGE_HTML)


def main():
    global session
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()
    session = Session(find_model(args.model))
    print(f"Serving on http://localhost:{args.port}  —  elimination mode (last standing loses)")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()

