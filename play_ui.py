"""
Play against the trained UNO transformer in your browser.

Usage:
    python play_ui.py [--model uno_brain_best.pkl] [--port 5000]

Then open http://localhost:5000

You are "You" (bottom seat). The three other seats are the neural network.
Rules match the training engine: drawing one card ends your turn,
no UNO-calling, deck reshuffles when empty.
"""
import argparse
import os
import pickle
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
        "Download uno_brain_best.pkl from the GitHub Actions training run\n"
        "(Artifacts section) into this folder, or pass --model PATH."
    )


class Session:
    def __init__(self, params):
        self.params = params
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.game = UnoGame(["You", "Bot West", "Bot North", "Bot East"])
        self.winner = None
        self.events = ["New game — you go first. Good luck!"]

    # ------------------------------------------------------------------ util
    @staticmethod
    def _card_dict(card, idx=None):
        return {"color": card.color, "value": card.value, "idx": idx}

    def _describe(self, pid, action_id):
        name = self.game.players[pid].name
        if action_id >= ACT_DRAW:
            return f"{name} draws a card"
        if action_id < ACT_WILD_BASE:
            ci, vi = divmod(action_id, len(Card.VALUES))
            return f"{name} plays {Card.COLORS[ci]} {Card.VALUES[vi]}"
        base = ACT_WILD_BASE if action_id < ACT_WDF_BASE else ACT_WDF_BASE
        val = "Wild" if base == ACT_WILD_BASE else "Wild Draw Four"
        return f"{name} plays {val} -> chooses {Card.COLORS[action_id - base]}"

    def _run_bots(self):
        """Bots move greedily until it is the human's turn or someone wins."""
        guard = 0
        while self.winner is None and self.game.current_player_idx != 0:
            guard += 1
            if guard > 2000:
                self.events.append("Game stalled — start a new game.")
                break
            pid = self.game.current_player_idx
            logits, _ = forward(self.params,
                                get_observation(self.game, pid),
                                get_belief_state(self.game))
            logits = np.array(logits)
            mask = get_action_mask(self.game, pid)
            logits[~mask] = -1e9
            action = int(np.argmax(logits))          # greedy at play time
            result, executed = execute_action(self.game, pid, action)
            self.events.append(self._describe(pid, executed))
            if "winner" in result:
                self.winner = result["winner"]
                self.events.append(f"{self.winner} wins!")

    # ------------------------------------------------------------------- API
    def state_json(self):
        g = self.game
        me = g.players[0]
        return {
            "your_turn": self.winner is None and g.current_player_idx == 0,
            "hand": [self._card_dict(c, i) for i, c in enumerate(me.hand)],
            "legal": [g.is_valid_move(c) for c in me.hand],
            "top": self._card_dict(g.discard_pile[-1]),
            "current_color": g.current_color,
            "direction": g.direction,
            "deck_count": len(g.deck.cards),
            "discard_count": len(g.discard_pile),
            "opponents": [{"name": p.name, "count": len(p.hand)}
                          for p in g.players[1:]],
            "winner": self.winner,
            "events": self.events[-8:],
        }

    def api_new(self):
        with self.lock:
            self.reset()
            return self.state_json()

    def api_play(self, idx, color):
        with self.lock:
            if self.winner:
                return {"error": "Game over — start a new game."}
            if self.game.current_player_idx != 0:
                return {"error": "Not your turn."}
            hand = self.game.players[0].hand
            if not isinstance(idx, int) or idx < 0 or idx >= len(hand):
                return {"error": "Bad card index."}
            card = hand[idx]
            if not self.game.is_valid_move(card):
                return {"error": "That card doesn't match."}
            chosen = color if card.color is None else None
            if card.color is None and chosen not in Card.COLORS:
                return {"error": "Pick a colour for the wild card."}
            result = self.game.play_turn(0, idx, chosen)
            if "winner" in result:
                self.winner = result["winner"]
                self.events.append(f"{result['winner']} wins!")
            else:
                shown = f"{card.color} {card.value}" if card.color \
                    else f"{card.value} -> {chosen}"
                self.events.append(f"You played {shown}")
                self._run_bots()
            return self.state_json()

    def api_draw(self):
        with self.lock:
            if self.winner:
                return {"error": "Game over — start a new game."}
            if self.game.current_player_idx != 0:
                return {"error": "Not your turn."}
            self.game.play_turn(0, None)
            self.events.append("You draw (turn passes)")
            self._run_bots()
            return self.state_json()


session = None
app = Flask(__name__)


@app.post("/api/new")
def route_new():
    return jsonify(session.api_new())


@app.get("/api/state")
def route_state():
    return jsonify(session.api_state())


@app.post("/api/play")
def route_play():
    data = request.get_json(force=True)
    out = session.api_play(data.get("idx"), (data.get("color") or "").title() or None)
    return jsonify(out)


@app.post("/api/draw")
def route_draw():
    return jsonify(session.api_draw())


PAGE_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>UNO vs Transformer</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 :root{--red:#ef4444;--blue:#3b82f6;--green:#22c55e;--yellow:#eab308}
 *{box-sizing:border-box;margin:0}
 body{font-family:system-ui,Segoe UI,sans-serif;background:linear-gradient(160deg,#0b1220,#101d33 55%,#0b1220);
      color:#e5e7eb;min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:14px}
 header{width:min(1000px,100%);display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
 h1{font-size:18px;font-weight:700} h1 span{color:#60a5fa}
 button{cursor:pointer;border:none;border-radius:10px;padding:9px 16px;font-weight:600;color:#fff;background:#2563eb}
 button:hover{filter:brightness(1.15)} button:disabled{opacity:.35;cursor:not-allowed;filter:none}
 .opp{display:flex;gap:26px;margin-bottom:6px}
 .seat{background:#1e293b;border-radius:12px;padding:8px 14px;text-align:center;min-width:110px}
 .seat .n{font-size:12px;color:#94a3b8}.seat .c{font-size:20px;font-weight:800}
 .table{background:radial-gradient(ellipse at center,#14532d33,#00000000 70%);border-radius:24px;
        padding:18px 34px;display:flex;align-items:center;gap:38px;margin:8px 0 4px}
 .pile{position:relative;width:86px;height:126px;border-radius:14px;display:flex;align-items:center;justify-content:center;
       font-weight:800;font-size:26px;color:#fff;text-shadow:0 2px 6px rgba(0,0,0,.6);border:3px solid #ffffffcc;
       box-shadow:0 8px 20px rgba(0,0,0,.45)}
 .deck{background:repeating-linear-gradient(45deg,#1d4ed8,#1d4ed8 8px,#1e40af 8px,#1e40af 16px);color:#dbeafe}
 .meta{display:flex;flex-direction:column;gap:6px;font-size:13px;color:#cbd5e1;align-items:center}
 .dot{width:22px;height:22px;border-radius:50%;border:2px solid #fff}
 .arrow{font-size:26px}
 #log{height:64px;width:min(1000px,100%);overflow-y:auto;background:#111a2e;border-radius:12px;
      padding:8px 14px;font-size:13px;color:#9fb2cc;line-height:1.5}
 #hand{display:flex;gap:10px;flex-wrap:wrap;justify-content:center;margin-top:14px;max-width:1000px}
 .card{width:78px;height:114px;border-radius:13px;border:3px solid rgba(255,255,255,.85);
       display:flex;align-items:center;justify-content:center;font-weight:800;font-size:22px;color:#fff;
       text-shadow:0 2px 5px rgba(0,0,0,.65);box-shadow:0 5px 14px rgba(0,0,0,.4);position:relative;
       transition:transform .12s, box-shadow .12s}
 .card small{position:absolute;top:4px;left:7px;font-size:11px}
 .card.playable{cursor:pointer}.card.playable:hover{transform:translateY(-12px) scale(1.05);box-shadow:0 14px 24px rgba(0,0,0,.6)}
 .card.dead{opacity:.42;filter:saturate(.4)}
 .wildbg{background:conic-gradient(var(--red) 0 25%,var(--blue) 25% 50%,#111 50% 75%,var(--green) 75% 100%)!important}
 #modal{position:fixed;inset:0;background:rgba(2,6,17,.75);display:none;align-items:center;justify-content:center;z-index:9}
 #modal .box{background:#16213a;padding:22px 28px;border-radius:16px;text-align:center}
 #modal h3{margin-bottom:14px}
 .crow{display:flex;gap:12px}.csel{width:52px;height:52px;border-radius:50%;cursor:pointer;border:3px solid #fff}
 .overlay{position:fixed;inset:0;background:rgba(2,6,17,.85);display:flex;align-items:center;justify-content:center;z-index:10}
 .overlay .box{text-align:center}.overlay h2{font-size:34px;margin-bottom:18px}
 .hidden{display:none!important}
</style></head><body>
<header>
 <h1>UNO <span>vs Transformer</span></h1>
 <div style="display:flex;gap:10px">
   <button id="drawBtn">Draw</button>
   <button onclick="api('/api/new','POST')">New game</button>
 </div>
</header>
<div class="opp" id="opps"></div>
<div class="table">
  <div style="text-align:center"><div class="pile deck" id="deck"></div>
    <div style="font-size:12px;margin-top:6px" id="deckN"></div></div>
  <div class="meta">
    <div>current colour</div><div class="dot" id="cdot"></div>
    <div class="arrow" id="dir">⟳</div>
    <div id="discN" style="font-size:11px;color:#64748b"></div>
  </div>
  <div class="pile" id="topCard"></div>
</div>
<div id="log"></div>
<div id="hand"></div>

<div id="modal"><div class="box"><h3>Choose a colour</h3>
 <div class="crow">
  <div class="csel" style="background:var(--red)"   onclick="pickColor('Red')"></div>
  <div class="csel" style="background:var(--blue)"  onclick="pickColor('Blue')"></div>
  <div class="csel" style="background:var(--green)" onclick="pickColor('Green')"></div>
  <div class="csel" style="background:var(--yellow)"onclick="pickColor('Yellow')"></div>
 </div></div></div>

<div id="over" class="overlay hidden"><div class="box">
 <h2 id="winTxt"></h2><button onclick="closeOver()">Play again</button>
</div></div>

<script>
let S=null, pendingIdx=null;
const COLORS={Red:'#ef4444',Blue:'#3b82f6',Green:'#22c55e',Yellow:'#eab308'};
const VAL={Skip:'⊘',Reverse:'⇄',DrawTwo:'+2',Wild:'W',WildDrawFour:'+4'};

async function api(url,method,body){
  const r=await fetch(url,{method:method||'GET',headers:{'Content-Type':'application/json'},
                           body:body?JSON.stringify(body):undefined});
  const j=await r.json(); if(j.error){alert(j.error);return S;}
  S=j; render(); return j;
}
function cardHTML(c,playable,idx){
  const cls=(playable?'playable':'dead')+(c.color?'':' wildbg');
  return `<div class="card ${cls}" ${playable?`data-i="${idx}"`:''}
    style="${c.color?`background:${COLORS[c.color]}`:''}">
    <small>${c.color?c.color:''}</small>${VAL[c.value]||c.value}</div>`;
}
function oppHTML(o){return `<div class="seat"><div class="n">${o.name}</div><div class="c">🎴 ${o.count}</div></div>`;}
function render(){
  if(!S)return;
  document.getElementById('opps').innerHTML=S.opponents.map(oppHTML).join('');
  document.getElementById('deck').textContent='';
  document.getElementById('deckN').textContent=`draw pile · ${S.deck_count}`;
  document.getElementById('discN').textContent=`discard · ${S.discard_count}`;
  document.getElementById('cdot').style.background=COLORS[S.current_color]||'#666';
  document.getElementById('dir').textContent=S.direction===1?'⟳ clockwise':'⟲ counter';
  const t=S.top;
  document.getElementById('topCard').style.background=t.color?COLORS[t.color]:null;
  document.getElementById('topCard').className='pile'+(t.color?'':' wildbg');
  document.getElementById('topCard').innerHTML=`<small style="position:absolute;top:5px;left:9px;font-size:12px">${t.color||''}</small>${VAL[t.value]||t.value}`;
  document.getElementById('hand').innerHTML=
    S.hand.map((c,i)=>cardHTML(c,S.your_turn&&S.legal[i],i)).join('');
  document.querySelectorAll('.card.playable').forEach(el=>{
    el.onclick=()=>{const i=+el.dataset.i;const c=S.hand[i];
      if(!c.color){pendingIdx=i;document.getElementById('modal').style.display='flex';}
      else api('/api/play','POST',{idx:i});};});
  document.getElementById('drawBtn').disabled=!S.your_turn;
  document.getElementById('log').innerHTML=S.events.map(e=>'· '+e).join('<br>');
  if(S.winner){document.getElementById('winTxt').textContent=
      S.winner==='You'?'🏆 You win!':`💀 ${S.winner} wins`;
    document.getElementById('over').classList.remove('hidden');}
}
function pickColor(col){document.getElementById('modal').style.display='none';
  api('/api/play','POST',{idx:pendingIdx,color:col});pendingIdx=null;}
function closeOver(){document.getElementById('over').classList.add('hidden');api('/api/new','POST');}
document.getElementById('drawBtn').onclick=()=>api('/api/draw','POST');
render(); api('/api/state','GET');
</script></body></html>"""


@app.get("/")
def route_index():
    return render_template_string(PAGE_HTML)


def main():
    global session
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None,
                   help="path to v3 checkpoint (default: uno_brain_best.pkl)")
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()
    session = Session(find_model(args.model))
    print(f"Serving on http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
