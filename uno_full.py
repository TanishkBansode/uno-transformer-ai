import random

class Card:
    COLORS = ['Red', 'Blue', 'Green', 'Yellow']
    VALUES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Skip', 'Reverse', 'DrawTwo']
    WILD_VALUES = ['Wild', 'WildDrawFour']

    def __init__(self, color, value):
        self.color = color
        self.value = value

    def __repr__(self):
        return f"{self.color} {self.value}" if self.color else f"{self.value}"

class Deck:
    def __init__(self):
        self.cards = []
        self._build_deck()
        self.shuffle()

    def _build_deck(self):
        for color in Card.COLORS:
            self.cards.append(Card(color, '0'))
            for val in Card.VALUES[1:]:
                self.cards.append(Card(color, val))
                self.cards.append(Card(color, val))
        for _ in range(4):
            self.cards.append(Card(None, 'Wild'))
            self.cards.append(Card(None, 'WildDrawFour'))

    def shuffle(self):
        random.shuffle(self.cards)

    def draw(self):
        return self.cards.pop() if self.cards else None

class Player:
    def __init__(self, name):
        self.name = name
        self.hand = []

    def draw_card(self, deck, count=1):
        drawn = []
        for _ in range(count):
            card = deck.draw()
            if card:
                self.hand.append(card)
                drawn.append(card)
        return drawn

class UnoGame:
    def __init__(self, player_names, elimination_mode=False):
        self.players = [Player(name) for name in player_names]
        self.deck = Deck()
        self.current_player_idx = 0
        self.direction = 1
        self.discard_pile = []
        # elimination / ranking mode (UI) — last man standing loses
        self.elimination_mode = elimination_mode
        self.finished = []          # list of player indices in order they emptied hand
        self.ranking = []           # list of player names in order they finished

        # Initialize starting card (must be a coloured card)
        start_card = self.deck.draw()
        while start_card.color is None:
            self.deck.cards.append(start_card)
            self.deck.shuffle()
            start_card = self.deck.draw()

        self.current_color = start_card.color
        self.current_value = start_card.value
        self.discard_pile.append(start_card)

        for player in self.players:
            player.draw_card(self.deck, 7)

    # ------------------------------------------------------------------
    def _active_indices(self):
        return [i for i in range(len(self.players)) if i not in self.finished]

    def _next_active(self, from_idx, steps=1):
        """Return the player index `steps` active turns ahead of from_idx."""
        if len(self.finished) >= len(self.players):
            return from_idx
        idx = from_idx
        for _ in range(steps):
            idx = (idx + self.direction) % len(self.players)
            guard = 0
            while idx in self.finished:
                idx = (idx + self.direction) % len(self.players)
                guard += 1
                if guard > len(self.players):
                    break
        return idx

    def _ensure_deck(self):
        """
        If the draw pile is empty, reshuffle all discard pile cards
        except the top card back into the deck.  This is standard UNO
        rules and prevents infinite draw-loops when the deck runs dry.
        """
        if len(self.deck.cards) == 0 and len(self.discard_pile) > 1:
            top = self.discard_pile[-1]
            self.deck.cards = self.discard_pile[:-1]
            self.discard_pile = [top]
            self.deck.shuffle()

    # ------------------------------------------------------------------
    def is_valid_move(self, card):
        if card.color is None: return True
        if card.color == self.current_color: return True
        if card.value == self.current_value: return True
        return False

    def play_turn(self, player_idx, card_idx, chosen_color=None):
        # Skip turns for already-finished players (elimination mode)
        if self.elimination_mode and player_idx in self.finished:
            self.current_player_idx = self._next_active(player_idx, 1)
            return {"status": "continue"}

        player = self.players[player_idx]

        if card_idx is None:
            self._ensure_deck()
            player.draw_card(self.deck)
            if self.elimination_mode:
                self.current_player_idx = self._next_active(player_idx, 1)
            else:
                self.current_player_idx = (self.current_player_idx + self.direction) % len(self.players)
            return {"status": "continue", "drew": True}

        card = player.hand[card_idx]

        if not self.is_valid_move(card):
            raise ValueError("Invalid move")

        if card.color is None and chosen_color not in Card.COLORS:
            raise ValueError(f"chosen_color must be one of {Card.COLORS}")

        player.hand.pop(card_idx)
        self.discard_pile.append(card)

        # Update colour / value first so ranking state is consistent
        if card.color is None:
            self.current_color = chosen_color
        else:
            self.current_color = card.color
        self.current_value = card.value

        # --- win / elimination handling ---
        if len(player.hand) == 0:
            if not self.elimination_mode:
                return {"winner": player.name}
            # elimination / ranking mode — last man standing loses
            self.finished.append(player_idx)
            self.ranking.append(player.name)
            active = self._active_indices()
            if len(active) <= 1:
                loser = self.players[active[0]].name if active else None
                return {"finished": True, "ranking": self.ranking.copy(),
                        "loser": loser, "eliminated": player.name}
            # apply the card's effect even when the player goes out, then
            # advance to the next *active* player.
            if card.value == 'Reverse':
                self.direction *= -1
                self.current_player_idx = self._next_active(player_idx, 1)
            elif card.value in ('Skip', 'DrawTwo', 'WildDrawFour'):
                # Draw targets get their cards even when finisher goes out
                if card.value == 'DrawTwo':
                    nxt = self._next_active(player_idx, 1)
                    self._ensure_deck()
                    self.players[nxt].draw_card(self.deck, 2)
                elif card.value == 'WildDrawFour':
                    nxt = self._next_active(player_idx, 1)
                    self._ensure_deck()
                    self.players[nxt].draw_card(self.deck, 4)
                # Skip the next active player (covers Skip + the draw cards)
                self.current_player_idx = self._next_active(player_idx, 2)
            else:
                # Wild or number card
                self.current_player_idx = self._next_active(player_idx, 1)
            return {"eliminated": player.name, "ranking": self.ranking.copy()}

        # --- normal (non-winning) card effects ---
        if self.elimination_mode:
            if card.value == 'Skip':
                self.current_player_idx = self._next_active(player_idx, 2)
            elif card.value == 'Reverse':
                self.direction *= -1
                self.current_player_idx = self._next_active(player_idx, 1)
            elif card.value == 'DrawTwo':
                nxt = self._next_active(player_idx, 1)
                self._ensure_deck()
                self.players[nxt].draw_card(self.deck, 2)
                self.current_player_idx = self._next_active(player_idx, 2)
            elif card.value == 'WildDrawFour':
                nxt = self._next_active(player_idx, 1)
                self._ensure_deck()
                self.players[nxt].draw_card(self.deck, 4)
                self.current_player_idx = self._next_active(player_idx, 2)
            else:  # Wild or number
                self.current_player_idx = self._next_active(player_idx, 1)
            return {"status": "continue"}

        # classic mode (training) — simple modulo arithmetic
        if card.value == 'Skip':
            self.current_player_idx = (self.current_player_idx + 2 * self.direction) % len(self.players)
        elif card.value == 'Reverse':
            self.direction *= -1
            self.current_player_idx = (self.current_player_idx + self.direction) % len(self.players)
        elif card.value == 'DrawTwo':
            nxt = (self.current_player_idx + self.direction) % len(self.players)
            self._ensure_deck()
            self.players[nxt].draw_card(self.deck, 2)
            self.current_player_idx = (self.current_player_idx + 2 * self.direction) % len(self.players)
        elif card.value == 'Wild':
            self.current_player_idx = (self.current_player_idx + self.direction) % len(self.players)
        elif card.value == 'WildDrawFour':
            nxt = (self.current_player_idx + self.direction) % len(self.players)
            self._ensure_deck()
            self.players[nxt].draw_card(self.deck, 4)
            self.current_player_idx = (self.current_player_idx + 2 * self.direction) % len(self.players)
        else:
            self.current_player_idx = (self.current_player_idx + self.direction) % len(self.players)

        return {"status": "continue"}
