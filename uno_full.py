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
    def __init__(self, player_names):
        self.players = [Player(name) for name in player_names]
        self.deck = Deck()
        self.current_player_idx = 0
        self.direction = 1
        self.discard_pile = []
        
        # Initialize starting card
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

    def is_valid_move(self, card):
        if card.color is None: return True
        if card.color == self.current_color: return True
        if card.value == self.current_value: return True
        return False

    def play_turn(self, player_idx, card_idx, chosen_color=None):
        player = self.players[player_idx]
        
        if card_idx is None:
            player.draw_card(self.deck)
            self.current_player_idx = (self.current_player_idx + self.direction) % len(self.players)
            return {"status": "continue"}

        card = player.hand[card_idx]
        
        if not self.is_valid_move(card):
            raise ValueError("Invalid move")

        if card.color is None and chosen_color not in Card.COLORS:
            raise ValueError(f"chosen_color must be one of {Card.COLORS}")

        # Logic to play card
        player.hand.pop(card_idx)
        self.discard_pile.append(card)
        
        if len(player.hand) == 0:
            return {"winner": player.name}

        if card.value == 'Skip':
            self.current_color = card.color
            self.current_value = card.value
            self.current_player_idx = (self.current_player_idx + 2 * self.direction) % len(self.players)
        elif card.value == 'Reverse':
            self.current_color = card.color
            self.current_value = card.value
            self.direction *= -1
            self.current_player_idx = (self.current_player_idx + self.direction) % len(self.players)
        elif card.value == 'DrawTwo':
            self.current_color = card.color
            self.current_value = card.value
            next_player = self.players[(self.current_player_idx + self.direction) % len(self.players)]
            next_player.draw_card(self.deck, 2)
            self.current_player_idx = (self.current_player_idx + 2 * self.direction) % len(self.players)
        elif card.value == 'Wild':
            self.current_color = chosen_color
            self.current_value = card.value
            self.current_player_idx = (self.current_player_idx + self.direction) % len(self.players)
        elif card.value == 'WildDrawFour':
            self.current_color = chosen_color
            self.current_value = card.value
            next_player = self.players[(self.current_player_idx + self.direction) % len(self.players)]
            next_player.draw_card(self.deck, 4)
            self.current_player_idx = (self.current_player_idx + 2 * self.direction) % len(self.players)
        else:
            self.current_color = card.color
            self.current_value = card.value
            self.current_player_idx = (self.current_player_idx + self.direction) % len(self.players)
            
        return {"status": "continue"}
