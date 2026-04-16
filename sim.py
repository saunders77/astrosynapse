# cards: name,      cost,   attack, health, trade,  colour, type
scout = ('scout',   0,      0,      0,      1,      'none', 'ship')
viper = ('viper',   0,      1,      0,      0,      'none', 'ship')

import random

class Player:
    def __init__(self, name):
        self.name = name
        self.deck = [scout, scout, scout, scout, scout, scout, scout, scout, viper, viper]
        random.shuffle(self.deck)
        self.discardPile = []
        self.hand = []
        self.draw(5)
        self.attack = 0
        self.authority = 50
        self.trade = 0

    def draw(self, n):
        for i in range(n):
            self.hand.append(self.deck.pop())

class Game:
    def __init__(self,p1name,p2name):
        self.players = [Player(p1name),Player(p2name)]
        