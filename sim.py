cards = [
    # name,             cost,   attack, health, trade,  colour, type,   ability,    allyability,    allyamount, scrapability,   scrapamount,    shield
    ('Scout',           0,      0,      0,      1,      'none', 'ship', 'none',     'none',         0,          'none',         0,              0),
    ('Viper',           0,      1,      0,      0,      'none', 'ship', 'none',     'none',         0,          'none',         0,              0),
    ('Battle Blob',     6,      8,      0,      0,      'green','ship', 'none',     'draw',         0,          'attack',       4,              0),
    ('Battle Pod',      2,      4,      0,      0,      'green','ship', 'rowscrap', 'attack',       2,          'none',         0,              0),
    ('Blob Carrier',    6,      7,      0,      0,      'green','ship', 'none',     'freebuy',      0,          'none',         0,              0),
    ('Blob Destroyer',  4,      6,      0,      0,      'green','ship', 'none',     'destroyscrap', 0,          'none',         0,              0),
    ('Blob Fighter',    1,      3,      0,      0,      'green','ship', 'none',     'draw',         0,          'none',         0,              0),
    ('Blob Wheel',      3,      1,      0,      0,      'green','base', 'none',     'none',         0,          'trade',        3,              5),
    ('Blob World',      8,      0,      0,      0,      'green','base', 'blobworld','none',         0,          'none',         0,              7),
    ('Mothership',      7,      6,      0,      0,      'green','ship', 'draw',     'draw',         0,          'none',         0,              0),
    ('Ram',             3,      5,      0,      0,      'green','ship', 'none',     'attack',       2,          'trade',        3,              0),
    ('The Hive',        5,      3,      0,      0,      'green','base', 'none',     'draw',         0,          'none',         0,              5),
    ('BTrade Pod',      2,      0,      0,      3,      'green','ship', 'none',     'attack',       2,          'none',         0,              0),
    ('Battle Mech',     5,      4,      0,      0,      'red',  'ship', 'scrapany', 'draw',         0,          'none',         0,              0),
    ('Battle Station',  3,      0,      0,      0,      'red',  'outp', 'none',     'none',         0,          'attack',       5,              5),
    ('Brain World',     8,      0,      0,      0,      'red',  'outp', 'scraptwo', 'none',         0,          'none',         0,              6),
    ('Junkyard',        6,      0,      0,      0,      'red',  'outp', 'scrapany', 'none',         0,          'none',         0,              5),
    ('Machine Base',    7,      0,      0,      0,      'red',  'outp', 'drawscrap','none',         0,          'none',         0,              6),
    ('Mech World',      5,      0,      0,      0,      'red',  'outp', 'allally',  'none',         0,          'none',         0,              6),
    ('Missile Bot',     2,      2,      0,      0,      'red',  'ship', 'scrapany', 'attack',       2,          'none',         0,              0),
    ('Missile Mech',    6,      6,      0,      0,      'red',  'ship', 'killbase', 'draw',         0,          'none',         0,              0),
    ('Patrol Mech',     4,      0,      0,      0,      'red',  'ship', '3tradeor5','scrapany',     0,          'none',         0,              0),
    ('Stealth Needle',  4,      0,      0,      0,      'red',  'ship', 'copyship', 'none',         0,          'none',         0,              0),
    ('Supply Bot',      3,      0,      0,      2,      'red',  'ship', 'scrapany', 'attack',       2,          'none',         0,              0),
    ('Trade Bot',       1,      0,      0,      1,      'red',  'ship', 'scrapany', 'attack',       2,          'none',         0,              0),
    ('Battlecruiser',   6,      5,      0,      0,      'yellow','ship','draw',     'opdiscard',    0,          'drawdestroy',  0,              0),
    ('Corvette',        2,      1,      0,      0,      'yellow','ship','draw',     'attack',       2,          'none',         0,              0),
    ('Dreadnaught',     7,      7,      0,      0,      'yellow','ship','draw',     'none',         0,          'attack',       5,              0),
    ('Fleet HQ',        8,      0,      0,      0,      'yellow','base','fleethq',  'none',         0,          'none',         0,              8),
    ('Imperial Fighter',1,      2,      0,      0,      'yellow','ship','opdiscard','attack',       2,          'none',         0,              0),
    ('Imperial Frigate',3,      4,      0,      0,      'yellow','ship','opdiscard','attack',       2,          'draw',         0,              0),
    ('Recycling Station',4,     0,      0,      0,      'yellow','outp','recycle',  'none',         0,          'none',         0,              4),
    ('Royal Redoubt',   6,      3,      0,      0,      'yellow','outp','none',     'opdiscard',    0,          'none',         0,              6),
    ('Space Station',   4,      2,      0,      0,      'yellow','outp','none',     'attack',       2,          'trade',        4,              4),
    ('Survey Ship',     3,      0,      0,      1,      'yellow','ship','draw',     'none',         0,          'opdiscard',    0,              0),
    ('War World',       5,      3,      0,      0,      'yellow','outp','none',     'attack',       4,          'none',         0,              4),

]
scout = cards[0]
viper = cards [1]

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
        