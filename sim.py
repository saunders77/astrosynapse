cards = [
    #                   0       1       2       3       4       5       6           7               8       9               10      11      12
    # name,             cost,   attack, health, trade,  colour, type,   ability,    allyability,    allyn, scrapability,    scrapn, shield, count
    ('Scout',           0,      0,      0,      1,      'none', 'ship', 'none',     'none',         0,      'none',         0,      0,      0),
    ('Viper',           0,      1,      0,      0,      'none', 'ship', 'none',     'none',         0,      'none',         0,      0,      0),
    ('Explorer',        2,      0,      0,      2,      'none', 'ship', 'none',     'none',         0,      'attack',       2,      0,      0),
    ('Battle Blob',     6,      8,      0,      0,      'green','ship', 'none',     'draw',         0,      'attack',       4,      0,      1),
    ('Battle Pod',      2,      4,      0,      0,      'green','ship', 'rowscrap', 'attack',       2,      'none',         0,      0,      2),
    ('Blob Carrier',    6,      7,      0,      0,      'green','ship', 'none',     'freebuy',      0,      'none',         0,      0,      1),
    ('Blob Destroyer',  4,      6,      0,      0,      'green','ship', 'none',     'destroyscrap', 0,      'none',         0,      0,      2),
    ('Blob Fighter',    1,      3,      0,      0,      'green','ship', 'none',     'draw',         0,      'none',         0,      0,      3),
    ('Blob Wheel',      3,      1,      0,      0,      'green','base', 'none',     'none',         0,      'trade',        3,      5,      3),
    ('Blob World',      8,      0,      0,      0,      'green','base', 'blobworld','none',         0,      'none',         0,      7,      1),
    ('Mothership',      7,      6,      0,      0,      'green','ship', 'draw',     'draw',         0,      'none',         0,      0,      1),
    ('Ram',             3,      5,      0,      0,      'green','ship', 'none',     'attack',       2,      'trade',        3,      0,      2),
    ('The Hive',        5,      3,      0,      0,      'green','base', 'none',     'draw',         0,      'none',         0,      5,      1),
    ('BTrade Pod',      2,      0,      0,      3,      'green','ship', 'none',     'attack',       2,      'none',         0,      0,      3),
    ('Battle Mech',     5,      4,      0,      0,      'red',  'ship', 'scrapany', 'draw',         0,      'none',         0,      0,      1),
    ('Battle Station',  3,      0,      0,      0,      'red',  'outp', 'none',     'none',         0,      'attack',       5,      5,      2),
    ('Brain World',     8,      0,      0,      0,      'red',  'outp', 'scraptwo', 'none',         0,      'none',         0,      6,      1),
    ('Junkyard',        6,      0,      0,      0,      'red',  'outp', 'scrapany', 'none',         0,      'none',         0,      5,      1),
    ('Machine Base',    7,      0,      0,      0,      'red',  'outp', 'drawscrap','none',         0,      'none',         0,      6,      1),
    ('Mech World',      5,      0,      0,      0,      'red',  'outp', 'allally',  'none',         0,      'none',         0,      6,      1),
    ('Missile Bot',     2,      2,      0,      0,      'red',  'ship', 'scrapany', 'attack',       2,      'none',         0,      0,      3),
    ('Missile Mech',    6,      6,      0,      0,      'red',  'ship', 'killbase', 'draw',         0,      'none',         0,      0,      1),
    ('Patrol Mech',     4,      0,      0,      0,      'red',  'ship', '5or0or3',  'scrapany',     0,      'none',         0,      0,      2),
    ('Stealth Needle',  4,      0,      0,      0,      'red',  'ship', 'copyship', 'none',         0,      'none',         0,      0,      1),
    ('Supply Bot',      3,      0,      0,      2,      'red',  'ship', 'scrapany', 'attack',       2,      'none',         0,      0,      3),
    ('Trade Bot',       1,      0,      0,      1,      'red',  'ship', 'scrapany', 'attack',       2,      'none',         0,      0,      3),
    ('Battlecruiser',   6,      5,      0,      0,      'yellow','ship','draw',     'opdiscard',    0,      'drawdestroy',  0,      0,      1),
    ('Corvette',        2,      1,      0,      0,      'yellow','ship','draw',     'attack',       2,      'none',         0,      0,      2),
    ('Dreadnaught',     7,      7,      0,      0,      'yellow','ship','draw',     'none',         0,      'attack',       5,      0,      1),
    ('Fleet HQ',        8,      0,      0,      0,      'yellow','base','fleethq',  'none',         0,      'none',         0,      8,      1),
    ('Imperial Fighter',1,      2,      0,      0,      'yellow','ship','opdiscard','attack',       2,      'none',         0,      0,      3),
    ('Imperial Frigate',3,      4,      0,      0,      'yellow','ship','opdiscard','attack',       2,      'draw',         0,      0,      3),
    ('Recycling Station',4,     0,      0,      0,      'yellow','outp','recycle',  'none',         0,      'none',         0,      4,      2),
    ('Royal Redoubt',   6,      3,      0,      0,      'yellow','outp','none',     'opdiscard',    0,      'none',         0,      6,      1),
    ('Space Station',   4,      2,      0,      0,      'yellow','outp','none',     'attack',       2,      'trade',        4,      4,      2),
    ('Survey Ship',     3,      0,      0,      1,      'yellow','ship','draw',     'none',         0,      'opdiscard',    0,      0,      3),
    ('War World',       5,      3,      0,      0,      'yellow','outp','none',     'attack',       4,      'none',         0,      4,      1),
    ('Barter World',    4,      0,      0,      0,      'blue', 'base', '0or2or2',  'none',         0,      'attack',       5,      4,      2),
    ('Central Office',  7,      0,      0,      2,      'blue', 'base', 'shiptop',  'draw',         0,      'none',         0,      6,      1),
    ('Command Ship',    8,      5,      4,      0,      'blue', 'ship', 'draw2',    'killbase',     0,      'none',         0,      0,      1),
    ('Cutter',          2,      0,      4,      2,      'blue', 'ship', 'none',     'attack',       4,      'none',         0,      0,      3),
    ('Defense Center',  5,      0,      0,      0,      'blue', 'outp', '2or3or0',  'attack',       2,      'none',         0,      5,      1),
    ('Embassy Yacht',   3,      0,      3,      2,      'blue', 'ship', 'bases2d2', 'none',         0,      'none',         0,      0,      2),
    ('Federation Shuttle',1,    0,      0,      2,      'blue', 'ship', 'none',     'authority',    4,      'none',         0,      0,      3),
    ('Flagship',        6,      0,      0,      0,      'blue', 'ship', 'draw',     'authority',    5,      'none',         0,      0,      1),
    ('Freighter',       4,      0,      0,      4,      'blue', 'ship', 'none',     'shiptop',      0,      'none',         0,      0,      2),
    ('Port of Call',    6,      0,      0,      3,      'blue', 'outp', 'none',     'none',         0,      'drawdestroy',  0,      6,      1),
    ('Trade Escort',    5,      4,      4,      0,      'blue', 'ship', 'none',     'draw',         0,      'none',         0,      0,      1),
    ('Trading Post',    3,      0,      0,      0,      'blue', 'outp', '0or1or1',  'none',         0,      'attack',       3,      4,      2),

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
        