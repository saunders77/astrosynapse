cardDetails = [
    #                   0       1       2       3       4       5       6           7               8       9               10      11      12
    # name,             cost,   attack, health, trade,  colour, type,   ability,    allyability,    allyn,  scrapability,   scrapn, shield, count
    ('Scout',           0,      0,      0,      1,      'none', 'ship', 'none',     'none',         0,      'none',         0,      0,      0),
    ('Viper',           0,      1,      0,      0,      'none', 'ship', 'none',     'none',         0,      'none',         0,      0,      0),
    ('Explorer',        2,      0,      0,      2,      'none', 'ship', 'none',     'none',         0,      'attack',       2,      0,      0),
    ('Battle Blob',     6,      8,      0,      0,      'green','ship', 'none',     'draw',         0,      'attack',       4,      0,      1),
    ('Battle Pod',      2,      4,      0,      0,      'green','ship', 'rowscrap', 'attack',       2,      'none',         0,      0,      2),
    ('Blob Carrier',    6,      7,      0,      0,      'green','ship', 'none',     '-freebuy',      0,      'none',         0,      0,      1),
    ('Blob Destroyer',  4,      6,      0,      0,      'green','ship', 'none',     '-destroyscrap',0,      'none',         0,      0,      2),
    ('Blob Fighter',    1,      3,      0,      0,      'green','ship', 'none',     'draw',         0,      'none',         0,      0,      3),
    ('Blob Wheel',      3,      1,      0,      0,      'green','base', 'none',     'none',         0,      'trade',        3,      5,      3),
    ('Blob World',      8,      0,      0,      0,      'green','base', '-5ordraws','none',         0,      'none',         0,      7,      1),
    ('Mothership',      7,      6,      0,      0,      'green','ship', 'draw',     'draw',         0,      'none',         0,      0,      1),
    ('Ram',             3,      5,      0,      0,      'green','ship', 'none',     'attack',       2,      'trade',        3,      0,      2),
    ('The Hive',        5,      3,      0,      0,      'green','base', 'none',     'draw',         0,      'none',         0,      5,      1),
    ('BTrade Pod',      2,      0,      0,      3,      'green','ship', 'none',     'attack',       2,      'none',         0,      0,      3),
    ('Battle Mech',     5,      4,      0,      0,      'red',  'ship', 'scrapany', 'draw',         0,      'none',         0,      0,      1),
    ('Battle Station',  3,      0,      0,      0,      'red',  'outp', 'none',     'none',         0,      'attack',       5,      5,      2),
    ('Brain World',     8,      0,      0,      0,      'red',  'outp', '-scraptwo','none',         0,      'none',         0,      6,      1),
    ('Junkyard',        6,      0,      0,      0,      'red',  'outp', '-scrapany','none',         0,      'none',         0,      5,      1),
    ('Machine Base',    7,      0,      0,      0,      'red',  'outp', '-drawscrap','none',        0,      'none',         0,      6,      1),
    ('Mech World',      5,      0,      0,      0,      'red',  'outp', 'allally',  'none',         0,      'none',         0,      6,      1),
    ('Missile Bot',     2,      2,      0,      0,      'red',  'ship', 'scrapany', 'attack',       2,      'none',         0,      0,      3),
    ('Missile Mech',    6,      6,      0,      0,      'red',  'ship', 'killbase', 'draw',         0,      'none',         0,      0,      1),
    ('Patrol Mech',     4,      0,      0,      0,      'red',  'ship', '5or0or3', '-scrapany',     0,      'none',         0,      0,      2),
    ('Stealth Needle',  4,      0,      0,      0,      'red',  'ship', 'copyship', 'none',         0,      'none',         0,      0,      1),
    ('Supply Bot',      3,      0,      0,      2,      'red',  'ship', 'scrapany', 'attack',       2,      'none',         0,      0,      3),
    ('Trade Bot',       1,      0,      0,      1,      'red',  'ship', 'scrapany', 'attack',       2,      'none',         0,      0,      3),
    ('Battlecruiser',   6,      5,      0,      0,      'yellow','ship','draw',     'opdiscard',    0,      'drawdestroy',  0,      0,      1),
    ('Corvette',        2,      1,      0,      0,      'yellow','ship','draw',     'attack',       2,      'none',         0,      0,      2),
    ('Dreadnaught',     7,      7,      0,      0,      'yellow','ship','draw',     'none',         0,      'attack',       5,      0,      1),
    ('Fleet HQ',        8,      0,      0,      0,      'yellow','base','fleethq',  'none',         0,      'none',         0,      8,      1),
    ('Imperial Fighter',1,      2,      0,      0,      'yellow','ship','opdiscard','attack',       2,      'none',         0,      0,      3),
    ('Imperial Frigate',3,      4,      0,      0,      'yellow','ship','opdiscard','attack',       2,      'draw',         0,      0,      3),
    ('Recycling Station',4,     0,      0,      0,      'yellow','outp','-recycle', 'none',         0,      'none',         0,      4,      2),
    ('Royal Redoubt',   6,      3,      0,      0,      'yellow','outp','none',     'opdiscard',    0,      'none',         0,      6,      1),
    ('Space Station',   4,      2,      0,      0,      'yellow','outp','none',     'attack',       2,      'trade',        4,      4,      2),
    ('Survey Ship',     3,      0,      0,      1,      'yellow','ship','draw',     'none',         0,      'opdiscard',    0,      0,      3),
    ('War World',       5,      3,      0,      0,      'yellow','outp','none',     'attack',       4,      'none',         0,      4,      1),
    ('Barter World',    4,      0,      0,      0,      'blue', 'base', '-0or2or2', 'none',         0,      'attack',       5,      4,      2),
    ('Central Office',  7,      0,      0,      2,      'blue', 'base', 'shiptop',  'draw',         0,      'none',         0,      6,      1),
    ('Command Ship',    8,      5,      4,      0,      'blue', 'ship', 'draw2',    '-killbase',    0,      'none',         0,      0,      1),
    ('Cutter',          2,      0,      4,      2,      'blue', 'ship', 'none',     'attack',       4,      'none',         0,      0,      3),
    ('Defense Center',  5,      0,      0,      0,      'blue', 'outp', '-2or3or0', 'attack',       2,      'none',         0,      5,      1),
    ('Embassy Yacht',   3,      0,      3,      2,      'blue', 'ship', 'bases2d2', 'none',         0,      'none',         0,      0,      2),
    ('Federation Shuttle',1,    0,      0,      2,      'blue', 'ship', 'none',     'authority',    4,      'none',         0,      0,      3),
    ('Flagship',        6,      0,      0,      0,      'blue', 'ship', 'draw',     'authority',    5,      'none',         0,      0,      1),
    ('Freighter',       4,      0,      0,      4,      'blue', 'ship', 'none',     'shiptop',      0,      'none',         0,      0,      2),
    ('Port of Call',    6,      0,      0,      3,      'blue', 'outp', 'none',     'none',         0,      'drawdestroy',  0,      6,      1),
    ('Trade Escort',    5,      4,      4,      0,      'blue', 'ship', 'none',     'draw',         0,      'none',         0,      0,      1),
    ('Trading Post',    3,      0,      0,      0,      'blue', 'outp', '-0or1or1', 'none',         0,      'attack',       3,      4,      2),

]
scout = cardDetails[0]
viper = cardDetails [1]
explorer = cardDetails[2]
factions = ('red','green','blue','yellow','none')

import random

class Player:
    def __init__(self, game, name):
        self.name = name
        self.game = game
        self.opponent = None
        self.deck = [scout, scout, scout, scout, scout, scout, scout, scout, viper, viper]
        random.shuffle(self.deck)
        self.discardPile = []
        self.hand = []
        self.authority = 50
        self.mustDiscard = 0     
        self.cardsInPlay = {'red':[], 'blue':[], 'green':[], 'yellow':[], 'none':[]}
        self.knownTopOfDeck = 0
        self.knownGameState = {
            'authority':None,
            'attack':0,
            'trade':0,
            'mustDiscard':None,
            'deck':None,
            'topCards':None,
            'hand':self.hand,
            'discardPile':self.discardPile,
            'cardsInPlay':self.cardsInPlay,
            'tradeRow':self.game.tradeRow,
            'nextShipTop':None,
            'opponentAuthority':None,
            'opponentMustDiscard':None,
            'opponentDeckAndHand':None,
            'opponentTopCards':None,
            'opponentDiscardPile':None,
            'opponentCardsInPlay':None,
        }

    def draw(self, n):
        for i in range(n):
            if len(self.deck) > 0:
                self.hand.append(self.deck.pop())
                if self.knownTopOfDeck > 0: self.knownTopOfDeck -= 1
            elif len(self.discardPile) > 0:
                # no more cards in the deck. use discard pile
                while len(self.discardPile) > 0:
                    self.deck.append(self.discardPile.pop())
                random.shuffle(self.deck)
                self.hand.append(self.deck.pop())
   
    def takeTurn(self):
        self.playOptions = []
        self.abilityOptions = []
        self.scrapOptions = []
        self.acquireOptions = []

        self.attack = 0
        self.trade = 0
        self.allAllied = False # for Mech World
        self.fleetActive = False # for FleetHQ
        self.nextShipTop = False # for several cards

        # take care of bases starting in play
        for faction in factions:
            for i in range(len(self.cardsInPlay[faction])):
                self.activateCard(faction,i)

        # discard if required, and reduce mustDiscard
        while self.mustDiscard > 0:
            decision = self.choose('discard')
            self.discardPile.append(self.hand.pop())

    def useAbility(self, abilityName, n = None):
        match abilityName:
            case 'attack': self.attack += n
            case 'trade': self.trade += n
            case 'authority': self.authority += n
            case 'draw': self.draw(1)
            case 'draw2': self.draw(2)
            case 'allally': self.allAllied = True
            case 'fleethq': self.fleetActive = True
            case 'opdiscard': self.opponent.mustDiscard += 1
            case 'shiptop': self.nextShipTop = True
            case 'bases2d2':
                baseCount = 0
                for faction in factions:
                    for card in self.cardsInPlay[faction]:
                        if card[0][5] != 'ship': baseCount += 1
                if baseCount >= 2: self.draw(2)

            case _: raise ValueError('Used unknown ability ' + str(abilityName))


        print(abilityName)

    def triggerAbilityOption(self, faction, position):
        card = self.cardsInPlay[faction][position]
        if card[3] == True and card[2] != 'used': # card is still in play and the ability has not been used
            self.useAbility(card[2][1:]) # [1:] removes the '-' character at the beginning
            card[2] = 'used'

    def playCard(self, cardDetails):
        #                                        0              1               2               3       4 
        #                                                       allyTriggeredIt,abilityOption,  inPlay, isStealth
        self.cardsInPlay[cardDetails[4]].append([cardDetails,   False,          None,           True,   cardDetails[6] == 'copyship'])
        self.activateCard(self, cardDetails[4], len(self.cardsInPlay) - 1)

    def removeCardFromPlay(self, faction, position):
        card = self.cardsInPlay[faction][position]
        if card[0][6] == 'allally': self.allAllied = False
        if card[0][6] == 'fleethq': self.fleetActive = False
        self.cardsInPlay[faction][position][3] = False
        self.cardsInPlay[faction].pop(position)
    
    def activateCard(self, faction, position):
        card = self.cardsInPlay[faction][position]
        self.attack += card[0][1]
        self.authority += card[0][2]
        self.trade += card[0][3]
        
        # use its ability
        if card[0][6][0] == '-' and card[2] == None: # the ability can be activated by the user at any time
            card[2] = card[0][6] # card[2] is the active ability option
            self.abilityOptions.append(card)
        elif card[0][6][0] == '-' and card[2] != None:
            raise ValueError('Option ability ' + str(card[0][6]) + ' could not be added because of an existing option: ' + str(card[2]))
        elif card[0][6] != 'none': 
            self.useAbility(card[0][6])
        
        # trigger other ally cards
        def triggerOtherAllyCards(faction,position):
            for i in range(len(self.cardsInPlay[faction])):
                allyAbility = self.cardsInPlay[faction][i][0][7]
                if i != position and allyAbility != 'none' and self.cardsInPlay[faction][i][1] == False: # the card is not this one and it has an ally ability and it hasn't been triggered
                    if allyAbility[0] == '-' and self.cardsInPlay[faction][i][2] == None: # the ability of the other card can be activated by the user at any time and hasn't already been added
                        self.cardsInPlay[faction][i][2] = allyAbility # card[2] is the active ability option
                        self.abilityOptions.append(self.cardsInPlay[faction][i])
                    elif allyAbility[0] == '-' and self.cardsInPlay[faction][i][2] != None:
                        raise ValueError('Option ability ' + str(allyAbility) + ' could not be added because of an existing option: ' + str(self.cardsInPlay[faction][i][2]))
                    else:
                        self.useAbility(allyAbility, self.cardsInPlay[faction][i][0][8])
                    self.cardsInPlay[faction][i][1] = True # mark that that card's ally ability has already been used now
        triggerOtherAllyCards(faction,position)
        if card[4] == True and faction != 'red': # then it's using the copyship and should also trigger red faction
            triggerOtherAllyCards('red',position)
        
        # trigger its own ally ability
        if card[0][7] != 'none' and (self.allAllied == True or self.cardsInPlay[faction] > 1):
            # ally ability is activated
            if card[0][7][0] == '-' and card[2] == None:  # the ability can be activated by the user at any time
                card[2] = card[0][7]
                self.abilityOptions.append(card)
            elif card[0][7][0] == '-' and card[2] != None:
                raise ValueError('Option ability ' + str(card[0][7]) + ' could not be added because of an existing option: ' + str(card[2]))
            else:
                self.useAbility(card[0][7],card[0][8])
            card[1] = True # the ally ability has now been used
        
        if card[0][9] != 'none':
            self.scrapOptions.append(card)      
    
    def scrapCard(self, faction, position):
        print("scrapping")
        # remember to turn off stuff for fleetHQ and mech world

    def endTurn(self):
        for faction in factions:
            i = 0
            while i < len(self.cardsInPlay[faction]):
                if self.cardsInPlay[faction][i][0][5] == 'ship':
                    self.discardPile.append(self.cardsInPlay[faction].pop(i)[0])
                else:
                    i += 1
                

        


class Game:
    def __init__(self,p1name,p2name):
        self.players = [Player(self,p1name),Player(self,p2name)]
        self.players[0].opponent = self.players[1]
        self.players[1].opponent = self.players[0]
        random.shuffle(self.players)
        self.players[0].draw(3)
        self.players[1].draw(5)
        self.explorers = 10
        self.readyCards = []
        for card in cards:
            for i in range(card[12]):
                self.readyCards.append(card) #12 --> count
        random.shuffle(self.readyCards)
        self.tradeRow = []
        for i in range(4):
            self.tradeRow.append(self.readyCards.pop())
        self.nextPlayer = 0
        self.winner = None
        while self.winner == None:
            self.players[self.nextPlayer].takeTurn()
            self.nextPlayer = (self.nextPlayer + 1) % 2
        print('The winner is: ' + self.winner.name + "!")

        

        