cardDetails = [
    # 0                 1       2       3       4       5       6       7           8               9       10              11      12      13
    # name,             cost,   attack, health, trade,  colour, type,   ability,    allyability,    allyn,  scrapability,   scrapn, shield, count
    ('Scout',           0,      0,      0,      1,      'none', 'ship', 'none',     'none',         0,      'none',         0,      0,      0),
    ('Viper',           0,      1,      0,      0,      'none', 'ship', 'none',     'none',         0,      'none',         0,      0,      0),
    ('Explorer',        2,      0,      0,      2,      'none', 'ship', 'none',     'none',         0,      'gainattack',   2,      0,      0),
    ('Battle Blob',     6,      8,      0,      0,      'green','ship', 'none',     'draw',         0,      'gainattack',   4,      0,      1),
    ('Battle Pod',      2,      4,      0,      0,      'green','ship', 'rowscrap', 'gainattack',   2,      'none',         0,      0,      2),
    ('Blob Carrier',    6,      7,      0,      0,      'green','ship', 'none',     '-freebuy',     0,      'none',         0,      0,      1),
    ('Blob Destroyer',  4,      6,      0,      0,      'green','ship', 'none',     '-destroyscrap',0,      'none',         0,      0,      2),
    ('Blob Fighter',    1,      3,      0,      0,      'green','ship', 'none',     'draw',         0,      'none',         0,      0,      3),
    ('Blob Wheel',      3,      1,      0,      0,      'green','base', 'none',     'none',         0,      'trade',        3,      5,      3),
    ('Blob World',      8,      0,      0,      0,      'green','base', '-5ordraws','none',         0,      'none',         0,      7,      1),
    ('Mothership',      7,      6,      0,      0,      'green','ship', 'draw',     'draw',         0,      'none',         0,      0,      1),
    ('Ram',             3,      5,      0,      0,      'green','ship', 'none',     'gainattack',   2,      'trade',        3,      0,      2),
    ('The Hive',        5,      3,      0,      0,      'green','base', 'none',     'draw',         0,      'none',         0,      5,      1),
    ('Trade Pod',       2,      0,      0,      3,      'green','ship', 'none',     'gainattack',   2,      'none',         0,      0,      3),
    ('Battle Mech',     5,      4,      0,      0,      'red',  'ship', 'scrapany', 'draw',         0,      'none',         0,      0,      1),
    ('Battle Station',  3,      0,      0,      0,      'red',  'outp', 'none',     'none',         0,      'gainattack',   5,      5,      2),
    ('Brain World',     8,      0,      0,      0,      'red',  'outp', '-scraptwo','none',         0,      'none',         0,      6,      1),
    ('Junkyard',        6,      0,      0,      0,      'red',  'outp', '-scrapany','none',         0,      'none',         0,      5,      1),
    ('Machine Base',    7,      0,      0,      0,      'red',  'outp', '-drawscrap','none',        0,      'none',         0,      6,      1),
    ('Mech World',      5,      0,      0,      0,      'red',  'outp', 'allally',  'none',         0,      'none',         0,      6,      1),
    ('Missile Bot',     2,      2,      0,      0,      'red',  'ship', 'scrapany', 'gainattack',   2,      'none',         0,      0,      3),
    ('Missile Mech',    6,      6,      0,      0,      'red',  'ship', 'killbase', 'draw',         0,      'none',         0,      0,      1),
    ('Patrol Mech',     4,      0,      0,      0,      'red',  'ship', '5or0or3', '-scrapany',     0,      'none',         0,      0,      2),
    ('Stealth Needle',  4,      0,      0,      0,      'red',  'ship', 'copyship', 'none',         0,      'none',         0,      0,      1),
    ('Supply Bot',      3,      0,      0,      2,      'red',  'ship', 'scrapany', 'gainattack',   2,      'none',         0,      0,      3),
    ('Trade Bot',       1,      0,      0,      1,      'red',  'ship', 'scrapany', 'gainattack',   2,      'none',         0,      0,      3),
    ('Battlecruiser',   6,      5,      0,      0,      'yellow','ship','draw',     'opdiscard',    0,      'drawdestroy',  0,      0,      1),
    ('Corvette',        2,      1,      0,      0,      'yellow','ship','draw',     'gainattack',   2,      'none',         0,      0,      2),
    ('Dreadnaught',     7,      7,      0,      0,      'yellow','ship','draw',     'none',         0,      'gainattack',   5,      0,      1),
    ('Fleet HQ',        8,      0,      0,      0,      'yellow','base','fleethq',  'none',         0,      'none',         0,      8,      1),
    ('Imperial Fighter',1,      2,      0,      0,      'yellow','ship','opdiscard','gainattack',   2,      'none',         0,      0,      3),
    ('Imperial Frigate',3,      4,      0,      0,      'yellow','ship','opdiscard','gainattack',   2,      'draw',         0,      0,      3),
    ('Recycling Station',4,     0,      0,      0,      'yellow','outp','-recycle', 'none',         0,      'none',         0,      4,      2),
    ('Royal Redoubt',   6,      3,      0,      0,      'yellow','outp','none',     'opdiscard',    0,      'none',         0,      6,      1),
    ('Space Station',   4,      2,      0,      0,      'yellow','outp','none',     'gainattack',   2,      'trade',        4,      4,      2),
    ('Survey Ship',     3,      0,      0,      1,      'yellow','ship','draw',     'none',         0,      'opdiscard',    0,      0,      3),
    ('War World',       5,      3,      0,      0,      'yellow','outp','none',     'gainattack',   4,      'none',         0,      4,      1),
    ('Barter World',    4,      0,      0,      0,      'blue', 'base', '-0or2or2', 'none',         0,      'gainattack',   5,      4,      2),
    ('Central Office',  7,      0,      0,      2,      'blue', 'base', 'shiptop',  'draw',         0,      'none',         0,      6,      1),
    ('Command Ship',    8,      5,      4,      0,      'blue', 'ship', 'draw2',    '-killbase',    0,      'none',         0,      0,      1),
    ('Cutter',          2,      0,      4,      2,      'blue', 'ship', 'none',     'gainattack',   4,      'none',         0,      0,      3),
    ('Defense Center',  5,      0,      0,      0,      'blue', 'outp', '-2or3or0', 'gainattack',   2,      'none',         0,      5,      1),
    ('Embassy Yacht',   3,      0,      3,      2,      'blue', 'ship', 'bases2d2', 'none',         0,      'none',         0,      0,      2),
    ('Federation Shuttle',1,    0,      0,      2,      'blue', 'ship', 'none',     'authority',    4,      'none',         0,      0,      3),
    ('Flagship',        6,      0,      0,      0,      'blue', 'ship', 'draw',     'authority',    5,      'none',         0,      0,      1),
    ('Freighter',       4,      0,      0,      4,      'blue', 'ship', 'none',     'shiptop',      0,      'none',         0,      0,      2),
    ('Port of Call',    6,      0,      0,      3,      'blue', 'outp', 'none',     'none',         0,      'drawdestroy',  0,      6,      1),
    ('Trade Escort',    5,      4,      4,      0,      'blue', 'ship', 'none',     'draw',         0,      'none',         0,      0,      1),
    ('Trading Post',    3,      0,      0,      0,      'blue', 'outp', '-0or1or1', 'none',         0,      'gainattack',   3,      4,      2),

]
scout = cardDetails[0]
viper = cardDetails [1]
explorer = cardDetails[2]
factions = ('red','green','blue','yellow','none')

import random
from chooser import choose

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
        self.opponentKnownHandCards = [] # the cards in my hand that my opponent knows about
        self.knownGameState = {
            'authority':None,
            'attack':0,
            'trade':0,
            'mustDiscard':None,
            'scrambleDeck':None,
            'topCards':None,
            'hand':self.hand,
            'discardPile':self.discardPile,
            'cardsInPlay':self.cardsInPlay,
            'tradeRow':self.game.tradeRow,
            'nextShipTop':None,
            'blobPlayCount':None,
            'opponentAuthority':None,
            'opponentMustDiscard':None,
            'opponentScrambleDeckAndHand':None,
            'opponentTopCards':None,
            'opponentHandCards':None,
            'opponentDiscardPile':None,
            'opponentCardsInPlay':None,
        }

    def calculateScrambleDeck(self):
        scrambleDeck = []
        topCards = []
        i = 0
        while i < len(self.deck) - self.knownTopOfDeck:
            scrambleDeck.append(self.deck[i])
            i += 1
        random.shuffle(scrambleDeck)
        self.knownGameState['scrambleDeck'] = scrambleDeck
        while i < len(self.deck):
            topCards.append(self.deck[i])
            i += 1
        self.knownGameState['topCards'] = topCards
    
    def calculateOpponentScrambleDeckAndHand(self):
        opScrambleDeckHand = []
        opTopCards = []
        i = 0
        while i < len(self.opponent.deck) - self.opponent.knownTopOfDeck:
            opScrambleDeckHand.append(self.opponent.deck[i])
            i += 1
        while i < len(self.opponent.deck):
            opTopCards.append(self.opponent.deck[i])
            i += 1
        i = 0
        knownOpponentHandCards = self.opponent.opponentKnownHandCards
        opHandTracker = []
        for item in knownOpponentHandCards: opHandTracker.append(item)
        while i < len(self.opponent.hand):
            if self.opponent.hand[i] in opHandTracker:
                opHandTracker.remove(self.opponent.hand[i]) # don't add it to the big scrambled list
            else:
                opScrambleDeckHand.append(self.opponent.hand[i])
            i += 1
        random.shuffle(opScrambleDeckHand)
        self.knownGameState['opponentScrambleDeckAndHand'] = opScrambleDeckHand
        self.knownGameState['opponentTopCards'] = opTopCards
        self.knownGameState['opponentHandCards'] = knownOpponentHandCards

    def sendChoice(self, options):
        self.knownGameState['authority'] = self.authority
        self.knownGameState['attack'] = self.attack
        self.knownGameState['trade'] = self.trade
        self.knownGameState['mustDiscard'] = self.mustDiscard
        # scrambleDeck and topCards are calculated dynamically        
        self.knownGameState['nextShipTop'] = self.nextShipTop
        self.knownGameState['blobPlayCount'] = self.blobCardsPlayed
        self.knownGameState['opponentAuthority'] = self.opponent.authority
        self.knownGameState['opponentMustDiscard'] = self.opponent.mustDiscard
        # opScrambleDeckAndHand, opTopCard, and opHandCards are calculated dynamically
        
        return choose(self.name, options, self.knownGameState)

    def draw(self, n):
        for i in range(n):
            if len(self.deck) > 0:
                self.hand.append(self.deck.pop())
                if self.knownTopOfDeck > 0: 
                    self.opponentKnownHandCards.append(self.hand[-1])
                    self.knownTopOfDeck -= 1
            elif len(self.discardPile) > 0:
                # no more cards in the deck. use discard pile
                while len(self.discardPile) > 0:
                    self.deck.append(self.discardPile.pop())
                random.shuffle(self.deck)
                self.hand.append(self.deck.pop())
        self.calculateScrambleDeck()
    
    def discard(self, type, required):
        options = []
        discardCount = 0
        for i in range(len(self.hand)):
            options.append(('discard' + type,i,self.hand[i]))
        if required == False:
            options.append(('nodiscard',))
        decision = self.sendChoice(options)
        if options[decision] != ('nodiscard',):
            self.discardPile.append(self.hand.pop(decision)) 
            discardCount = 1
        return discardCount
   
    def takeTurn(self):
        self.calculateOpponentScrambleDeckAndHand()

        self.attack = 0
        self.trade = 0
        self.allAllied = False # for Mech World
        self.fleetActive = False # for FleetHQ
        self.nextShipTop = False # for several cards
        self.blobCardsPlayed = 0 # for Blob World

        for faction in factions:
            for card in self.cardsInPlay[faction]:
                card[1] = False
                card[2] = None

        # take care of bases starting in play
        for faction in factions:
            for i in range(len(self.cardsInPlay[faction])):
                self.activateCard(faction,i)

        # discard if required, and reduce mustDiscard
        if self.mustDiscard >= len(self.hand): 
            for i in range(len(self.hand)):
                self.discardPile.append(self.hand.pop())
            self.mustDiscard = 0
        while self.mustDiscard > 0:
            self.mustDiscard -= self.discard('Normal', True)

        mainPhase = True
        while mainPhase == True:
            # generate options
            options = []
            
            # can play any card
            for i in range(len(self.hand)):
                options.append(('play', i, self.hand[i]))
            
            opOutposts = []
            opBases = []

            for faction in factions:
                # identify scrap targets and abilityOption targets
                for i in range(len(self.cardsInPlay[faction])):
                    if self.cardsInPlay[faction][i][2] not in (None, 'used'):
                        options.append(('abilityOption', faction, i, self.cardsInPlay[faction][i][2]))
                    if self.cardsInPlay[faction][i][0][10] != 'none':
                        options.append(('scrapFromPlay', faction, i, self.cardsInPlay[faction][i], self.cardsInPlay[faction][i][0][10], self.cardsInPlay[faction][i][0][11]))
                
                # identify attack targets
                if self.attack > 0:
                    for i in range(len(self.opponent.cardsInPlay[faction])): # should all be bases/outposts 
                        if self.opponent.cardsInPlay[faction][i][0][6] == 'base':
                            opBases.append((faction,i,self.opponent.cardsInPlay[faction][i][0][12],self.opponent.cardsInPlay[faction][i][0]))
                        else:
                            opOutposts.append((faction,i,self.opponent.cardsInPlay[faction][i][0][12],self.opponent.cardsInPlay[faction][i][0]))

            if self.attack > 0:
                if len(opOutposts) > 0:
                    for outpostInfo in opOutposts:
                        if self.attack >= outpostInfo[2]:
                            #                           1faction        2index,         3shield amount, 4available attack
                            options.append(('attack',   outpostInfo[0], outpostInfo[1], outpostInfo[2], self.attack))
                else:
                    for baseInfo in opBases:
                        if self.attack >= baseInfo[2]:
                            options.append(('attack', baseInfo[0], baseInfo[1], baseInfo[2], self.attack))
                    options.append(('attackOpponent', self.attack))
               
            # identify acquisition targets
            if self.trade > 0:
                for i in range(5):
                    if self.game.tradeRow[i][1] <= self.trade:
                        options.append(('acquire', i, self.game.tradeRow[i]))
            
            if len(self.hand) == 0:
                options.append(('endTurn',))

            decision = self.sendChoice(options)
            match options[decision][0]:
                case 'play': self.playCard(options[decision][1])
                case 'abilityOption': self.triggerAbilityOption(options[decision][1], options[decision][2])
                case 'scrapFromPlay': self.scrapFromPlay(options[decision][1], options[decision][2], options[decision][4], options[decision][5])
                case 'attack': self.attackBase(options[decision][1], options[decision][2], options[decision][3])
                case 'attackOpponent':
                    self.opponent.authority -= self.attack
                    self.attack = 0
                    if self.opponent.authority <= 0:
                        mainPhase = False
                        self.game.winner = self
                case 'acquire': self.acquire(options[decision][1], options[decision][2][1])
                case 'endTurn': mainPhase = False
        
        # remove stuff from play area
        for faction in factions:
            i = 0
            while i < len(self.cardsInPlay[faction]):
                if self.cardsInPlay[faction][i][0][6] == 'ship':
                    if self.cardsInPlay[faction][i][4] == True: # it's a copyship
                        self.discardPile.append(('Stealth Needle',  4,      0,      0,      0,      'red',  'ship', 'copyship', 'none',         0,      'none',         0,      0,      1))
                    else:
                        self.discardPile.append(self.cardsInPlay[faction][i][0])
                    self.removeCardFromPlay(faction,i)
                else:
                    i += 1
        
        self.opponentKnownHandCards = []
        # draw for my next turn
        self.draw(5)

    def useAbility(self, abilityName, n = None):
        match abilityName:
            case 'gainattack': self.attack += n
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
                        if card[0][6] != 'ship': baseCount += 1
                if baseCount >= 2: self.draw(2)
            case 'rowscrap': self.selectAndScrapFromTradeRow()
            case '5ordraws':
                choice = self.sendChoice([('gainattack',5),('draw',self.blobCardsPlayed)])
                if choice == 0: self.attack += 5
                else: self.draw(self.blobCardsPlayed)
            case 'scrapany': self.scrapAny('Normal')
            case 'scraptwo':
                scraps = 0
                scraps += self.scrapAny('Draw')
                scraps += self.scrapAny('Draw')
                self.draw(scraps)
            case 'drawscrap':
                self.draw(1)
                self.mustScrapFromHand()
            case 'killbase': self.selectAndDestroyBase() 
            case '5or0or3':
                choice = self.sendChoice([('trade', 3),('gainattack', 5)])
                if choice == 0: self.trade += 3
                else: self.attack += 5
            case 'copyship':
                options = [('nocopy',)]
                copierIndex = None
                for faction in factions:
                    for i in range(len(self.cardsInPlay[faction])):
                        if self.cardsInPlay[faction][i][0][6] == 'ship':
                            if self.cardsInPlay[faction][i][0][7] == 'copyship':
                                copierIndex = i
                            else:
                                options.append(('copyship', self.cardsInPlay[faction][i][0]))
                decision = self.sendChoice(options)
                if decision > 0:
                    copiedCard = options[decision][1]
                    copiedFaction = 'red'
                    self.cardsInPlay['red'][copierIndex][0] = copiedCard # modifies the needle details
                    if copiedCard[5] != 'red': #then we need to move the ship
                        self.cardsInPlay[copiedCard[5]].append(self.cardsInPlay['red'][copierIndex])
                        self.cardsInPlay['red'].pop(copierIndex)
                        copiedFaction = copiedCard[5]
                        copierIndex = len(self.cardsInPlay[copiedFaction]) - 1
                    if copiedCard[5] == 'green':
                        self.blobCardsPlayed += 1
                    self.activateCard(copiedFaction, copierIndex)
            case 'recycle':
                totalDiscardCount = 0
                totalDiscardCount += self.discard('Draw', False)
                totalDiscardCount += self.discard('Draw', False)
                self.draw(totalDiscardCount)
            case '0or2or2':
                choice = self.sendChoice([('authority', 2),('trade', 2)])
                if choice == 0: self.authority += 2
                else: self.trade += 2
            case '2or3or0':
                choice = self.sendChoice([('gainattack', 2),('authority', 3)])
                if choice == 0: self.attack += 2
                else: self.authority += 3
            case '0or1or1':
                choice = self.sendChoice([('authority', 1),('trade', 1)])
                if choice == 0: self.authority += 1
                else: self.trade += 1
            case 'freebuy':
                options = []
                for i in range(5):
                    if self.game.tradeRow[i][0] != 'none' and self.game.tradeRow[i][6] == 'ship':
                        options.append(('freeAcquire', i, self.game.tradeRow[i]))
                if len(options) > 0:
                    self.nextShipTop = True
                    decision = self.sendChoice(options)
                    self.acquire(options[decision][1], 0)
            case 'destroyscrap':
                self.selectAndDestroyBase()
                self.selectAndScrapFromTradeRow()
            case 'drawdestroy':
                self.selectAndDestroyBase()
                self.draw(1)       
            case _: raise ValueError('Used unknown ability ' + str(abilityName))

    def selectAndDestroyBase(self):
        outpostsExist = False
        outpostOptions = []
        baseOptions = []
        options = []
        for faction in factions:
            for i in range(len(self.opponent.cardsInPlay[faction])): 
                if self.opponent.cardsInPlay[faction][i][0][6] == 'outp':
                    outpostsExist = True
                    outpostOptions.append(('killbase', faction, i, self.opponent.cardsInPlay[faction][i][0]))
                elif outpostsExist == False:
                    baseOptions.append(('killbase', faction, i, self.opponent.cardsInPlay[faction][i][0]))
        if outpostsExist == True:
            options = outpostOptions
        else:
            options = baseOptions
        options.append(('nokill',))
        decision = self.sendChoice(options)
        if decision < len(options) - 1: # then it's a kill
            self.opponent.discardPile.append(self.opponent.removeCardFromPlay(options[decision][1], options[decision][2])) 

    def selectAndScrapFromTradeRow(self):
        options = []
        for i in range(5):
            if self.game.tradeRow[i][0] != 'none':
                options.append(('rowscrap', i, self.game.tradeRow[i]))
        options.append(('noRowScrap',))
        decision = self.sendChoice(options)
        if options[decision][0] == 'rowscrap':
            self.removeFromTradeRow(options[decision][1])

    def triggerAbilityOption(self, faction, position):
        card = self.cardsInPlay[faction][position]
        if card[3] == True and card[2] != 'used': # card is still in play and the ability has not been used
            self.useAbility(card[2][1:]) # [1:] removes the '-' character at the beginning
            card[2] = 'used'

    def acquire(self, i, cost):
        target = self.game.tradeRow[i]
        
        # get the card
        if self.nextShipTop == True and target[6] == 'ship':
            self.deck.append(target)
            self.knownTopOfDeck += 1
            self.calculateScrambleDeck()
            self.nextShipTop = False
        else:
            self.discardPile.append(target)
        self.trade -= cost
        
        # show the next card
        self.removeFromTradeRow(i)

    def playCard(self, handId):
        cardDetails = self.hand.pop(handId)
        #                                        0              1               2               3       4 
        #                                                       allyTriggeredIt,abilityOption,  inPlay, isStealth
        self.cardsInPlay[cardDetails[5]].append([cardDetails,   False,          None,           True,   cardDetails[7] == 'copyship'])
        if cardDetails[5] == 'green':
            self.blobCardsPlayed += 1
        self.activateCard(cardDetails[5], len(self.cardsInPlay[cardDetails[5]]) - 1)

    def attackBase(self, faction, position, shieldAmount):
        self.attack -= shieldAmount
        self.opponent.discardPile.append(self.opponent.removeCardFromPlay(faction, position))
        
    def removeCardFromPlay(self, faction, position):
        card = self.cardsInPlay[faction][position]
        if card[0][7] == 'allally': self.allAllied = False
        if card[0][7] == 'fleethq': self.fleetActive = False
        self.cardsInPlay[faction][position][3] = False
        self.cardsInPlay[faction].pop(position)
        return card[0]
    
    def activateCard(self, faction, position):
        card = self.cardsInPlay[faction][position]
        self.attack += card[0][2]
        self.authority += card[0][3]
        self.trade += card[0][4]
        
        # use its ability
        if card[0][7][0] == '-' and card[2] == None: # the ability can be activated by the user at any time
            card[2] = card[0][7] # card[2] is the active ability option
        elif card[0][7][0] == '-' and card[2] != None:
            raise ValueError('Option ability ' + str(card[0][7]) + ' could not be added because of an existing option: ' + str(card[2]))
        elif card[0][7] != 'none': 
            self.useAbility(card[0][7])
        
        # trigger other ally cards
        def triggerOtherAllyCards(faction,position):
            for i in range(len(self.cardsInPlay[faction])):
                allyAbility = self.cardsInPlay[faction][i][0][8]
                if i != position and allyAbility != 'none' and self.cardsInPlay[faction][i][1] == False: # the card is not this one and it has an ally ability and it hasn't been triggered
                    if allyAbility[0] == '-' and self.cardsInPlay[faction][i][2] == None: # the ability of the other card can be activated by the user at any time and hasn't already been added
                        self.cardsInPlay[faction][i][2] = allyAbility # card[2] is the active ability option
                    elif allyAbility[0] == '-' and self.cardsInPlay[faction][i][2] != None:
                        raise ValueError('Option ability ' + str(allyAbility) + ' could not be added because of an existing option: ' + str(self.cardsInPlay[faction][i][2]))
                    else:
                        self.useAbility(allyAbility, self.cardsInPlay[faction][i][0][9])
                    self.cardsInPlay[faction][i][1] = True # mark that that card's ally ability has already been used now
        triggerOtherAllyCards(faction,position)
        if card[0][7] == 'allally':
            triggerOtherAllyCards('blue',-1)
            triggerOtherAllyCards('green',-1)
            triggerOtherAllyCards('yellow',-1)
        if card[4] == True and faction != 'red': # then it's using the copyship and should also trigger red faction
            triggerOtherAllyCards('red',position)
        if self.fleetActive == True and card[0][6] == 'ship':
            self.attack += 1
        
        # trigger its own ally ability
        if card[1] == False and card[0][8] != 'none' and (self.allAllied == True or len(self.cardsInPlay[faction]) > 1):
            # ally ability is activated
            if card[0][8][0] == '-' and card[2] == None:  # the ability can be activated by the user at any time
                card[2] = card[0][8]
            elif card[0][8][0] == '-' and card[2] != None:
                raise ValueError('Option ability ' + str(card[0][8]) + ' could not be added because of an existing option: ' + str(card[2]))
            else:
                self.useAbility(card[0][8],card[0][9])
            card[1] = True # the ally ability has now been used 
    
    def scrapFromPlay(self, faction, position, ability, abilityN):
        self.useAbility(ability, abilityN)
        self.removeCardFromPlay(faction, position)
    
    def scrapAny(self, type, required=False):
        options = []
        totalScrapped = 0
        for i in range(len(self.hand)):
            options.append(('scrapFromHand' + type, i, self.hand[i]))
        for i in range(len(self.discardPile)):
            options.append(('scrapFromDiscard' + type, i, self.discardPile[i]))
        if required == False:
            options.append(('noScrapFromHand',))
        decision = self.sendChoice(options)
        if options[decision][0] == 'scrapFromHand' + type:
            self.hand.pop(options[decision][1])
            totalScrapped += 1
        elif options[decision][0] == 'scrapFromDiscard' + type:
            self.discardPile.pop(options[decision][1])
            totalScrapped += 1
        return totalScrapped
    
    def mustScrapFromHand(self):
        options = []
        for i in range(len(self.hand)):
            options.append(('scrapFromHandNormal', i, self.hand[i]))
        if len(options) > 0:
            self.hand.pop(self.sendChoice(options))

    def removeFromTradeRow(self, i):
        if i == 0: #explorer
            self.game.explorers -= 1
            if self.game.explorers == 0:
                self.game.tradeRow[0] = ('none', 1000) # signifies it cannot be acquired
            else:
                self.game.tradeRow[0] = explorer
        else:
            if len(self.game.readyCards) == 0:
                self.game.tradeRow[i] = ('none', 1000)
            else:
                self.game.tradeRow[i] = self.game.readyCards.pop()

    def endTurn(self):
        for faction in factions:
            i = 0
            while i < len(self.cardsInPlay[faction]):
                if self.cardsInPlay[faction][i][0][6] == 'ship':
                    self.discardPile.append(self.cardsInPlay[faction].pop(i)[0])
                else:
                    i += 1
                

        


class Game:
    def __init__(self,p1name,p2name):
        self.tradeRow = []
        self.players = [Player(self,p1name),Player(self,p2name)]
        self.players[0].opponent = self.players[1]
        self.players[1].opponent = self.players[0]
        self.players[0].knownGameState['opponentDiscardPile'] = self.players[1].discardPile
        self.players[0].knownGameState['opponentCardsInPlay'] = self.players[1].cardsInPlay
        self.players[1].knownGameState['opponentDiscardPile'] = self.players[0].discardPile
        self.players[1].knownGameState['opponentCardsInPlay'] = self.players[0].cardsInPlay
        random.shuffle(self.players)
        self.players[0].draw(3)
        self.players[1].draw(5)
        self.explorers = 10
        self.readyCards = []
        for card in cardDetails:
            for i in range(card[13]):
                self.readyCards.append(card) #12 --> count
        random.shuffle(self.readyCards)
        self.tradeRow.append(explorer)
        for i in range(4):
            self.tradeRow.append(self.readyCards.pop())
        self.nextPlayer = 0
        self.winner = None
        while self.winner == None:
            self.players[self.nextPlayer].takeTurn()
            self.nextPlayer = (self.nextPlayer + 1) % 2
        print('The winner is: ' + self.winner.name + "!")

        

        
