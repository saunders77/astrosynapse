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
    ('Flagship',        6,      5,      0,      0,      'blue', 'ship', 'draw',     'authority',    5,      'none',         0,      0,      1),
    ('Freighter',       4,      0,      0,      4,      'blue', 'ship', 'none',     'shiptop',      0,      'none',         0,      0,      2),
    ('Port of Call',    6,      0,      0,      3,      'blue', 'outp', 'none',     'none',         0,      'drawdestroy',  0,      6,      1),
    ('Trade Escort',    5,      4,      4,      0,      'blue', 'ship', 'none',     'draw',         0,      'none',         0,      0,      1),
    ('Trading Post',    3,      0,      0,      0,      'blue', 'outp', '-0or1or1', 'none',         0,      'gainattack',   3,      4,      2),

]
scout = cardDetails[0]
viper = cardDetails [1]
explorer = cardDetails[2]
factions = ('red','green','blue','yellow','none')

def isFirstEqualOrBetter(first, second):
    if first == second: return True
    
    if first[2] < second[2]: return False # attack must be greater or equal
    if first[3] < second[3]: return False # authority
    if first[4] < second[4]: return False # trade
    if first[9] < second[9]: return False # ally N
    if first[11] < second[11]: return False # scrap N
    if first[12] < second[12]: return False # shield
    if second[5] != 'none' and first[5] != second[5]: return False # colours
    if first[6] != second[6] and (first[6] != 'outp' or second[6] != 'base'): return False 
    if second[7] != 'none' and first[7] != second[7]: return False # ability
    if second[8] != 'none' and first[8] != second[8]: return False # ally ability
    if second[10] != 'none' and first[10] != second[10]: return False #scrap

    return True



import random
from chooser import choose as default_choose

class Player:
    def __init__(self, game, name, chooser_fn=None, turn_summary_callback=None):
        self.name = name
        self.game = game
        self.chooser_fn = chooser_fn or default_choose
        self.turn_summary_callback = turn_summary_callback
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
        split_index = len(self.deck) - self.knownTopOfDeck
        scrambleDeck = list(self.deck[:split_index])
        random.shuffle(scrambleDeck)
        self.knownGameState['scrambleDeck'] = scrambleDeck
        self.knownGameState['topCards'] = list(self.deck[split_index:])
    
    def calculateOpponentScrambleDeckAndHand(self):
        opponent_deck = self.opponent.deck
        split_index = len(opponent_deck) - self.opponent.knownTopOfDeck
        opScrambleDeckHand = list(opponent_deck[:split_index])
        opTopCards = list(opponent_deck[split_index:])
        knownOpponentHandCards = self.opponent.opponentKnownHandCards
        opHandTracker = []
        for item in knownOpponentHandCards: opHandTracker.append(item)
        for handCard in self.opponent.hand:
            if handCard in opHandTracker:
                opHandTracker.remove(handCard) # don't add it to the big scrambled list
            else:
                opScrambleDeckHand.append(handCard)
        random.shuffle(opScrambleDeckHand)
        self.knownGameState['opponentScrambleDeckAndHand'] = opScrambleDeckHand
        self.knownGameState['opponentTopCards'] = opTopCards
        self.knownGameState['opponentHandCards'] = knownOpponentHandCards

    def sendChoice(self, options):
        knownGameState = self.knownGameState
        opponent = self.opponent
        knownGameState['authority'] = self.authority
        knownGameState['attack'] = self.attack
        knownGameState['trade'] = self.trade
        knownGameState['mustDiscard'] = self.mustDiscard
        # scrambleDeck and topCards are calculated dynamically        
        knownGameState['nextShipTop'] = self.nextShipTop
        knownGameState['blobPlayCount'] = self.blobCardsPlayed
        knownGameState['opponentAuthority'] = opponent.authority
        knownGameState['opponentMustDiscard'] = opponent.mustDiscard
        # opScrambleDeckAndHand, opTopCard, and opHandCards are calculated dynamically
        
        return self.chooser_fn(self.name, options, knownGameState)

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
        discard_action = 'discard' + type
        options = []
        for i in range(len(self.hand)):
            discardCandidate = True
            j = 0
            while discardCandidate == True and j < len(self.hand):
                if i != j and isFirstEqualOrBetter(self.hand[i], self.hand[j]) == True and self.hand[i] != self.hand[j]: # j must be worse. so don't discard i
                    discardCandidate = False
                j += 1
            if discardCandidate == True:
                options.append((discard_action, i, self.hand[i]))
        if required == False:
            options.append(('nodiscard',))
        decision = self.sendChoice(options)
        if options[decision] != ('nodiscard',):
            self.discardPile.append(self.hand.pop(options[decision][1]))
            return 1
        return 0
   
    def takeTurn(self):
        self.calculateOpponentScrambleDeckAndHand()

        self.attack = 0
        self.trade = 0
        if self.turn_summary_callback is not None:
            self.turnAcquisitionEvents = []
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
        actionCount = 0
        while mainPhase == True:
            actionCount += 1
            if actionCount > self.game.max_actions_per_turn:
                mainPhase = False
                break

            # generate options
            options = []
            append_option = options.append
            hand = self.hand
            cards_in_play = self.cardsInPlay
            opponent_cards_in_play = self.opponent.cardsInPlay
            current_attack = self.attack
            current_trade = self.trade
            
            # can play any card
            for i, card in enumerate(hand):
                append_option(('play', i, card))
            
            opOutposts = []
            opBases = []

            for faction in factions:
                faction_cards = cards_in_play[faction]
                # identify scrap targets and abilityOption targets
                for i, playCard in enumerate(faction_cards):
                    if playCard[2] not in (None, 'used'):
                        append_option(('abilityOption', faction, i, playCard[2]))
                    if playCard[0][10] != 'none':
                        append_option(('scrapFromPlay', faction, i, playCard, playCard[0][10], playCard[0][11]))
                
                # identify attack targets
                if current_attack > 0:
                    for i, opponentCard in enumerate(opponent_cards_in_play[faction]): # should all be bases/outposts 
                        if opponentCard[0][6] == 'base':
                            opBases.append((faction, i, opponentCard[0][12], opponentCard[0]))
                        else:
                            opOutposts.append((faction, i, opponentCard[0][12], opponentCard[0]))

            if current_attack > 0:
                if len(opOutposts) > 0:
                    for outpostInfo in opOutposts:
                        if current_attack >= outpostInfo[2]:
                            #                           1faction        2index,         3shield amount, 4available attack
                            append_option(('attack', outpostInfo[0], outpostInfo[1], outpostInfo[2], current_attack))
                else:
                    for baseInfo in opBases:
                        if current_attack >= baseInfo[2]:
                            append_option(('attack', baseInfo[0], baseInfo[1], baseInfo[2], current_attack))
                    append_option(('attackOpponent', current_attack))
               
            # identify acquisition targets
            if current_trade > 0:
                tradeRow = self.game.tradeRow
                for i, tradeCard in enumerate(tradeRow):
                    if tradeCard[1] <= current_trade:
                        append_option(('acquire', i, tradeCard))
            
            if len(hand) == 0:
                append_option(('endTurn',))

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
                case 'acquire':
                    if self.turn_summary_callback is not None:
                        self.recordAcquisitionEvent(
                            'acquire',
                            options[decision][2],
                            self.game.tradeRow,
                            options[decision][2][1],
                            current_trade,
                        )
                    self.acquire(options[decision][1], options[decision][2][1])
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
        self.finishTurnSummary()
        # draw for my next turn
        self.draw(5)

    def recordAcquisitionEvent(self, acquisitionType, card, tradeRowSnapshot, cost, tradeAvailable):
        if self.turn_summary_callback is None:
            return
        self.turnAcquisitionEvents.append({
            'type': acquisitionType,
            'cardName': card[0],
            'cardCost': card[1],
            'costPaid': cost,
            'tradeAvailable': tradeAvailable,
            'tradeRowSnapshot': list(tradeRowSnapshot),
        })

    def finishTurnSummary(self):
        if self.turn_summary_callback is None:
            return
        turn_acquisition_events = self.turnAcquisitionEvents
        total_trade_gained = self.trade + sum(event['costPaid'] for event in turn_acquisition_events)
        self.turn_summary_callback({
            'playerName': self.name,
            'acquisitionEvents': list(turn_acquisition_events),
            'totalAcquisitions': len(turn_acquisition_events),
            'remainingTrade': self.trade,
            'totalTradeGained': total_trade_gained,
        })

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
                                copyCandidate = True
                                for j in range(len(self.cardsInPlay[faction])):
                                    if i != j and isFirstEqualOrBetter(self.cardsInPlay[faction][j][0], self.cardsInPlay[faction][i][0]) == True and self.cardsInPlay[faction][i][0] != self.cardsInPlay[faction][j][0]: # j is strictly better than this one
                                        copyCandidate = False
                                if copyCandidate == True:
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
                choice = self.sendChoice([('trade', 1),('switch', 0)])
                if choice == 0: self.trade += 1
                else:
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
                    if self.turn_summary_callback is not None:
                        self.recordAcquisitionEvent('freeAcquire', options[decision][2], self.game.tradeRow, 0, self.trade)
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
        opponent_cards_in_play = self.opponent.cardsInPlay
        for faction in factions:
            for i, opponentCard in enumerate(opponent_cards_in_play[faction]): 
                if opponentCard[0][6] == 'outp':
                    outpostsExist = True
                    outpostOptions.append(('killbase', faction, i, opponentCard[0]))
                elif outpostsExist == False:
                    baseOptions.append(('killbase', faction, i, opponentCard[0]))
        options = outpostOptions if outpostsExist else baseOptions
        options.append(('nokill',))
        decision = self.sendChoice(options)
        if decision < len(options) - 1: # then it's a kill
            self.opponent.discardPile.append(self.opponent.removeCardFromPlay(options[decision][1], options[decision][2])) 

    def selectAndScrapFromTradeRow(self):
        options = [('rowscrap', i, tradeCard) for i, tradeCard in enumerate(self.game.tradeRow) if tradeCard[0] != 'none']
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
        scrap_from_hand = 'scrapFromHand' + type
        scrap_from_discard = 'scrapFromDiscard' + type
        
        options = []
        for i in range(len(self.discardPile)):
            scrapCandidate = True
            for j in range(len(self.discardPile)):
                if j != i and isFirstEqualOrBetter(self.discardPile[i], self.discardPile[j]) == True and self.discardPile[i] != self.discardPile[j]:
                    scrapCandidate = False
            if scrapCandidate == True:
                options.append((scrap_from_discard, i, self.discardPile[i]))
        for i in range(len(self.hand)):
            scrapCandidate = True
            for j in range(len(options)):
                if self.hand[i] == options[j][2]: scrapCandidate = False
                if isFirstEqualOrBetter(self.hand[i], options[j][2]) == True: scrapCandidate = False
            if scrapCandidate == True:
                for j in range(len(self.hand)):
                    if j != i and isFirstEqualOrBetter(self.hand[i], self.hand[j]) == True and self.hand[i] != self.hand[j]:
                        scrapCandidate = False
                if scrapCandidate == True:
                    options.append((scrap_from_hand, i, self.hand[i]))

        if required == False and len(options) == 0:
            return 0
        if required == False:
            options.append(('noScrapFromHand',))
        decision = self.sendChoice(options)
        if options[decision][0] == scrap_from_hand:
            self.hand.pop(options[decision][1])
            return 1
        if options[decision][0] == scrap_from_discard:
            self.discardPile.pop(options[decision][1])
            return 1
        return 0
    
    def mustScrapFromHand(self):
        options = []
        for i in range(len(self.hand)):
            scrapCandidate = True
            for j in range(len(self.hand)):
                if j != i and isFirstEqualOrBetter(self.hand[i], self.hand[j]) == True and self.hand[i] != self.hand[j]:
                    scrapCandidate = False
            if scrapCandidate == True:
                options.append(('scrapFromHandNormal', i, self.hand[i]))
        if len(options) > 0:
            self.hand.pop(options[self.sendChoice(options)][1])

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
    def __init__(self, p1name, p2name, p1_choose=None, p2_choose=None, verbose=True, max_turns=400, max_actions_per_turn=200, turn_summary_callback=None):
        self.verbose = verbose
        self.max_turns = max_turns
        self.max_actions_per_turn = max_actions_per_turn
        self.tradeRow = []
        self.players = [
            Player(self, p1name, p1_choose, turn_summary_callback=turn_summary_callback),
            Player(self, p2name, p2_choose, turn_summary_callback=turn_summary_callback),
        ]
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
        for i in range(5):
            self.tradeRow.append(self.readyCards.pop())
        self.nextPlayer = 0
        self.winner = None
        self.ended_by_limit = False
        self.turnsTaken = 0
        while self.winner == None and self.turnsTaken < self.max_turns:
            self.players[self.nextPlayer].takeTurn()
            self.turnsTaken += 1
            if self.winner == None:
                self.nextPlayer = (self.nextPlayer + 1) % 2
        if self.winner == None:
            self.ended_by_limit = True
            self.winner = self.resolveStalemate()
        if self.verbose:
            print('The winner is: ' + self.winner.name + "!")

    def resolveStalemate(self):
        def playerScore(player):
            score = player.authority
            for card in player.deck:
                score += card[1] + card[2] + card[4] + card[12]
            for card in player.hand:
                score += card[1] + card[2] + card[4] + card[12]
            for card in player.discardPile:
                score += card[1] + card[2] + card[4] + card[12]
            for faction in factions:
                for cardInPlay in player.cardsInPlay[faction]:
                    card = cardInPlay[0]
                    score += card[1] + card[2] + card[4] + card[12] + 1
            return score

        p1 = self.players[0]
        p2 = self.players[1]
        p1Score = playerScore(p1)
        p2Score = playerScore(p2)
        if p1.authority != p2.authority:
            return p1 if p1.authority > p2.authority else p2
        if p1Score != p2Score:
            return p1 if p1Score > p2Score else p2
        return p1 if p1.name < p2.name else p2

if __name__ == "__main__":
    Game('michael', 'caitlin')

        

        
