def choose(playerName, options, state):
    decisionIndex = 0
    print('Authority: ' + str(state['authority']) + '/' + str(state['opponentAuthority']))
    print('Attack: ' + str(state['attack']) + '. Trade: ' + str(state['trade']))
    cardsInPlayStr = 'Cards in play: '
    for factionCards in state['cardsInPlay'].values():
        for card in factionCards:
            cardsInPlayStr += str(card[0][1]) + card[0][0] + '('
            if card[2] not in (None, 'none'): cardsInPlayStr += card[2]
            if card[4] == True: cardsInPlayStr += ',Stealth'
            cardsInPlayStr += '), '
    print(cardsInPlayStr)
    opcardsInPlayStr = 'Opponent cards in play: '
    for factionCards in state['opponentCardsInPlay'].values():
        for card in factionCards:
            opcardsInPlayStr += str(card[0][1]) + card[0][0] + '('
            if card[2] not in (None, 'none'): opcardsInPlayStr += card[2]
            if card[4] == True: opcardsInPlayStr += ',Stealth'
            opcardsInPlayStr += '), '
    print(opcardsInPlayStr)
    tradeStr = 'Trade row: '
    for card in state['tradeRow']:
        tradeStr += str(card[1]) + card[0] + ', '
    print(tradeStr)
    handStr = 'Hand: '
    for card in state['hand']:
        handStr += str(card[1]) + card[0] + ', '
    print(handStr)
    deckStr = 'Deck: '
    for card in state['scrambleDeck']:
        deckStr += str(card[1]) + card[0] + ', '
    for card in state['topCards']:
        deckStr += str(card[1]) + card[0] + '(top), '
    print(deckStr)
    discardStr = 'Discard pile: '
    for card in state['discardPile']:
        discardStr += str(card[1]) + card[0] + ', '
    print(discardStr)
    opDeckHandStr = 'Opponent deck+hand: '
    for card in state['opponentScrambleDeckAndHand']:
        opDeckHandStr += str(card[1]) + card[0] + ', '
    for card in state['opponentTopCards']:
        opDeckHandStr += str(card[1]) + card[0] + '(top), '
    if len(state['opponentHandCards']) > 0:
        opDeckHandStr += '. Hand contains: '    
        for card in state['opponentHandCards']:
            opDeckHandStr += str(card[1]) + card[0] + ', '
    print(opDeckHandStr)
    opdiscardStr = 'Opponent discard pile: '
    for card in state['opponentDiscardPile']:
        opdiscardStr += str(card[1]) + card[0] + ', '
    print(opdiscardStr)
    print('Must discard: ' + str(state['mustDiscard']) + ', Opponent must discard: ' + str(state['opponentMustDiscard']) + ', nextShipTop: ' + str(state['nextShipTop']) + ', blobPlayCount: ' + str(state['blobPlayCount']))
    print('Select an option:')
    print(str(options))
    decisionIndex = int(input())
    return decisionIndex
