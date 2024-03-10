from pprint import pprint
import random
import nltk

knowledge = { ("person1", "name", "?"), \
              ("person1", "town", "?"),
              ("person1", "street", "?") }

active = True
while active:
    unknowns = { (person,fact,value) for (person,fact,value) \
                 in knowledge if value=="?" }
    print("UNKNOWN:")
    pprint(unknowns)
    print("KNOWN:")
    pprint(knowledge - unknowns)
    if unknowns: #is non-empty
        currentQuery = random.choice(list(unknowns))  # choose an item from unknowns at random
        question = "What is your " + currentQuery[1] + ": "  # set question to ask about a factt.
        knowledge.remove(currentQuery)  # remove from what we do not know since we are about to find out.
        reply = input(question)
        # to fill in - process replys
        # knowledge.add((currentQuery[0], currentQuery[1], reply))  # update what we know with the value in reply.
        # Prev processing was crude. What if reply is a sentence, for instance? Therefore, it needs further processing.
        listOfWords = nltk.word_tokenize(reply)  # break down reply into tokens
        taggedListOfWords = nltk.pos_tag(listOfWords)  # returns tuples with words and grammatical functions
        for taggedWord in taggedListOfWords:
            if taggedWord[1] == "NNP":  # if the word is tagged as a proper noun.
                knowledge.add((currentQuery[0], currentQuery[1], taggedWord[0]))  # Now the bot is a little smarter.

    else:
        question = "How can I help you? "
        helpRequest = input(question)
        # to fill in - process reply
    print()
