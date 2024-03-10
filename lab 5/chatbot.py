from pprint import pprint
import random

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
        knowledge.add((currentQuery[0], currentQuery[1], reply))  # update what we know with the value in reply.
    else:
        question = "How can I help you? "
        helpRequest = input(question)
        # to fill in - process reply
    print()
