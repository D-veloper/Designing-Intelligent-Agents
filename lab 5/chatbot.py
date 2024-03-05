from pprint import pprint

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
        question = "## to fill in ## "
        reply = input(question)
        # to fill in - process reply
    else:
        question = "How can I help you? "
        helpRequest = input(question)
        # to fill in - process reply
    print()
