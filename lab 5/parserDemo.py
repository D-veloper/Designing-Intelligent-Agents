import nltk
from stat_parser import Parser
from pprint import pprint

parser = Parser()
sentence = "My street is Triumph Road"
t = parser.parse(sentence)
t.draw()
pprint(t)
