import nltk
from nltk.parse import RecursiveDescentParser
grammar = nltk.CFG.fromstring("\n"
                              "S  -> NP VP\n"
                              "NP -> Det N | PN \n"
                              "NP -> PN Conj NP | Det AdjP N\n"
                              "VP -> V Adj NP | V VP NP | V Adv \n"
                              "VP -> V NP PP | V Conj V NP | V NP\n"
                              "PP -> P NP\n"
                              "AdjP -> Adv Adv Adv Adj\n"
                              "Det -> 'the' | 'a' | 'The'\n"
                              "N -> 'sandwich' | 'president' | 'waiter' | 'chairs' | 'tables'\n"
                              "PN -> 'Oscar'| 'Sally' | 'Mary' | 'Bob'\n"
                              "Adj  -> 'eating' | 'perplexed' \n"
                              "V ->  'ate'  | 'wanted' | 'is' | 'died' | 'put' | 'called' | 'saw'\n"
                              "P -> 'on'\n"
                              "Adv -> 'suddenly' | 'very' | 'lazy'\n"
                              "Conj -> 'and'\n"
                              )
rd= nltk.RecursiveDescentParser(grammar)
sent = "Mary saw Bob".split()
for tree in rd.parse(sent):
    print(tree)
sent1 = "Sally ate a sandwich".split()
for tree in rd.parse(sent1):
    print(tree)
sent2 = "Sally and the president wanted and ate a sandwich".split()
for tree in rd.parse(sent2):
    print(tree)
sent3 = "the very very very perplexed president ate a sandwich".split()
for tree in rd.parse(sent3):
    print(tree)
sent4 = "Sally is lazy".split()
for tree in rd.parse(sent4):
    print(tree)
sent5 = "Oscar died suddenly".split()
for tree in rd.parse(sent5):
    print(tree)
sent6 = "The waiter put the chairs on the tables".split()
for tree in rd.parse(sent6):
    print(tree)
sent7 = "Oscar called the waiter".split()
for tree in rd.parse(sent7):
    print(tree)
sent8 = "Sally is eating a sandwich".split()
for tree in rd.parse(sent8):
    print(tree)


