import nltk
grammar1 = nltk.CFG.fromstring("""
S -> NP VP
PP -> P NP
V -> "saw" | "ate" | "walked" | "died" | "called" | "wanted" | "put" | "eating" | "lazy" | "perplexed"
VP -> V NP | V NP PP | V Adv
PN -> "John" | "Mary" | "Bob" | "Oscar" | "Paris" | "Sally"
NP -> Det N | Det N PP | PN
Det -> "a" | "an" | "the" | "my" | "is"
N -> "man" | "dog" | "cat" | "telescope" | "park" | "president" | "chair" | "sandwich"| "waiter"
P -> "in" | "on" | "by" | "with"
Adv -> "suddenly" | "quickly" | "slowly" |"very" 
""")
sent = 'Sally ate a sandwich'.split()
rd_parser = nltk.RecursiveDescentParser(grammar1)
for tree in rd_parser.parse(sent):
    print(tree)