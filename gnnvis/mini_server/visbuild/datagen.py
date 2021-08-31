import random

f = open("eweights-2.txt", "w")

i=1
random.seed(1)
while i<=13264:
    y=random.random()
    print(round(y,1),end=', ', file = f)
    i=i+1

f.close()