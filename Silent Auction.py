
'''
python basics
date - April 26 , 2024
day - 16
'''
import os
d = {}
def details():
    name = input('Enter your name ')
    bid = int(input("Enter your bid "))
    d[name] = bid
    print('Any other bidder(0 for no , 1 for yes) : ')
    b = int(input())
    if(b==0):
        return
    else:
        os.system('cls')
        details()
details()

# print(d)
l = list(d.values())
# print(l)
m = max(l)
c = []
os.system('cls')
for i,j in d.items():
    if j==m:
        c.append(i)
        # print(i)
        # d.pop(i)

print('Highest bider/bidders are with value of',m)
for i in c:
    print(i)
