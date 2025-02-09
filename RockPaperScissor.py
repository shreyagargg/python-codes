'''
python basics
date - April 26 , 2024
day - 16
'''
import random as r
l = []

''' 0 - rock
1-paper
2-scissor '''
while(True):
    print('Choose any one :-\n1 : rock\t2 : scissor\n3 : paper\t4 : exit the game')
    c = r.randint(1,99)
    # print(c)
    comp = c%3
    print(comp)
    user = int(input())
    if(user==4):
        break
    elif((user==1 and comp==1) or (user==2 and comp==0) or (user==3 and comp==2)):
        print('Computer won')
        l.append(-1)
    elif((user==1 and comp==2) or (user==2 and comp==1) or (user==3 and comp==0)):
        print('You won')
        l.append(1)
    else:
        print('')
        l.append(0)
print('Computer won :',l.count(-1))
print('you won :',l.count(1))
print('Match drawn :',l.count(0))
