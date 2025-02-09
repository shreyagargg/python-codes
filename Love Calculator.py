'''
python basics
date - May 17 , 2024
day - 5
'''
import os

print('''       Love calculator
    A beautiful way to calculate your love
    Answer some questions for yourself''')
ans1 = []
ans2 = []
def calC():
    a = input('When is your birthday(DD/MM/YYYY) ')
    ans1.append(a)
    a = input('when you both met(DD/MM/YYYY) ')
    ans1.append(a)
    a = input('what is your favourite colour ')
    ans1.append(a)
    a = input('The most affordable gift you loves the most ')
    ans1.append(a)
    a = input('Can you cry freely in front of your partner ')
    ans1.append(a)
    print("Your partner's turn")

    os.system('cls')
    a = input('When is your partner birthday(DD/MM/YYYY) ')
    ans2.append(a)
    a = input('when you both met(DD/MM/YYYY) ')
    ans2.append(a)
    a = input('what is your partner favourite colour ')
    ans2.append(a)
    a = input('The most affordable gift your partner loves the most ')
    ans2.append(a)
    a = input('Can your partner cry freely in front of your partner ')
    ans2.append(a)
    print("Now answer about yourself")
    count = 0
    for i in range(0,5):
        if(ans1[i] == ans2[i]):
            count += 1
        else:
            pass
    return count

d = calC()
b = calC()
os.system('cls')
print('Time for results : ')
print('Love percentage :',(b+d)*10)

  