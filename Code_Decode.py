'''
python basics
date - Mar 29 , 2024
day - 8
'''

#  coding:
# if the word contains atleast 3 characters, remove the first letter and append it at the end
#   now append three random characters at the starting and the end
# else:
#   simply reverse the string

# Decoding:
# if the word contains less than 3 characters, reverse it
# else:
#   remove 3 random characters from start and end. Now remove the last letter and append it to the beginning

# Your program should ask whether you want to code or decode
print('Enter message :- ')
msg = input()
l = []
word = msg.split(' ')
print('what do you want\n1. code\t\t2. decode')
a = int(input())
if(a==1):
    for i in word:
      length = len(i)
      if(length < 3):
        i0 = i[::-1]
        l.append(i0)
      else:
        s1 = 'abc'
        s2 = 'xyz'
        i0 = s1 + i[1:] + i[0] + s2
        l.append(i0)
    print('coded msg -')
    for i in l:
       print(i,end=' ')
 
if(a==2):
    for i in word:
      length = len(i)
      if(length < 3):
        i0 = i[::-1]
        l.append(i0)
      else:
        i0 = i[-4]+i[3:-4]
        l.append(i0)
    print('decoded msg -')
    for i in l:
       print(i,end=' ')
   