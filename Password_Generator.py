'''
python basics
date - April 15 , 2024
day - 13
'''
# combination of alphanumerics and symbols
import random
import string
n = int(input('enter length of password : '))
'''l = string.ascii_lowercase
u = string.ascii_uppercase
d = string.digits
p = string.punctuation'''
# str = l+u+d+p
# print(str)
str = string.printable
# print(len(str))
pwd = random.sample(str,n)
# print(pwd)
ps = []
for i in range(0,n):
    i = random.randint(0,len(str))
    a = str[i]
    ps.append(a)
# print(ps)
for i in ps:
    print(i,end='')

print()