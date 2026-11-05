s = ' '.join(map(str, range(100000)))
print(sum(int(x) for x in s.split(' ')))
