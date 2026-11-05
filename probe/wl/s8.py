v = [0] * 1000
for i in range(200000):
    v[i % 1000] = i
print(sum(v))
