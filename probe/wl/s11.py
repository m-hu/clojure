n = 20000
words = ['w' + str((i * 31) % 5000) for i in range(n)]
freq = {}
for w in words:
    freq[w] = freq.get(w, 0) + 1
print(len(freq), max(freq.values()))
