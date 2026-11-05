grid = [[(i * j) % 7 for j in range(300)] for i in range(300)]
print(sum(1 for i in range(300) for j in range(300) if grid[i][j] % 2 == 1))
