with open("input.txt", "r") as f:
    levels = [list(map(int, line.split())) for line in f]

safes = 0

for level in levels:
    increasing = None
    is_safe = True

    for i in range(len(level) - 1):
        diff = level[i + 1] - level[i]
        
        if increasing is None:
            increasing = diff > 0
        
        if (increasing and diff <= 0) or (not increasing and diff >= 0):
            is_safe = False
            break
        
        if not (1 <= abs(diff) <= 3):
            is_safe = False
            break

    if is_safe:
        safes += 1

print(safes)