"""--- Part One ---"""

with open("input.txt", "r") as f:
    levels = [list(map(int, line.split())) for line in f]

safes = 0

for report in levels:
    increasing = None
    is_safe = True

    for i in range(len(report) - 1):
        diff = report[i + 1] - report[i]
        
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


"""--- Part Two ---"""

def is_safe(report, skip_index=None):
    if skip_index is not None:
        report = report[:skip_index] + report[skip_index+1:]

    increasing = None
    for i in range(len(report) - 1):
        diff = report[i + 1] - report[i]
        if increasing is None:
            increasing = diff > 0
        if (increasing and diff <= 0) or (not increasing and diff >= 0):
            return False
        if not (1 <= abs(diff) <= 3):
            return False
    return True

safes = 0

for report in levels:
    if is_safe(report):  
        safes += 1
    else:
        for i in range(len(report)):
            if is_safe(report, i):
                safes += 1
                break

print(safes)
