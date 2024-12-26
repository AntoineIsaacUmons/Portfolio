"""--- Part One ---"""

with open('input.txt', 'r') as file:
    lines = file.readlines()


left_list = []
right_list = []

for line in lines:
    nums = line.strip().split()
    if len(nums) == 2:
        left_list.append(int(nums[0]))
        right_list.append(int(nums[1]))


sorted_left_list = sorted(left_list)
sorted_right_list = sorted(right_list)

total_distance = sum(abs(a - b) for a, b in zip(sorted_left_list, sorted_right_list))

print("Part One : ", total_distance)


"""--- Part Two --- """

from collections import Counter

right_counts = Counter(sorted_right_list)

similarity = 0
for e in sorted_left_list:
    similarity += e * right_counts[e]  

print("Part Two : ", similarity)
