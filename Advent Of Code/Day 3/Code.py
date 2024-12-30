import re

def parse_mul_instruction(line):
    matches = re.findall(r'mul\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)', line)
    total = 0
    for match in matches:
        x, y = map(int, match)
        total += x * y
    return total

with open("input.txt", "r") as f:
    corrupted_memory = f.read().strip().split('\n')

total_sum = 0
for line in corrupted_memory:
    total_sum += parse_mul_instruction(line)

print(total_sum)


"""--- Part Two ---"""

import re

def parse_instructions(line):
    mul_instructions = re.finditer(r'mul\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)', line)
    do_instruction = [(m.start(), 0) for m in re.finditer(r'do\(\)', line)]
    dont_instruction = [(m.start(), 1) for m in re.finditer(r"don't\(\)", line)]
    control_instructions = sorted(do_instruction + dont_instruction)
    return mul_instructions, control_instructions

def process_instructions(lines):
    enabled = True
    total_sum = 0
    control_stack = []
    
    for line in lines:
        mul_instructions, control_instructions = parse_instructions(line)
        
        control_stack.extend(control_instructions)
        
        for mul in mul_instructions:
            pos = mul.start()
            while control_stack and control_stack[0][0] < pos:
                _, state = control_stack.pop(0)
                enabled = (state == 0)  # 0 for do(), 1 for don't()
            
            if enabled:
                x, y = map(int, mul.groups())
                total_sum += x * y
    
    return total_sum

with open("input.txt", "r") as f:
    corrupted_memory = f.read().strip().split('\n')

result = process_instructions(corrupted_memory)
print(result)