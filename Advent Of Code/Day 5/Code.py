def parse_input(input_data):
    rules_section, updates_section = input_data.strip().split("\n\n")
    rules = []
    for rule in rules_section.splitlines():
        x, y = map(int, rule.split('|'))
        rules.append((x, y))
    updates = [list(map(int, update.split(','))) for update in updates_section.splitlines()]
    return rules, updates

def is_update_valid(update, rules):
    positions = {page: idx for idx, page in enumerate(update)}
    for x, y in rules:
        if x in positions and y in positions:
            if positions[x] > positions[y]: 
                return False
    return True

def get_middle_number(update):
    length = len(update)
    middle_index = length // 2
    if length % 2 == 0:
        return update[middle_index - 1]
    else:
        return update[middle_index]

def order_update(update, rules):
    # Tri topologique simple
    graph = {page: set() for page in update}
    for x, y in rules:
        if x in graph and y in graph:
            graph[x].add(y)
    
    ordered = []
    while graph:
        for node in list(graph.keys()):
            if not graph[node]:
                ordered.append(node)
                del graph[node]
                for other_node in graph:
                    graph[other_node].discard(node)
                break
        else:
            return sorted(update)
    return ordered

def solve_puzzle_part_one(input_data):
    rules, updates = parse_input(input_data)
    print("Rules:", rules)  
    print("Updates:", updates) 
    
    valid_updates = [update for update in updates if is_update_valid(update, rules)]
    print("Valid updates:", valid_updates) 

    middle_numbers = [get_middle_number(update) for update in valid_updates]
    print("Middle numbers:", middle_numbers) 

    return sum(middle_numbers)

def solve_puzzle_part_two(input_data):
    rules, updates = parse_input(input_data)
    invalid_updates = [update for update in updates if not is_update_valid(update, rules)]
    print("Invalid updates:", invalid_updates)
    
    correctly_ordered_invalid = [order_update(update, rules) for update in invalid_updates]
    print("Correctly ordered invalid updates:", correctly_ordered_invalid)
    
    middle_numbers = [get_middle_number(update) for update in correctly_ordered_invalid]
    print("Middle numbers of ordered invalid updates:", middle_numbers)

    return sum(middle_numbers)

with open('input.txt', 'r') as file:
    input_data = file.read()

result_part_one = solve_puzzle_part_one(input_data)
print("Sum of middle page numbers for correctly ordered updates:", result_part_one)

print("""Part Two""")

result_part_two = solve_puzzle_part_two(input_data)
print("Sum of middle page numbers of incorrectly ordered updates after ordering them:", result_part_two)