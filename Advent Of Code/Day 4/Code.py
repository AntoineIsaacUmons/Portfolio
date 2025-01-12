def read_input(file_name):
    with open(file_name, 'r') as file:
        return [line.strip() for line in file.readlines()]

def count_xmas(grid):
    rows = len(grid)
    cols = len(grid[0])
    word = "XMAS"
    word_len = len(word)
    count = 0

    # Directions possibles (horizontal, vertical, diagonales)
    directions = [
        (0, 1),  # Droite
        (1, 0),  # Bas
        (1, 1),  # Diagonale bas-droite
        (1, -1), # Diagonale bas-gauche
        (0, -1), # Gauche
        (-1, 0), # Haut
        (-1, -1),# Diagonale haut-gauche
        (-1, 1)  # Diagonale haut-droite
    ]

    def check_direction(r, c, dr, dc):
        """Vérifie si 'XMAS' existe à partir de (r, c) dans une direction donnée."""
        for i in range(word_len):
            nr, nc = r + i * dr, c + i * dc
            if not (0 <= nr < rows and 0 <= nc < cols) or grid[nr][nc] != word[i]:
                return False
        return True

    for r in range(rows):
        for c in range(cols):
            for dr, dc in directions:
                if check_direction(r, c, dr, dc):
                    count += 1

    return count


if __name__ == "__main__":
    grid = read_input("input.txt")
    
    part1_result = count_xmas(grid)
    print("Partie 1: Nombre d'occurrences de 'XMAS':", part1_result)
