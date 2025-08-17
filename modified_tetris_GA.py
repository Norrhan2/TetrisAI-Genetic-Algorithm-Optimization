from tetris_base import *
import random
import copy
import matplotlib.pyplot as plt


# Define parameters for the genetic algorithm
POPULATION_SIZE = 15
NUM_GENERATIONS = 5
TOURNAMENT_SIZE = 3
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
ELITISM_COUNT = 5
MAX_PIECES = 500  # Limit to 500 pieces per game

# Features to consider in evaluation
# 1. Aggregate height: Sum of heights of all columns
# 2. Complete lines: Number of complete lines
# 3. Holes: Count of empty cells with filled cells above them
# 4. Bumpiness: Sum of differences in height between adjacent columns
# 5. Piece contact: How many sides of piece are in contact with others
# 6. Wall contact: Contact with walls
# 7. Floor contact: Contact with floor
# 8. Blocking blocks: Blocks above holes


def evaluator(weights):
    # Set up the game
    board = get_blank_board()
    score = 0
    level = 1
    piece = get_new_piece()
    next_piece = get_new_piece()
    game_over = False
    pieces_played = 0  # Track number of pieces instead of moves
    
    # Play game until it's over or max pieces reached
    while not game_over and pieces_played < MAX_PIECES:
        # Get current state information
        total_holes, total_blocking_blocks = calc_initial_move_info(board)
        
        # Find best move for current piece
        best_move = find_best_move(board, piece, total_holes, total_blocking_blocks, weights)
        
        if best_move is None:
            # No valid moves, game over
            game_over = True
            continue
        
        # Apply the best move
        x, r = best_move
        piece['rotation'] = r
        piece['x'] = x
        
        # Drop the piece
        while is_valid_position(board, piece, adj_Y=1):
            piece['y'] += 1
            
        # Add piece to board
        add_to_board(board, piece)
        pieces_played += 1  # Increment pieces counter
        
        # Check for completed lines
        num_removed_lines = remove_complete_lines(board)
        
        # Update score based on lines cleared
        if num_removed_lines == 1:
            score += 40 * level
        elif num_removed_lines == 2:
            score += 100 * level
        elif num_removed_lines == 3:
            score += 300 * level
        elif num_removed_lines == 4:
            score += 1200 * level
            
        # Update level
        level, _ = calc_level_and_fall_freq(score)
        
        # Get next piece
        piece = next_piece
        next_piece = get_new_piece()
        
        # Check if new piece can be placed
        if not is_valid_position(board, piece):
            game_over = True
    
    # Return tuple with score and pieces played for logging
    return score, pieces_played


def find_best_move(board, piece, total_holes, total_blocking_blocks, weights):
    """
    Find the best move (rotation and x position) for the current piece
    Returns (x, rotation) or None if no valid move
    """
    best_score = float('-inf')
    best_move = None
    
    # Try all possible rotations and x positions
    for r in range(len(PIECES[piece['shape']])):
        for x in range(-2, BOARDWIDTH - 2):
            # Deep copy the piece to avoid modifying the original
            test_piece = copy.deepcopy(piece)
            
            # Calculate move info (returns metrics about this move)
            move_info = calc_move_info(board, test_piece, x, r, total_holes, total_blocking_blocks)
            
            if move_info[0]:  # If move is valid
                # Calculate score using weights
                # [valid, max_height, lines_cleared, new_holes, new_blocking_blocks,
                # piece_sides, floor_sides, wall_sides]
                move_score = (
                    weights[0] * move_info[1] +      # Max height (negative impact)
                    weights[1] * move_info[2] +      # Lines cleared (positive impact)
                    weights[2] * move_info[3] +      # New holes (negative impact)
                    weights[3] * move_info[4] +      # New blocking blocks (negative impact)
                    weights[4] * move_info[5] +      # Piece sides contact (positive impact)
                    weights[5] * move_info[6] +      # Floor sides contact (positive impact)
                    weights[6] * move_info[7]        # Wall sides contact (can be positive)
                )
                
                if move_score > best_score:
                    best_score = move_score
                    best_move = (x, r)
    
    return best_move


def Genetic_Algorithm():
    """
    Implements the genetic algorithm to evolve weights for Tetris evaluation
    """
    # Initialize population with random weights
    population = []
    best_fitness_history = []
    avg_fitness_history = []
    best_chromosomes_log = []  # Log for best chromosomes in each generation
    
    print("Initializing population...")
    for _ in range(POPULATION_SIZE):
        # Generate random weights
        # Weight ranges:
        # Height: negative (-1 to 0)
        # Lines: positive (0 to 1)
        # Holes: negative (-1 to 0)
        # Blocking: negative (-1 to 0)
        # Piece sides: positive (0 to 1)
        # Floor sides: positive (0 to 1)
        # Wall sides: mixed (-0.5 to 0.5)
        weights = [
            -random.uniform(0.1, 1),     # Height (negative)
            random.uniform(0.1, 1),      # Lines cleared (positive)
            -random.uniform(0.1, 1),     # Holes (negative)
            -random.uniform(0.1, 1),     # Blocking blocks (negative)
            random.uniform(0.1, 1),      # Piece sides (positive)
            random.uniform(0.1, 1),      # Floor sides (positive)
            random.uniform(-0.5, 0.5),   # Wall sides (mixed)
        ]
        population.append(weights)
    
    # Evolution loop
    for generation in range(NUM_GENERATIONS):
        print(f"Generation {generation+1}/{NUM_GENERATIONS}")
        
        # Evaluate fitness for each individual
        fitness_scores = []
        pieces_played_list = []
        for individual in population:
            # Run the game using these weights
            score, pieces_played = evaluator(individual)
            fitness_scores.append(score)
            pieces_played_list.append(pieces_played)
            print(f"Individual fitness: {score}, Pieces: {pieces_played}, Weights: {[round(w, 2) for w in individual]}")
        
        # Track statistics
        best_fitness = max(fitness_scores)
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        
        # Get best chromosomes
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
        best_idx = sorted_indices[0]
        second_best_idx = sorted_indices[1]
        
        # Log the best chromosomes for this generation
        gen_log = {
            "generation": generation + 1,
            "best": {
                "weights": [round(w, 4) for w in population[best_idx]],
                "score": fitness_scores[best_idx],
                "pieces_played": pieces_played_list[best_idx]
            },
            "second_best": {
                "weights": [round(w, 4) for w in population[second_best_idx]],
                "score": fitness_scores[second_best_idx],
                "pieces_played": pieces_played_list[second_best_idx]
            }
        }
        best_chromosomes_log.append(gen_log)
        
        print(f"Generation {generation+1}: Best fitness = {best_fitness}")
        print(f"Best weights: {[round(w, 4) for w in population[best_idx]]}")
        
        # Create new population
        new_population = []
        
        # Elitism: keep the best individuals
        for i in range(ELITISM_COUNT):
            new_population.append(population[sorted_indices[i]])
        
        # Fill the rest of the population with offspring
        while len(new_population) < POPULATION_SIZE:
            # Tournament selection
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)
            
            # Crossover
            if random.random() < CROSSOVER_RATE:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            mutate(child1)
            mutate(child2)
            
            new_population.append(child1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)
        
        # Replace old population
        population = new_population
    
    # Find the best individual in final population
    final_fitness_scores = []
    final_pieces_played = []
    for individual in population:
        score, pieces = evaluator(individual)
        final_fitness_scores.append(score)
        final_pieces_played.append(pieces)
    
    best_idx = final_fitness_scores.index(max(final_fitness_scores))
    best_weights = population[best_idx]
    best_score = final_fitness_scores[best_idx]
    best_pieces = final_pieces_played[best_idx]
    
    print("\nGenetic Algorithm completed!")
    print(f"Best weights found: {[round(w, 4) for w in best_weights]}")
    print(f"Best fitness score: {best_score} with {best_pieces} pieces played")
    
    # Plot fitness over generations
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, NUM_GENERATIONS + 1), best_fitness_history, label='Best Fitness')
    plt.plot(range(1, NUM_GENERATIONS + 1), avg_fitness_history, label='Avg Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.title('Fitness Evolution in Genetic Algorithm')
    plt.legend()
    plt.grid(True)
    plt.savefig('fitness_evolution.png')
    plt.show()
    
    # Save best weights to file
    with open('best_weights.txt', 'w') as f:
        f.write(f"# Best weights found by GA for Tetris\n")
        f.write(f"# Score: {best_score}, Pieces: {best_pieces}\n")
        for i, w in enumerate(best_weights):
            f.write(f"{w}\n")
    
    # Save best chromosomes log to file
    with open('best_chromosomes_log.txt', 'w') as f:
        f.write("# Best Chromosomes Log\n")
        f.write("# Format: Generation, Best Score, Best Pieces, [Best Weights],"
                " Second Best Score, Second Best Pieces, [Second Best Weights]\n\n")
        for gen_data in best_chromosomes_log:
            f.write(f"Generation {gen_data['generation']}: \n")
            f.write(f"  Best: Score={gen_data['best']['score']}, Pieces={gen_data['best']['pieces_played']}\n")
            f.write(f"  Weights={gen_data['best']['weights']}\n")
            f.write(f"  Second: Score={gen_data['second_best']['score']},"
                    f" Pieces={gen_data['second_best']['pieces_played']}\n")
            f.write(f"  Weights={gen_data['second_best']['weights']}\n\n")
    
    return best_weights


def tournament_selection(population, fitness_scores):
    """
    Tournament selection - randomly select individuals and pick the best one
    """
    indices = random.sample(range(len(population)), TOURNAMENT_SIZE)
    best_idx = indices[0]
    
    for idx in indices:
        if fitness_scores[idx] > fitness_scores[best_idx]:
            best_idx = idx
    
    return population[best_idx].copy()


def crossover(parent1, parent2):
    """
    Perform crossover between two parents to create two children
    Using uniform crossover
    """
    child1 = []
    child2 = []
    
    for i in range(len(parent1)):
        # With 50% probability, swap the genes
        if random.random() < 0.5:
            child1.append(parent1[i])
            child2.append(parent2[i])
        else:
            child1.append(parent2[i])
            child2.append(parent1[i])
    
    return child1, child2


def mutate(individual):
    """
    Mutate an individual by randomly changing some of its weights
    """
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            # Apply mutation based on weight type
            if i == 0 or i == 2 or i == 3:  # Negative weights
                individual[i] = -random.uniform(0.1, 1)
            elif i == 1 or i == 4 or i == 5:  # Positive weights
                individual[i] = random.uniform(0.1, 1)
            else:  # Mixed weights (wall sides)
                individual[i] = random.uniform(-0.5, 0.5)


def play_game_with_weights(weights, max_pieces=MAX_PIECES):
    """
    Play a game using the specified weights
    """
    # Make Tetris game manual to visualize
    global MANUAL_GAME
    old_manual_setting = MANUAL_GAME
    MANUAL_GAME = True
    
    # Setup game
    board = get_blank_board()
    score = 0
    level = 1
    piece = get_new_piece()
    next_piece = get_new_piece()
    pieces_played = 0
    
    # Setup pygame display
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    DISPLAYSURF = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
    BASICFONT = pygame.font.Font('freesansbold.ttf', 18)
    # BIGFONT = pygame.font.Font('freesansbold.ttf', 100)
    pygame.display.set_caption('Tetris AI')
    
    # Game variables
    last_fall_time = time.time()
    fall_freq = 0.0001  # Slow down for visualization
    game_over = False
    
    while not game_over and pieces_played < max_pieces:
        # Check for quit
        check_quit()
        for event in pygame.event.get():
            if event.type == KEYUP and event.key == K_ESCAPE:
                pygame.quit()
                return
        
        if piece is None:
            piece = next_piece
            next_piece = get_new_piece()
            last_fall_time = time.time()
            
            if not is_valid_position(board, piece):
                game_over = True
                continue
        
        # Get current state
        total_holes, total_blocking_blocks = calc_initial_move_info(board)
        
        # Find best move
        best_move = find_best_move(board, piece, total_holes, total_blocking_blocks, weights)
        
        if best_move is None:
            game_over = True
            continue
        
        # Apply best move
        x, r = best_move
        piece['rotation'] = r
        piece['x'] = x
        
        # Drop the piece
        if time.time() - last_fall_time > fall_freq:
            if is_valid_position(board, piece, adj_Y=1):
                piece['y'] += 1
                last_fall_time = time.time()
            else:
                # Piece has landed
                add_to_board(board, piece)
                pieces_played += 1  # Increment pieces counter
                
                num_removed_lines = remove_complete_lines(board)
                
                # Update score
                if num_removed_lines == 1:
                    score += 40 * level
                elif num_removed_lines == 2:
                    score += 100 * level
                elif num_removed_lines == 3:
                    score += 300 * level
                elif num_removed_lines == 4:
                    score += 1200 * level
                
                # Reset piece
                piece = None
        
        # Display game state
        DISPLAYSURF.fill(BGCOLOR)
        draw_board(board)
        draw_status(score, level)
        
        # Display pieces played
        pieces_surf = BASICFONT.render(f'Pieces: {pieces_played}/{max_pieces}', True, TEXTCOLOR)
        pieces_rect = pieces_surf.get_rect()
        pieces_rect.topleft = (WINDOWWIDTH - 150, 180)
        DISPLAYSURF.blit(pieces_surf, pieces_rect)
        
        if next_piece:
            draw_next_piece(next_piece)
        
        if piece:
            draw_piece(piece)
        
        pygame.display.update()
        FPSCLOCK.tick(FPS)
    
    # Game over message
    show_text_screen('Game Over')
    
    # Restore manual game setting
    MANUAL_GAME = old_manual_setting
    
    return score, pieces_played


# Run the genetic algorithm to find optimal weights
if __name__ == "__main__":
    # best_weights = Genetic_Algorithm()
    best_weights = [-0.1, 0.9, -0.99, -0.22, 0.61, 0.8, 0.36]
    # Play a game with the best weights found (with 600 pieces as requested)
    print("\nPlaying a test game with the best weights (600 pieces limit)...")
    final_score, final_pieces = play_game_with_weights(best_weights, max_pieces=600)
    print(f"Final game score: {final_score} with {final_pieces} pieces played")
