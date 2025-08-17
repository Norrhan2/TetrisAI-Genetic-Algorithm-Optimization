# Tetris AI – Genetic Algorithm Optimization

## 📌 Project Overview
This project implements an **AI agent to play Tetris** using a **Genetic Algorithm (GA)**.  
The agent evolves over multiple generations by optimizing a fitness function based on board features, aiming to improve decision-making and overall gameplay performance.

---

## 🚀 Features
- **Genetic Algorithm** implementation with crossover, mutation, and tournament selection.  
- **Custom Fitness Function** using board features (e.g., cleared lines, holes, height, bumpiness).  
- **Visualization** of the AI's evolution progress using Matplotlib.  
- Adjustable parameters: population size, mutation rate, number of generations.  
- Performance analysis of evolved agents.  

---

## 🛠️ Tech Stack
- **Programming Language:** Python  
- **Libraries:** NumPy, Matplotlib, random  

---

## ⚙️ How It Works
1. **Initialization:** Random population of candidate solutions (weights for board features).  
2. **Evaluation:** Each agent plays multiple Tetris games; fitness is computed.  
3. **Selection:** Tournament selection picks the best individuals.  
4. **Crossover & Mutation:** Generate a new population with diversity.  
5. **Evolution:** Repeat over generations until convergence.  

---

## 📊 Results
- Agents improved significantly over generations.  
- Visualization of average and best fitness scores across generations.  
- Demonstrated the effectiveness of evolutionary algorithms in solving game-playing tasks.  

