import random
import numpy as np
import time

# Define the data values and requirements
activities = [
    {"name": "SLA100A", "enrollment": 50, "preferred_facilitators": ["Glen", "Lock", "Banks", "Zeldin"],
     "other_facilitators": ["Numen", "Richards"]},
    {"name": "SLA100B", "enrollment": 50, "preferred_facilitators": ["Glen", "Lock", "Banks", "Zeldin"],
     "other_facilitators": ["Numen", "Richards"]},
    {"name": "SLA191A", "enrollment": 50, "preferred_facilitators": ["Glen", "Lock", "Banks", "Zeldin"],
     "other_facilitators": ["Numen", "Richards"]},
    {"name": "SLA191B", "enrollment": 50, "preferred_facilitators": ["Glen", "Lock", "Banks", "Zeldin"],
     "other_facilitators": ["Numen", "Richards"]},
    {"name": "SLA201", "enrollment": 50, "preferred_facilitators": ["Glen", "Banks", "Zeldin", "Shaw"],
     "other_facilitators": ["Numen", "Richards", "Singer"]},
    {"name": "SLA291", "enrollment": 50, "preferred_facilitators": ["Lock", "Banks", "Zeldin", "Singer"],
     "other_facilitators": ["Numen", "Richards", "Shaw", "Tyler"]},
    {"name": "SLA303", "enrollment": 60, "preferred_facilitators": ["Glen", "Zeldin", "Banks"],
     "other_facilitators": ["Numen", "Singer", "Shaw"]},
    {"name": "SLA304", "enrollment": 25, "preferred_facilitators": ["Glen", "Banks", "Tyler"],
     "other_facilitators": ["Numen", "Singer", "Shaw", "Richards", "Uther", "Zeldin"]},
    {"name": "SLA394", "enrollment": 20, "preferred_facilitators": ["Tyler", "Singer"],
     "other_facilitators": ["Richards", "Zeldin"]},
    {"name": "SLA449", "enrollment": 60, "preferred_facilitators": ["Tyler", "Singer", "Shaw"],
     "other_facilitators": ["Zeldin", "Uther"]},
    {"name": "SLA451", "enrollment": 100, "preferred_facilitators": ["Tyler", "Singer", "Shaw"],
     "other_facilitators": ["Zeldin", "Uther", "Richards", "Banks"]},
]

facilitators = ["Lock", "Glen", "Banks", "Richards", "Shaw", "Singer", "Uther", "Tyler", "Numen", "Zeldin"]

rooms = [
    {"name": "Slater 003", "capacity": 45},
    {"name": "Roman 216", "capacity": 30},
    {"name": "Loft 206", "capacity": 75},
    {"name": "Roman 201", "capacity": 50},
    {"name": "Loft 310", "capacity": 108},
    {"name": "Beach 201", "capacity": 60},
    {"name": "Beach 301", "capacity": 75},
    {"name": "Logos 325", "capacity": 450},
    {"name": "Frank 119", "capacity": 60},
]

timeslots = ["10 AM", "11 AM", "12 PM", "1 PM", "2 PM", "3 PM"]


# Define the GeneticAlgorithm class
class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            schedule = self.generate_random_schedule()
            population.append(schedule)
        return population

    def generate_random_schedule(self):
        schedule = []
        for activity in activities:
            room = random.choice(rooms)
            time = random.choice(timeslots)
            preferred_facilitator = random.choice(activity["preferred_facilitators"])
            schedule.append({"activity": activity["name"], "time": time, "room": room["name"],
                             "facilitator": preferred_facilitator})
        return schedule

    def evaluate_fitness(self, schedule):
        fitness = 0
        facilitator_load = {facilitator: 0 for facilitator in facilitators}

        for i, activity1 in enumerate(schedule):
            for j, activity2 in enumerate(schedule):
                if i != j and activity1["time"] == activity2["time"] and activity1["room"] == activity2["room"]:
                    fitness -= 0.5

        for activity in schedule:
            expected_enrollment = next((a["enrollment"] for a in activities if a["name"] == activity["activity"]), None)
            room_capacity = next((room["capacity"] for room in rooms if room["name"] == activity["room"]), None)
            if room_capacity is not None:
                if room_capacity < expected_enrollment:
                    fitness -= 0.5
                elif room_capacity > 6 * expected_enrollment:
                    fitness -= 0.4
                elif room_capacity > 3 * expected_enrollment:
                    fitness -= 0.2
                else:
                    fitness += 0.3

            preferred_facilitators = next(
                (a["preferred_facilitators"] for a in activities if a["name"] == activity["activity"]), [])
            other_facilitators = next(
                (a["other_facilitators"] for a in activities if a["name"] == activity["activity"]), [])
            if activity["facilitator"] in preferred_facilitators:
                fitness += 0.5
            elif activity["facilitator"] in other_facilitators:
                fitness += 0.2
            else:
                fitness -= 0.1

            facilitator_load[activity["facilitator"]] += 1

        for facilitator, load in facilitator_load.items():
            if facilitator == "Dr. Tyler" and load < 2:
                continue
            if load > 4:
                fitness -= 0.5
            elif load > 2:
                fitness -= 0.4

        for i in range(len(schedule) - 1):
            current_time = timeslots.index(schedule[i]["time"])
            next_time = timeslots.index(schedule[i + 1]["time"])
            if next_time == current_time + 1:
                # Implement rules for consecutive time slots
                pass

        # Activity-specific adjustments
        for i in range(len(schedule) - 1):
            activity1 = schedule[i]
            activity2 = schedule[i + 1]

            if (activity1["activity"] == "SLA101A" and activity2["activity"] == "SLA101B") or \
                    (activity1["activity"] == "SLA101B" and activity2["activity"] == "SLA101A"):
                if abs(timeslots.index(activity1["time"]) - timeslots.index(activity2["time"])) > 4:
                    fitness += 0.5
                if activity1["time"] == activity2["time"]:
                    fitness -= 0.5

            if (activity1["activity"] == "SLA191A" and activity2["activity"] == "SLA191B") or \
                    (activity1["activity"] == "SLA191B" and activity2["activity"] == "SLA191A"):
                if abs(timeslots.index(activity1["time"]) - timeslots.index(activity2["time"])) > 4:
                    fitness += 0.5
                if activity1["time"] == activity2["time"]:
                    fitness -= 0.5

            if (activity1["activity"].startswith("SLA101") and activity2["activity"].startswith("SLA191")) or \
                    (activity1["activity"].startswith("SLA191") and activity2["activity"].startswith("SLA101")):
                if abs(timeslots.index(activity1["time"]) - timeslots.index(activity2["time"])) == 1:
                    fitness += 0.25
                elif activity1["time"] == activity2["time"]:
                    fitness -= 0.25

            if (activity1["activity"].startswith("SLA101") and activity2["activity"].startswith("SLA191")) or \
                    (activity1["activity"].startswith("SLA191") and activity2["activity"].startswith("SLA101")):
                if abs(timeslots.index(activity1["time"]) - timeslots.index(activity2["time"])) == 1:
                    if activity1["room"].startswith("Roman") or activity1["room"].startswith("Beach") or \
                            activity2["room"].startswith("Roman") or activity2["room"].startswith("Beach"):
                        fitness -= 0.4

        return fitness

    def softmax(self, fitness_scores):
        exp_scores = np.exp(fitness_scores)
        probs = exp_scores / np.sum(exp_scores)
        return probs

    def select_parents(self):
        parent_indices = np.random.choice(len(self.population), size=2, replace=False)
        return self.population[parent_indices[0]], self.population[parent_indices[1]]

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(parent1) - 1)
        offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
        offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
        return offspring1, offspring2

    def mutate(self, schedule):
        mutated_schedule = schedule[:]
        for i in range(len(mutated_schedule)):
            if random.random() < self.mutation_rate:
                mutated_schedule[i]["room"] = random.choice(rooms)["name"]
                mutated_schedule[i]["time"] = random.choice(timeslots)
                mutated_schedule[i]["facilitator"] = random.choice(facilitators)
        return mutated_schedule

    def evolve(self):
        new_population = []
        fitness_scores = [self.evaluate_fitness(schedule) for schedule in self.population]
        # print("Fitness scores:", fitness_scores)
        probabilities = self.softmax(fitness_scores)
        # print("Probabilities:", probabilities)

        for _ in range(self.population_size // 2):
            # print("Creating offspring...")
            parent1, parent2 = self.select_parents()
            # print("Selected parents:", parent1, parent2)
            offspring1, offspring2 = self.crossover(parent1, parent2)
            # print("Offspring after crossover:", offspring1, offspring2)
            offspring1 = self.mutate(offspring1)
            offspring2 = self.mutate(offspring2)
            new_population.extend([offspring1, offspring2])
            # print("New population size:", len(new_population))

        self.population = new_population
        # print("Population after evolution:", self.population)

    def optimize(self, max_generations):
        improvement_threshold = 0.01
        last_avg_fitness = float('inf')
        generations_without_improvement = 0

        for generation in range(max_generations):
            self.evolve()
            fitness_scores = [self.evaluate_fitness(schedule) for schedule in self.population]
            avg_fitness = np.mean(fitness_scores)

            if last_avg_fitness != 0 and last_avg_fitness != float('inf'):
                improvement_ratio = (last_avg_fitness - avg_fitness) / last_avg_fitness
                if improvement_ratio < improvement_threshold:
                    generations_without_improvement += 1
                else:
                    generations_without_improvement = 0
            else:
                generations_without_improvement = 0

            if generations_without_improvement >= 100:
                break

            last_avg_fitness = avg_fitness

        best_fitness_index = np.argmax(fitness_scores)
        best_fitness = fitness_scores[best_fitness_index]
        best_schedule = self.population[best_fitness_index]

        return best_fitness, best_schedule


# Population
population_size = 500
# Mutation Rate
mutation_rate = 0.01
# Maximum number of generations
max_generations = 100

# Timer
start_time = time.time()

# Start the genetic algorithm
ga = GeneticAlgorithm(population_size, mutation_rate)
best_fitness, best_schedule = ga.optimize(max_generations)

# End Timer + calculate total time
end_time = time.time()
total_time = end_time - start_time

print("Best Fitness:", best_fitness)
print("Best Schedule:", best_schedule)
print("Time it took to generate: ", total_time, " seconds")
