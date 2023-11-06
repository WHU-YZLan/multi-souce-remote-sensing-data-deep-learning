import random

random_numbers = []
for _ in range(6):
    number = random.randint(0, 9)
    random_numbers.append(number)

print(random_numbers)