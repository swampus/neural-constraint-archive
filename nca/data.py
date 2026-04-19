import random

def generate_sample():
    return [random.randint(10, 100) for _ in range(3)]

def generate_dataset(n=5000):
    return [generate_sample() for _ in range(n)]