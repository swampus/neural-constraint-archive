DATA_MIN = 10.0
DATA_MAX = 100.0

def normalize(x):
    return (x - DATA_MIN) / (DATA_MAX - DATA_MIN)

def denormalize(x):
    return x * (DATA_MAX - DATA_MIN) + DATA_MIN