import math

def calculate_a(d, beta):
    # Ensure beta is in radians
    a =  d * math.sin(math.radians(beta))*2/3
    return a

# Example usage
print(math.sin(math.radians(5)))
