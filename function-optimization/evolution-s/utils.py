import time

def calculate_time(func):

    def wrapper(*args, **kwargs):
        # storing time before function execution
        begin = time.time()
        func(*args, **kwargs)
        # storing time after function execution
        end = time.time()
        print("Total time:" , func.__name__, end - begin)

    return time