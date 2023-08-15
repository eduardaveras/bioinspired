import cProfile
from evolution import Evolution

pr = cProfile.Profile()
pr.enable()

# Call your run method
evolution_instance = Evolution()
evolution_instance.run()

pr.disable()
pr.print_stats(sort='cumtime')