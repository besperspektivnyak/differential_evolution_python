from test_functions import func, func2
from rede import ReSHADE

test_case = ReSHADE(borders=[-100, 100], function=func, max_nfes=100000, population_size=100, problem_size=30)
# print(test_case.ReDE(0.8, 0.7))
print(test_case.ReSHADE())
