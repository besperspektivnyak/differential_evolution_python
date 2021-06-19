def func(x):
    res = 0
    for x_ in x:
        res += x_ ** 2
    return res


def func2(x):
    sum = 0
    p = 0
    for i in range(len(x)):
        sum += math.fabs(x[i])
        p *= math.fabs(x[i])
    res = sum + p
    return res