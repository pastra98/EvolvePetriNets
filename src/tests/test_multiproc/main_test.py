from func import expensive_func
from multiprocessing import Pool

if __name__ == "main":
    with Pool as p:
        res = p.map(expensive_func, [1,2,3,4,5,6,7,8])
