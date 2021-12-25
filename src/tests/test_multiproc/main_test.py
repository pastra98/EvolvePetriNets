from func import expensive_func
from multiprocessing import Pool

def main():
    with Pool() as p:
        res = p.map(expensive_func, [1,2,3,4,5,6,7,8])
    print(res)


if __name__ == "__main__":
    main()
else:
    raise Exception("woopsie, no run as main")
