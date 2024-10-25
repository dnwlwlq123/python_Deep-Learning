from collections import defaultdict
from typing import Any, Callable, List, Tuple

def select_by_coverage(
        hist: dict[Any, int], 
        coverage: float = 0.999, 
        key: Callable = lambda x: x[1],
        reverse: bool = True, 
    ) -> List[Tuple[Any, int]]:
    
    lst: List[Tuple[Any, int]] = list(hist.items())

    lst = sorted(lst, key = key, reverse = True)
    total = sum([e[1] for e in lst])
    s = 0

    for idx, (elem, freq) in enumerate(lst): 
        # s += freq 
        if s > total * coverage:
            break 
        s += freq 
    
    return lst[:idx]

def generate_histogram(
        lst: List[Any], 
        key: Callable, 
        default: Callable = int, 
    ) -> defaultdict[Any, int]:

    res: defaultdict[Any, int] = defaultdict(default)

    for elem in lst:
        res[key(elem)] += 1 

    return res 

if __name__ == '__main__':
    import os
    from config import DATA_DIR

    # testing select_by_coverage 
    hist = {
        'a' : 10, 
        'b' : 5, 
        'c' : 1, 
        'd' : 1, 
    }
    print(select_by_coverage(hist, coverage = 0.8))

    korea_tokens = []
    with open(os.path.join(DATA_DIR, 'kor.txt'), 'r',
        encoding = 'utf-8') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            kor = line[0]

            for tok in kor.split():
                korea_tokens.append(tok)
    hist: defaultdict[str, int] = generate_histogram(korea_tokens, key = str)

    lst: List[Tuple[str, int]] = select_by_coverage(hist, coverage = 0.99)

    for k, v in lst:
        print(k, v)
    print(len(lst))

    length_hist: defaultdict[str, int] = generate_histogram(korea_tokens, key = len)

    lst: List[Tuple[int, int]] = select_by_coverage(length_hist, coverage = 0.99)

    for k, v in lst:
        print(k, v)
    print(len(lst))


    lst: List[Tuple[int, int]] = select_by_coverage(length_hist, coverage=0.99, key = lambda x:x[0],
                                                    revers = True)

    for k, v in lst:
        print(k, v)
    print(len(lst))

