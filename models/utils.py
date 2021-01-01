import itertools


def grouper(n, iterable, squash=None):
    it = iter(iterable)
    while True:
        if squash:
            chunk = [
                [None if (j != 0 and i in squash) else el[i] for i in range(len(el))]
                for j, el in enumerate(itertools.islice(it, n))
            ]
        else:
            chunk = list(itertools.islice(it, n))

        if not chunk:
            return
        elif len(chunk) != n:
            chunk += [None] * (n - len(chunk))
        yield chunk
