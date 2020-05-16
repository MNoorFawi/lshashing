from bisect import insort

## to store nearest neighbors
class DistHeap:
    def __init__(self):
        self._container = []

    def push(self, item):
        insort(self._container, item)  # in by sort

    def pushitems(self, items):
        for i in items:
            self.push(i)

    def __getitem__(self, item):
        return self._container[item]

    def __len__(self):
        return len(self._container)

    def __repr__(self):
        return repr(self._container)

