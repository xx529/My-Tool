# https://github.com/jasondelaat/pymonad

class Failure():
    def __init__(self, value, failed=False):
        self.value = value
        self.failed = failed
    def get(self):
        return self.value
    def is_failed(self):
        return self.failed
    def __str__(self):
        return ' '.join([str(self.value), str(self.failed)])
    def __or__(self, f):
        if self.failed:
            return self
        try:
            x = f(self.get())
            return Failure(x)
        except:
            return Failure(None, True)


# This will work.
from operator import neg
x = '1'
y = Failure(x) | int | neg | str
print(y)
>>> -1 False

# This will not
from operator import neg
x = 'hahaha'
y = Failure(x) | int | neg | str
print(y)
>>> None True
