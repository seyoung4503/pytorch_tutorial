class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._children = _children
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), _op='+')
    
    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), _op='*')
    
if __name__ == "__main__":
    a = Value(10)
    b = Value(20)
    c = a + b

    c = Value(5)
    e = a + b*c

    print(e)