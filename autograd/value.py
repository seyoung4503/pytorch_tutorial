class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), _op='+')
    
    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), _op='*')
    
    def tanh(self):
        import math
        t = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        return Value(t, (self,), _op='tanh')
    
    
if __name__ == "__main__":
    a = Value(10)
    b = Value(20)
    c = a + b

    c = Value(5)
    e = a + b*c

    print(e)

    import visualize as viz
    dot = viz.draw_dot(e)

    dot.render('graphtree', view=True)