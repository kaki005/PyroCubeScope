from pyro.import PyroModule

class Regime(PyroModule):
    def A(self):
        pass
    def B(self):
        pass

    def __init__():
        super().__init__()

    def forward(self, input_):
        return self.bias + input_ @ self.weight
