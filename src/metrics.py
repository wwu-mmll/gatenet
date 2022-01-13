from fastai.basics import accuracy, BalancedAccuracy, F1Score, SigmoidRange, Tensor


def get_metrics(n_classes: int, sigmoid: bool = True) -> dict:
    names = ('ACC', 'BACC', 'F1_weighted', 'F1_macro')
    if n_classes == 2:
        funcs = [ACC(sigmoid), BACC(sigmoid), F1_weighted(sigmoid), F1_macro(sigmoid)]
    else:
        funcs = [accuracy, BalancedAccuracy(), F1Score(average='weighted'), F1Score(average='macro')]
    return {name: func for name, func in zip(names, funcs)}


class BinaryMetric:
    def __init__(self, sigmoid: bool):
        self.sigmoid = sigmoid

    def prepare(self, preds, targs):
        if self.sigmoid:
            preds = self.apply_sigmoid(preds)
        return self.to_binary(preds), self.to_binary(targs)

    @staticmethod
    def apply_sigmoid(x: Tensor) -> Tensor:
        return SigmoidRange(0., 1.)(x)

    @staticmethod
    def to_binary(x: Tensor) -> Tensor:
        return (x > .5).int()


class ACC(BinaryMetric):
    def __init__(self, sigmoid):
        super(ACC, self).__init__(sigmoid)
        self.__name__ = 'ACC'

    def __call__(self, preds: Tensor, targs: Tensor) -> Tensor:
        preds, targs = self.prepare(preds, targs)
        preds = preds.view(-1)
        return (preds == targs).float().mean()


class BACC(BinaryMetric):
    def __init__(self, sigmoid):
        super(BACC, self).__init__(sigmoid)
        self.__name__ = 'BACC'

    def __call__(self, preds: Tensor, targs: Tensor) -> float:
        preds, targs = self.prepare(preds, targs)
        return BalancedAccuracy()(preds, targs)


class F1_weighted(BinaryMetric):
    def __init__(self, sigmoid):
        super(F1_weighted, self).__init__(sigmoid)
        self.__name__ = 'F1_weighted'

    def __call__(self, preds: Tensor, targs: Tensor) -> float:
        preds, targs = self.prepare(preds, targs)
        return F1Score(average='weighted')(preds, targs)


class F1_macro(BinaryMetric):
    def __init__(self, sigmoid):
        super(F1_macro, self).__init__(sigmoid)
        self.__name__ = 'F1_macro'

    def __call__(self, preds: Tensor, targs: Tensor) -> float:
        preds, targs = self.prepare(preds, targs)
        return F1Score(average='macro')(preds, targs)
