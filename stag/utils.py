class EarlyStopping(object):
    best_loss = None
    counter = 0

    def __init__(self, patience=10):
        self.patience = patience

    def __call__(self, loss):
        print("current loss: ", loss, "best loss: ", self.best_loss, "counter: ", self.counter)
        if self.best_loss is None:
            self.best_loss = loss
            self.counter = 0
        elif loss >= self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                print("exceeding patience ", "counter: ", self.counter, "patience: ", self.patience)
                return True
        else:
            self.best_loss = loss
            self.counter = 0
