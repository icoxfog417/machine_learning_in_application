import numpy as np


class DataProcessor():

    def __init__(self, means=(), stds=()):
        self.means = means
        self.stds = stds
    
    def format_x(self, x, size=-1):
        _x = x
        if isinstance(x, (tuple, list)):
            _x = np.array([x])
        
        if size > 0 and _x.shape[1] != size:
            _x = self.adjust(x, size)

        _x = _x.astype(np.float32, copy=False)

        if len(self.means) > 0 and len(self.stds) > 0:
            return (_x - self.means) / self.stds
        else:
            return _x
    
    def adjust(self, x, size):
        def max_pooling(v):
            sqrt = lambda _x: int(np.ceil(np.sqrt(_x)))
            _target_square_size = sqrt(size)
            square_size = sqrt(len(v))
            conv_size = int(square_size // _target_square_size)
            image = np.reshape(v, (square_size, square_size))
            _pooled = []
            for i in range(size):
                row, col = int(i // _target_square_size * conv_size), int(i % _target_square_size * conv_size)
                mp = np.max(image[row:row + conv_size, col: col + conv_size])
                _pooled.append(mp)
            return np.array(_pooled)
        
        x = np.array([max_pooling(_v) for _v in x])
        return x

    def format_y(self, y):
        _y = y
        if isinstance(y , int):
            _y = np.array([y])
        _y = _y.astype(np.int32, copy=False)
        return _y 
    
    def set_normalization_params(self, x):
        self.means = np.mean(x, axis=0, dtype=np.float32)
        self.stds = np.std(x, axis=0, dtype=np.float32)
        # simple trick to avoid 0 divide
        self.stds[self.stds < 1.0e-6] = np.max(x) - np.min(x)
        self.means[self.stds < 1.0e-6] = np.min(x)

    def batch_iter(self, X, y, batch_size, epoch=1):
        indices = np.array(range(len(y)))
        appendix = batch_size - len(y) % batch_size
        for e in range(epoch):
            np.random.shuffle(indices)
            batch_indices = np.concatenate([indices, indices[:appendix]])
            batch_count = len(batch_indices) // batch_size
            for b in range(batch_count):
                elements = batch_indices[b * batch_size:(b + 1) * batch_size]
                x_batch = X[elements]
                y_batch = y[elements]
                epoch_end = True if b == batch_count - 1 else False
                yield x_batch, y_batch, epoch_end
