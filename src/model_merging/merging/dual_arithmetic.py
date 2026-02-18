
def svd_orthogonalize(M):
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    return U @ Vh

class LinearSVD(Atom):
    def __init__(self, fanout, fanin):
        super().__init__()
        self.fanin = fanin
        self.fanout = fanout
        self.smooth = True
        self.mass = 0.5
        self.sensitivity = 1

    def forward(self, x, w):
        weights = w[0]  # [fanout, fanin]
        return x @ weights.T

    def initialize(self, key=None):
        print("No need init")
        return None

    def project(self, w):
        weight = w[0]
        weight = svd_orthogonalize(weight) * sqrt(self.fanout / self.fanin)
        return [weight]

    def dualize(self, grad_w, target_norm=1.0):
        grad = grad_w[0]
        
        # 1. Calculate the scalar factor
        scalar_factor = sqrt(self.fanout / self.fanin) * target_norm
        
        # 2. Compute the dual weight
        # svd_orthogonalize returns the "direction", scalar_factor applies the "magnitude"
        d_weight = svd_orthogonalize(grad) * scalar_factor
        
    
        
        return [d_weight]


class Conv2DSVD(Atom):
    def __init__(self, fanout, fanin, kernel_size):
        super().__init__()
        self.fanin = fanin
        self.fanout = fanout
        self.kernel_size = kernel_size
        self.smooth = True
        self.mass = 0.5
        self.sensitivity = 1

    def forward(self, x, w):
        weights = w[0]  # [fanout, fanin, k, k]
        return torch.nn.functional.conv2d(x, weights, padding='same')

    def initialize(self, key=None):
        weight = torch.randn(self.fanout, self.fanin, self.kernel_size, self.kernel_size)
        weight = self._ortho_spatial(weight)
        scale = (1.0 / self.kernel_size ** 2) * sqrt(self.fanout / self.fanin)
        return [weight * scale]

    def project(self, w):
        weight = w[0]
        weight = self._ortho_spatial(weight)
        scale = (1.0 / self.kernel_size ** 2) * sqrt(self.fanout / self.fanin)
        return [weight * scale]

    def dualize(self, grad_w, target_norm=1.0):
        grad = grad_w[0]
        
        # 1. Calculate the scalar factor
        # The paper defines this specifically for Conv2D to normalize spatial dimensions
        scalar_factor = (1.0 / self.kernel_size ** 2) * sqrt(self.fanout / self.fanin) * target_norm
        
        # 2. Compute the dual weight
        d_weight_ortho = self._ortho_spatial(grad)
        d_weight = d_weight_ortho * scalar_factor
        
        # 3. Print Debug Info
        
        return [d_weight]

    def _ortho_spatial(self, weight):
        """SVD orthogonalize each [fanout, fanin] slice over spatial dims."""
        k = self.kernel_size
        # weight shape: [fanout, fanin, k, k]
        result = torch.zeros_like(weight)
        for i in range(k):
            for j in range(k):
                result[:, :, i, j] = svd_orthogonalize(weight[:, :, i, j])
        return result
