x = x[:, 1:, :]  # Remove class token

batch_size, tokens, dimensions = x.shape
e = dimensions // self.groups
x = x.reshape((batch_size, tokens, e, self.groups))

x = x.clamp(min=self.eps).pow(self.p_params)
x = x.mean(dim=1)
x = x.pow(1. / self.p_params)

x = x.reshape((batch_size, dimensions))
