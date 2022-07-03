import bmtrain as bmt
import torch

class BMPruneLossController:
    def get_loss(self):
        return 0

class BrutePenalty(BMPruneLossController):
    def __init__(self, lmbd, size_calculator):
        self.lmbd = lmbd
        self.size_calculator = size_calculator
        self.original_size = self.size_calculator.get_size()
        bmt.print_rank(self.original_size)
    def get_loss(self):
        return (self.lmbd * (self.size_calculator.get_size() / self.original_size)).to(torch.half)

class LagrangianPenalty(BMPruneLossController):
    def __init__(self, lmbd, size_calculator, target_sparsity, optimizer):
        self.lmbd = lmbd
        self.size_calculator = size_calculator
        self.l1 = torch.nn.Parameter(
            torch.HalfTensor([0.0]).cuda()
        )
        self.l2 = torch.nn.Parameter(
            torch.HalfTensor([0.0]).cuda()
        )
        self.original_size = self.size_calculator.get_size()
        self.target_sparsity = target_sparsity
        optimizer.add_param_group({'params': self.l1, 'maximize': True, 'lr': 1e-2})
        optimizer.add_param_group({'params': self.l2, 'maximize': True, 'lr': 1e-2})
    
    def get_loss(self):
        s = (self.size_calculator.get_size() / self.original_size).to(torch.half)
        t = self.target_sparsity
        bmt.print_rank(self.l1, self.l2, s, t)
        return self.lmbd * (self.l1*(s-t) + self.l2*(s-t)*(s-t))

class LinearSpace:
    def __init__(
        self,
        dim : int
    ):
        self.dim = dim
    
    def get_dim(self):
        return self.dim

class LinearLayer:
    def __init__(
        self,
        space_in : LinearSpace,
        space_out : LinearSpace,
    ):
        self.space_in = space_in
        self.space_out = space_out

    def get_size(self):
        return self.space_in.get_dim()*self.space_out.get_dim()

class Attention:
    def __init__(
        self,
        space_model : LinearSpace,
        num_heads : int,
        dim_head : int,
        shared_kv : bool = False
    ):
        dim_head_kv = 1 if shared_kv else dim_head

        self.space_q = LinearSpace(num_heads * dim_head)
        self.space_k = LinearSpace(num_heads * dim_head_kv)
        self.space_v = LinearSpace(num_heads * dim_head_kv)

        self.proj_q = LinearLayer(space_model, self.space_q)
        self.proj_k = LinearLayer(space_model, self.space_k)
        self.proj_v = LinearLayer(space_model, self.space_v)

    def get_size(self):
        return self.proj_q.get_size()+self.proj_k.get_size()+self.proj_v.get_size()

class FeedForward:
    def __init__(
        self,
        space_model: LinearSpace,
        dim_ff: int
    ):
        self.space_ff = LinearSpace(dim_ff)
        self.w_up = LinearLayer(space_model, self.space_ff)
        self.w_down = LinearLayer(self.space_ff, space_model)

    def get_size(self):
        return self.w_up.get_size()+self.w_down.get_size()



class TransformerBlock:
    def __init__(
        self,
        space_model : LinearSpace,
        dim_ff : int,
        num_heads : int,
        dim_head : int
    ):
        self.attn = Attention(
            space_model,
            num_heads,
            dim_head
        )
        self.ffn = FeedForward(
            space_model,
            dim_ff
        )
    
    def get_size(self):
        return self.attn.get_size()+self.ffn.get_size()


class Encoder:
    def __init__(
        self,
        num_layers : int,
        space_model : LinearSpace,
        dim_ff : int,
        num_heads : int,
        dim_head : int
    ):
        self.layers = [TransformerBlock(
            space_model = space_model,
            dim_ff = dim_ff,
            num_heads=num_heads,
            dim_head=dim_head
        ) for _ in range(num_layers)]

    def get_size(self):
        return sum(layer.get_size() for layer in self.layers)


class GPT2SizeCalculator():
    def __init__(
        self,
        config
    ):
        self.space_model = LinearSpace(config.dim_model)
        self.encoder = Encoder(
            num_layers = config.num_layers,
            space_model = self.space_model,
            dim_ff = config.dim_ff,
            num_heads = config.num_heads,
            dim_head = config.dim_head
        )

    def get_size(self):
        return self.encoder.get_size()