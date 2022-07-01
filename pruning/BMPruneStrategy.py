from abc import abstractmethod
from time import sleep
from turtle import forward
import types
import torch
import bmtrain as bmt
import torch.nn.functional as F
from torch.autograd import Variable
import model_center

limit_a, limit_b, epsilon = -.1, 1.1, 1e-4

class BMPruneStrategy(bmt.DistributedModule):
    def __init__(self,
        targets,
        type):
        '''
        :param targets: List of Linear(), targets to be pruned.
        :param type: 'pre', 'post', whether to add the mask pre or post the layer.
        '''
        super().__init__()
        self.targets = targets
        self.type = type
    
    @abstractmethod
    def get_mask(self):
        pass

    def print_targets(self):
        bmt.print_rank(self.targets)

    @abstractmethod
    def apply_mask(self, x):
        pass

    @abstractmethod
    def set_optimizer(self, optimizer):
        pass

    @abstractmethod
    def get_sparsity(self):
        pass

    def inject_mask(self, model):
        for k, v in model.named_modules():
            if k in self.targets:
                v.forward_without_mask = v.forward
                if self.type == 'pre':
                    def _forward(module_self, x, **kwargs):
                        x = self.apply_mask(x)
                        return module_self.forward_without_mask(x, **kwargs)
                elif self.type == 'post':
                    def _forward(module_self, *input, **kwargs):
                        x = module_self.forward_without_mask(*input, **kwargs)
                        return self.apply_mask(x)
            
                v.forward = types.MethodType(_forward, v)

    @abstractmethod
    def inject_sparsity(self, calculator):
        pass


class HardConcretePruning(BMPruneStrategy):
    def __init__(self, dim, targets):
        self.dim = dim
        
        super().__init__(
            targets = targets,
            type = 'post'
        )
        self.loga =  torch.nn.Parameter(
            torch.FloatTensor([2.5]*dim)
        )
        bmt.synchronize()

    def set_optimizer(self, optimizer):
        optimizer.add_param_group({'params': self.parameters()})

    def quantile_concrete(self, x, loga):
        y = torch.sigmoid((torch.log(x) - torch.log(1-x) + loga))
        return y * (limit_b - limit_a) + limit_a

    def get_eps(self, size):
        eps = torch.FloatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = torch.nn.Parameter(eps, requires_grad=False)
        return eps

    def get_mask(self):
        z = self.quantile_concrete(self.get_eps(self.loga.size()), self.loga)
        z = F.hardtanh(z, min_val=0, max_val=1)
        z = z.to(torch.half)
        return z

    def apply_mask(self, x):
        z = self.get_mask().to(x.device)
        x = x * z
        return x

    def get_sparsity(self):
        shift = torch.FloatTensor([2.4])
        shift = Variable(shift)
        return torch.sigmoid(self.loga+shift).mean()

    def print_mask(self):
        bmt.print_rank(self.loga)
        avg = torch.FloatTensor([0.5]*self.dim)
        avg = torch.nn.Parameter(avg, requires_grad=False)
        bmt.print_rank(self.quantile_concrete(avg, self.loga).sum()/self.dim)


class MHALayerPruning(HardConcretePruning):
    def __init__(self, layer):
        self.layer = layer
        super().__init__(
            dim = 1,
            targets = ['encoder.layers.'+str(layer)+'.self_att.self_attention']
        )

    def inject_sparsity(self, calc):
        space_q = calc.encoder.layers[self.layer].attn.space_q
        f = space_q.get_dim
        space_q.get_dim = lambda : f()*self.get_sparsity()

        space_k = calc.encoder.layers[self.layer].attn.space_k
        f = space_k.get_dim
        space_k.get_dim = lambda : f()*self.get_sparsity()

        space_v = calc.encoder.layers[self.layer].attn.space_v
        f = space_v.get_dim
        space_v.get_dim = lambda : f()*self.get_sparsity()



class FFNLayerPruning(HardConcretePruning):
    def __init__(self, layer):
        self.layer = layer
        super().__init__(
            dim = 1,
            targets = ['encoder.layers.'+str(layer)+'.ffn.ffn']
        )

    def inject_sparsity(self, calc):
        space_int = calc.encoder.layers[self.layer].ffn.space_ff
        f = space_int.get_dim
        space_int.get_dim = lambda : f()*self.get_sparsity()

class AttentionHeadPruning(BMPruneStrategy): # TODO: not complete
    def __init__(self, num_heads, layer):
        self.num_heads = num_heads
        self.mask = torch.rand(num_heads, dtype = torch.half).view(num_heads, 1)
        super().__init__(
            targets = ['encoder.layers.'+str(layer)+'.self_att.self_attention.attention_out'],
            type = 'pre')
        
        
    def apply_mask(self, x):
        '''
        :param x: (batch_size, dim_model, num_heads * dim_head)
        '''
        batch_size, dim_model, dim_last = x.size()
        num_heads = self.num_heads
        dim_head = dim_last/num_heads
        x = x.view(batch_size, dim_model, num_heads, dim_head)
        x = x * self.mask
        x = x.view(batch_size, dim_model, dim_last)
        return x


class FFNIntermediatePruning(HardConcretePruning): 
    def __init__(self, dim_int, layer):
        self.dim_int = dim_int
        self.layer = layer
        super().__init__(
            dim = dim_int,
            targets = ['encoder.layers.'+str(layer)+'.ffn.ffn.w_in'],
        )

    def inject_sparsity(self, calc):
        space_int = calc.encoder.layers[self.layer].ffn.space_ff
        f = space_int.get_dim
        space_int.get_dim = lambda : f()*self.get_sparsity()

    