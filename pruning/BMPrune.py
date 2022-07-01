from typing import List
import bmtrain as bmt
from .BMPruneLoss import BMPruneLossController
from .BMPruneStrategy import BMPruneStrategy

class BMPrune:
    def set_forward(
        self,
        model,
        forward_fn,
        prune_loss_controller : BMPruneLossController,
        strategies: List[BMPruneStrategy]):

        for strategy in strategies:
            strategy.inject_mask(model)
            strategy.inject_sparsity(prune_loss_controller.size_calculator)
        
        def forward(model, dec_input, dec_length, targets, loss_func):
            outputs = forward_fn(
                model, dec_input, dec_length, targets, loss_func
            )

            loss = outputs[0]
            p_loss = prune_loss_controller.get_loss()
            bmt.print_rank(p_loss)
            loss = loss + p_loss

            outputs[0] = loss
            outputs = outputs + [p_loss, ]

            return outputs

        return forward