import torch
import torch.nn as nn

from onmt.utils.misc import aeq 
from onmt.utils.loss import LossComputeBase #, NMTLossCompute
from onmt.modules.copy_generator import CopyGeneratorLossCompute, CopyGenerator, collapse_copy_scores


class LossWrapper(LossComputeBase):
    def __init__(self, base_loss):
        criterion = base_loss.criterion
        generator = base_loss.generator
        super(LossWrapper, self).__init__(criterion, generator)

        self.base_loss = base_loss

    def _make_shard_state(self, *args, **kwargs):
        shard = self.base_loss._make_shard_state(*args, **kwargs)
        return self.wrap_shard_state(shard, *args, **kwargs)
    
    def wrap_shard_state(self, shard, *args, **kwargs):
        return shard

    def _compute_loss(self, *args, **kwargs):
        loss, stats = self.base_loss._compute_loss(*args, **kwargs)
        return self.wrap_loss(loss, stats, *args, **kwargs)
        
    def wrap_loss(self, loss, stats, *args, **kwargs):
        return loss, stats

class AbstractiveLossCompute(LossWrapper):
    def __init__(self, base_loss):
        super(AbstractiveLossCompute, self).__init__(base_loss)
        self.agg = torch.sum
        self.similarity = torch.nn.functional.cosine_similarity
        self.metric = "cosine"
        
    def wrap_loss(self, loss, stats, batch, src_embs, tgt_embs, output, target,
                      std_attn=None, coverage_attn=None,
                      **kwargs):
        nopad_mask = batch.tgt[1:, :, 0].ne(self.padding_idx)
        tgt_lens = nopad_mask.int().sum(0)
        tgt_lens.detach().requires_grad_(False)
        
        def _masked_embs(embs, lengths):
            masked = embs * 1.0
            # print(lengths, lengths.size())
            for i, l in enumerate(lengths):
                masked[l:, i] *= 0.0
            return masked

        if self.metric == "cosine":
            src, src_lens = batch.src
            masked_src_embs = _masked_embs(src_embs, src_lens)
            masked_tgt_embs = _masked_embs(tgt_embs, tgt_lens)
            masked_out_embs = _masked_embs(output, tgt_lens)

            agg_src = self.agg(masked_src_embs, dim=0)
            agg_tgt = self.agg(masked_tgt_embs, dim=0)
            agg_out = self.agg(masked_out_embs, dim=0)

            tgt_sim = self.similarity(agg_tgt, agg_src, dim=-1)
            out_sim = self.similarity(agg_out, agg_src, dim=-1)

            delta_sim = torch.exp(torch.abs(out_sim - tgt_sim))

            abs_loss = (delta_sim * tgt_lens.float()).sum()

            stats.abs_loss += abs_loss.item()

            loss += abs_loss
        else:
            raise ValueError("Unknow metric '%s'" % str(self.metric))


        return loss, stats
