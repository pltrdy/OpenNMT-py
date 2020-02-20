import torch
import torch.nn as nn

from onmt.utils.misc import aeq 
from onmt.utils.loss import LossComputeBase #, NMTLossCompute
from onmt.modules.copy_generator import CopyGeneratorLossCompute, CopyGenerator, collapse_copy_scores


def assert_size(a, b): 
    if isinstance(a, torch.Tensor):
        size_a = list(a.size())
    else:
        size_a = list(a)

    if isinstance(b, torch.Tensor):
        size_b = list(b.size())
    else:
        size_b = list(b)
    
    if len(size_a) != len(size_b):
        raise ValueError("Different number of dimensions %s != %s"
                         % (str(size_a), str(size_b)))

    for i, (sa, sb) in enumerate(zip(size_a, size_b)):
        if sa != sb: 
            raise ValueError("Different size for dimension %d, %d != %d in %s %s"
                             % (i, sa, sb, str(size_a), str(size_b)))



def dist_cross_entropy(P, Q, name="noname", eps=1e-20):
    """ 
        P, Q, two distributions (normalized)
        P: [n x d]
        Q: [n x d]

    """
    PlogQ = P*(Q+eps).log()

    if not PlogQ.eq(PlogQ).all():
        print("P: ", torch.min(P), torch.max(P))
        print("Q: ", torch.min(Q), torch.max(Q))
        print("PlogQ: ", torch.min(PlogQ), torch.max(PlogQ))
        raise ValueError("NaN in cross_entropy %s" % str(name))
    CE = -PlogQ.sum(dim=-1) 
    return CE

def vec_cross_entropy(U, V, eps=1e-20):
    """
        vectors -> distributions -> cross entropy
        U: [n x d]
        V: [n x d]
    """
    P = torch.nn.functional.softmax(U, dim=-1)
    Q = torch.nn.functional.softmax(V, dim=-1)
    return dist_cross_entropy(P, Q, eps=eps)

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
    def __init__(self, base_loss, abstract_metric="cosine", abstract_lambda=1.0):
        super(AbstractiveLossCompute, self).__init__(base_loss)
        self.agg = torch.sum
        self.cosine_similarity = torch.nn.functional.cosine_similarity
        self.metric = abstract_metric
        self.abstract_lambda = abstract_lambda
        
    def _mask(self, t, lengths):
        masked = t * 1.0
        # print(lengths, lengths.size())
        for i, l in enumerate(lengths):
            masked[l:, i] *= 0.0
        return masked

    def wrap_loss(self, loss, stats, batch, src_embs, tgt_embs, output, target,
                      std_attn=None,
                      coverage_attn=None,
                      decoder_context=None,
                      **kwargs):
        nopad_mask = batch.tgt[1:, :, 0].ne(self.padding_idx)
        tgt_lens = nopad_mask.int().sum(0)
        tgt_lens.detach().requires_grad_(False)
        abs_loss = None

        fct = getattr(self, self.metric)
        abs_loss = fct(loss, stats, batch, src_embs, tgt_embs, output, target, tgt_lens,
                   std_attn=std_attn, coverage_attn=coverage_attn, decoder_context=decoder_context, **kwargs)

        if not self.metric == "monitor":
            loss = (loss * self.abstract_lambda) + abs_loss
            stats.abs_loss += abs_loss.item()
        return loss, stats
    
    def monitor(self, loss, stats, *args, **kwargs):
        metrics = {
            "ctxt_cos": self.context_cosine,
            "ctxt_xent": self.context_xent,
            "sh_cos": self.shifted_cosine,
            "cos": self.cosine_loss,
        }
        results = {}

        for k, f in metrics.items():
            _loss = f(loss, stats, *args, **kwargs)
            results[k] = _loss.item()
            del _loss

        stats.update_additional_metrics(results)
        return None
    
    def output_cosine(self, *args, **kwargs):
        sim_fct = self.cosine_similarity
        return self._output_loss(sim_fct, *args, **kwargs)

    def output_xent(self, *args, **kwargs):
        sim_fct = vec_cross_entropy
        return self._output_loss(sim_fct, *args, **kwargs)

    def _output_loss(self, sim_fct, loss, stats, batch, src_embs, tgt_embs, output, target, tgt_lens,
                  std_attn=None, coverage_attn=None, decoder_context=None, **kwargs):
        tgt_embs = tgt_embs[1:]
        output = output[:-1]
        tgt_lens -= 1
        
        n, b, d = tgt_embs.size()
        def _bottle(t):
            """ from [n x b x d] 
                to   [b x (n*d)]
            """
            return t.transpose(0, 1).contiguous().view(b, (n*d)).contiguous()

        bott_tgt_embs = _bottle(self._mask(tgt_embs, tgt_lens))
        bott_out_embs = _bottle(self._mask(output, tgt_lens))

        out_sim = sim_fct(bott_out_embs, bott_tgt_embs)
        abs_loss = (out_sim * tgt_lens.float()).sum()
        return abs_loss

    def context_cosine(self, *args, **kwargs):
        sim_fct = self.cosine_similarity
        return self._context_loss(sim_fct, *args, **kwargs)

    def context_xent(self, *args, **kwargs):
        sim_fct = vec_cross_entropy 
        return self._context_loss(sim_fct, *args, **kwargs)

    def _context_loss(self, sim_fct, loss, stats, batch, src_embs, tgt_embs, output, target, tgt_lens,
                      std_attn=None, coverage_attn=None, decoder_context=None, **kwargs):
        assert decoder_context is not None
        tgt_embs = tgt_embs[1:]
        output = output[:-1]
        tgt_lens -= 1
        src_embs = decoder_context.transpose(0, 1)[:-1].contiguous()

        assert_size(src_embs, tgt_embs)
        assert_size(src_embs, output)

        n, b, d = src_embs.size()
        def _bottle(t):
            """ from [n x b x d] 
                to   [b x (n*d)]
            """
            return t.transpose(0, 1).contiguous().view(b, (n*d)).contiguous()
       
        # need masking after cosine, not before
        bott_src_embs = _bottle(self._mask(src_embs, tgt_lens))
        bott_tgt_embs = _bottle(self._mask(tgt_embs, tgt_lens))
        bott_out_embs = _bottle(self._mask(output, tgt_lens))
        
        assert_size(bott_src_embs, bott_tgt_embs)
        assert_size(bott_src_embs, bott_out_embs)
      
        # tgt_sim = _unbottle(sim_fct(bott_tgt_embs, bott_src_embs), dim=1)
        # out_sim = _unbottle(sim_fct(bott_out_embs, bott_src_embs), dim=1)
        
        tgt_sim = sim_fct(bott_tgt_embs, bott_src_embs)
        out_sim = sim_fct(bott_out_embs, bott_src_embs)


        # masked_tgt_sim = _mask(tgt_sim, tgt_lens)
        # masked_out_sim = _mask(out_sim, tgt_lens)
        masked_tgt_sim = tgt_sim
        masked_out_sim = out_sim

        delta_sim = torch.exp(torch.abs(masked_out_sim - masked_tgt_sim))
        abs_loss = (delta_sim * tgt_lens.float()).sum()
        return abs_loss

    def shifted_cosine(self, loss, stats, batch, src_embs, tgt_embs, output, target, tgt_lens,
                      std_attn=None, coverage_attn=None, decoder_context=None, **kwargs):
        # pb with cosine is it considers "tgt_embs" to be targets
        # where tgt_embs really is decoder inputs embeddings
        # so we need to shift before applying cosine
        # and we never have emb(<eos>) so we exclude last output aswell
        # by decrementing all tgt lens
        tgt_embs = tgt_embs[1:]
        tgt_lens -= 1
        return self.cosine_loss(loss, stats, batch, src_embs, tgt_embs, output, target, tgt_lens,
                                std_attn=std_attn, coverage_attn=coverage_attn, decoder_context=decoder_context, **kwargs)

    def cosine_loss(self, loss, stats, batch, src_embs, tgt_embs, output, target, tgt_lens,
                      std_attn=None, coverage_attn=None, decoder_context=None, **kwargs):
        src, src_lens = batch.src
        masked_src_embs = self._mask(src_embs, src_lens)
        masked_tgt_embs = self._mask(tgt_embs, tgt_lens)
        masked_out_embs = self._mask(output, tgt_lens)

        agg_src = self.agg(masked_src_embs, dim=0)
        agg_tgt = self.agg(masked_tgt_embs, dim=0)
        agg_out = self.agg(masked_out_embs, dim=0)
        
        tgt_sim = self.cosine_similarity(agg_tgt, agg_src, dim=-1)
        out_sim = self.cosine_similarity(agg_out, agg_src, dim=-1)

        delta_sim = torch.exp(torch.abs(out_sim - tgt_sim))

        abs_loss = (delta_sim * tgt_lens.float()).sum()
    
        return loss
       
