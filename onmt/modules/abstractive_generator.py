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
    def __init__(self, base_loss, abstract_metric="cosine", abstract_lambda=1.0, abstract_pen=None):
        super(AbstractiveLossCompute, self).__init__(base_loss)
        self.agg = torch.sum
        self.cosine_similarity = torch.nn.functional.cosine_similarity
        self.metric = abstract_metric
        self.abstract_lambda = abstract_lambda
        self.abstract_pen = abstract_pen
        
        if abstract_pen is not None:
            assert self.metric == "abstrpen", "abstract pen can only be set with abstrpen metric"
        
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
            # print(loss)
            loss = (loss * self.abstract_lambda) + abs_loss
            # print(loss)
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
    def abstrpen(self, loss, stats, batch, src_embs, tgt_embs, output, target, tgt_lens, *args, **kwargs):
        def align(X, Y):
            """Given two matrices X [n x b], Y [m x b]
               Return mask of Y being in X
            """
            def twodims(t):
                if len(t.size()) == 3:
                    assert t.size(2) == 1, "Invalid >1 on 3rd dim"
                    return t.squeeze(2)
                return t
            X = twodims(X)
            Y = twodims(Y)
            
            # do not consider

            mask = X.gt(1)
            
            n, b = X.size()
            m, b = Y.size()

            # Reshape both to [m x n x b]
            XX = X.unsqueeze(0).expand(m, n, b)
            YY = Y.unsqueeze(1).expand(m, n, b)
        
            eq = XX.eq(YY)

            Y_in_X = eq.sum(1).gt(0)
            return Y_in_X

        src, tgt = batch.src, batch.tgt
        src, src_lens = src
 
        src = src[2:]
        tgt = tgt[3:]
        output = output[2:]
 
        m, b, feats = src.size()
        n, _b, _feats = tgt.size()
        __n, __b, dim = output.size()
        assert _b == b
        bottled_output = self._bottle(output)
        scores = self.generator(bottled_output).view(n, b, -1)
        pred = scores.argmax(2)
        
        tgt_in_src = align(src, tgt)
        assert list(tgt_in_src.size()) == [n, b]
    
        penalty = self.abstract_pen if self.abstract_pen is not None else 0.5
        penalties = torch.ones(scores.size(), dtype=torch.float, device=scores.device)
        for i in range(b):
            # print(penalties[:, i, :].size(), src[:, i, 0]())
            penalties[:, i, :] = penalties[:, i, :].index_fill(-1, src[:, i, 0], penalty)
        scores =  scores * penalties
        loss = torch.nn.functional.nll_loss(self._bottle(scores), tgt.view(-1), ignore_index=self.padding_idx, reduction='sum')
        return loss


    def abstrtok(self, loss, stats, batch, src_embs, tgt_embs, output, target, tgt_lens, *args, **kwargs):
        """ loss related to abstr token
            we consider tgt first token to be `a` the second `l`
        """
        # loss, stats, batch, src_embs, tgt_embs, output, target, tgt_lens
        deciles = [
            0.25508317929759705, 0.35135135135135137, 0.4242424242424242, 0.49122807017543857, 0.54838709667741935,
            0.603448275862069, 0.65625, 0.7115384615384616, 0.774207821866972
        ]
        # vocab[16] = COPY_0
        atok_offset = 16

    
        src, tgt = batch.src, batch.tgt
        src, src_lens = src
        
        src = src[2:]
        tgt = tgt[3:]
        output = output[2:]
        
        # print(src.size())
        # print(tgt.size())
        # print(output.size())

        m, b, feats = src.size()
        n, _b, _feats = tgt.size()
        __n, __b, dim = output.size()
        assert _b == b
        assert feats == _feats
        
        assert __n == n
        assert __b == b

        
        bottled_output = self._bottle(output)

        # [n x b x voc]
        scores = self.generator(bottled_output).view(n, b, -1)
        
        # [n x b]
        pred = scores.argmax(2)
        __n, __b = pred.size()
        assert __n == n
        assert __b == b

        def n_copy(X, Y):
            """Given two matrices X [n x b], Y [m x b]
               Return \sum{x_i == y_j}
            """
            def twodims(t):
                if len(t.size()) == 3:
                    assert t.size(2) == 1, "Invalid >1 on 3rd dim"
                    return t.squeeze(2)
                return t
            X = twodims(X)
            Y = twodims(Y)
            
            # do not consider

            mask = X.gt(1)
            
            n, b = X.size()
            m, b = Y.size()

            # Reshape both to [m x n x b]
            XX = X.unsqueeze(0).expand(m, n, b)
            YY = Y.unsqueeze(1).expand(m, n, b)
        
            eq = XX.eq(YY)

            Y_in_X = eq.sum(1).gt(0)

            count = Y_in_X.float().sum(0)

            return count

        nc = n_copy(pred, pred).sum().item()
        
        assert nc == (n * b), "%s != %s" % (str(nc), str(n*b))

        gold_copy = n_copy(src, tgt)
        pred_copy = n_copy(src, pred)
        # MSE-Sum divide by, n better implement our own
        # loss = torch.nn.functional.mse_loss(pred_copy, gold_copy, reduction='sum')
        loss = ((pred_copy - gold_copy) ** 2) 
        loss = loss.sum()
        return loss


    def abstrrate(self, loss, stats, batch, src_embs, tgt_embs, output, target, tgt_lens, *args, **kwargs):
        """ loss related to abstr%
        """
        # loss, stats, batch, src_embs, tgt_embs, output, target, tgt_lens
        src, tgt = batch.src, batch.tgt
        src, src_lens = src
        
        src = src[2:]
        tgt = tgt[3:]
        output = output[2:]
        
        m, b, feats = src.size()
        n, _b, _feats = tgt.size()
        __n, __b, dim = output.size()
        assert _b == b
        assert feats == _feats
        
        assert __n == n
        assert __b == b

        
        bottled_output = self._bottle(output)

        # [n x b x voc]
        scores = self.generator(bottled_output).view(n, b, -1)
        
        # [n x b]
        pred = scores.argmax(2)
        __n, __b = pred.size()
        assert __n == n
        assert __b == b

        def n_copy(X, Y):
            """Given two matrices X [n x b], Y [m x b]
               Return \sum{x_i == y_j}
            """
            def twodims(t):
                if len(t.size()) == 3:
                    assert t.size(2) == 1, "Invalid >1 on 3rd dim"
                    return t.squeeze(2)
                return t
            X = twodims(X)
            Y = twodims(Y)
            
            # do not consider

            mask = X.gt(1)
            
            n, b = X.size()
            m, b = Y.size()

            # Reshape both to [m x n x b]
            XX = X.unsqueeze(0).expand(m, n, b)
            YY = Y.unsqueeze(1).expand(m, n, b)
        
            eq = XX.eq(YY)

            X_in_Y = eq.sum(1).gt(0)

            count = X_in_Y.float().sum(0)

            return count

        nc = n_copy(pred, pred).sum().item()
        
        assert nc == (n * b), "%s != %s" % (str(nc), str(n*b))

        gold_copy = n_copy(src, tgt)
        pred_copy = n_copy(src, pred)
        # MSE-Sum divide by, n better implement our own
        # loss = torch.nn.functional.mse_loss(pred_copy, gold_copy, reduction='sum')
        loss = ((pred_copy - gold_copy) ** 2) 
        loss = loss.sum()
        return loss

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
       
