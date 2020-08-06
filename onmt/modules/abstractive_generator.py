import torch
import torch.nn as nn

from onmt.utils.misc import aeq 
from onmt.utils.loss import LossComputeBase #, NMTLossCompute
from onmt.modules.copy_generator import CopyGeneratorLossCompute, CopyGenerator, collapse_copy_scores

def no_nan(t, at=""):
    e = t.eq(t)
    if not e.all():
        raise ValueError("NaN at: '%s', %s" % (at, str(e)))

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
        loss, stats, sent_loss, scores = self.base_loss._compute_loss(*args, ret_scores=True, **kwargs)
        wloss, wstats = self.wrap_loss(loss, stats, *args, scores=scores, **kwargs)
        # print(loss, wloss)
        return wloss, wstats, sent_loss
        
    def wrap_loss(self, loss, stats, *args, **kwargs):
        raise NotImplemented

class AbstractiveLossCompute(LossWrapper):
    def __init__(self, base_loss, abstract_metric="cosine", abstract_lambda=1.0, abstract_pen=None, abstract_ntoks=10):
        super(AbstractiveLossCompute, self).__init__(base_loss)
        self.agg = torch.sum
        self.cosine_similarity = torch.nn.functional.cosine_similarity
        self.metric = abstract_metric
        self.abstract_lambda = abstract_lambda
        self.abstract_pen = abstract_pen
        self.abstract_ntoks = abstract_ntoks
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
            _ = loss
            loss = (loss * self.abstract_lambda) + abs_loss
            # print(loss)
            stats.abs_loss += abs_loss.item()
            # print("Prevloss: ", _, "Act: ", loss)
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

    def copypen(self, loss, stats, batch, src_embs, tgt_embs, output, target, tgt_lens, *args, scores=None, **kwargs):
        """
            minimize p(w_t \in x-y*)
            i.e. do not copy words from x not in the target
        """
        if scores is None:
            raise ValueError()
        epsilon = 10e-6

        # loss, stats, batch, src_embs, tgt_embs, output, target, tgt_lens
        src, tgt = batch.src, batch.tgt
        src, src_lens = src
        
        src = src[2:, :, 0]
        tgt = tgt[3:, :, 0]
        output = output[2:]
        
        # m, b, feats = src.size()
        # n, _b, _feats = tgt.size()
        # __n, __b, dim = output.size()
        # assert _b == b
        # assert feats == _feats
        # 
        # assert __n == n
        # assert __b == b
        verbose = True and False
        def log(*args, **kwargs):
            def _str(o):
                s = str(o)
                if type(o) == torch.Tensor:
                    s = "%s %s" % (list(o.size()), s)
                return s

            if verbose:
                print(" ".join([_str(_) for _ in args]), **kwargs)
        
        X = src
        Ys = tgt
        # finding x inter y*
        n, b = X.size()
        m, _b = Ys.size()
        assert b == _b

        
        # bottled_output = self._bottle(output)
        # scores = self.generator(bottled_output).view(m, b, -1)
        scores = scores.view(m+2, b, -1)[2:]
        log("scores", scores) 
        no_nan(scores, "scores")

        # Reshape both to [m x n x b]
        XX = X.unsqueeze(0).expand(m, n, b)
        YYs = Ys.unsqueeze(1).expand(m, n, b)
        
        # do not consider spe toks
        last_spetok = 3
        maskX = X.gt(last_spetok).long()
        ntokX = maskX.sum(0)
        maskXX = XX.gt(last_spetok).long()
        maskYs = Ys.gt(last_spetok).long()
        ntokYs = maskYs.sum(0)
        maskYYs = YYs.gt(last_spetok).long()

        ntoky = maskYYs.sum(1).gt(0).float().sum(0)

        # equality without mask
        E = XX.eq(YYs) * maskXX * maskYYs
        no_nan(E, "E")

        # [n, b]
        X_in_Ys = E.sum(0).gt(0)
        assert_size(X_in_Ys, X)

        X_notin_Ys = ~X_in_Ys
        Xtoks_notin_Ys = X * X_notin_Ys.long()
        # Xtoks_notin_Ys = X[X_notin_Ys]
        log("Xtoks_notin_Ys", Xtoks_notin_Ys) 
        no_nan(Xtoks_notin_Ys, "Xtoks_notin_Ys")
        if True:
            # scores [m, b, voc]
            # FROM logprob TO prob
            logprobs = scores - epsilon
            assert logprobs.lt(0).all(), "logprobs>= 0 %s\n%s" % (str(logprobs[logprobs.ge(0)]), str(logprobs.ge(0).nonzero()))
            assert logprobs.gt(float("-inf")).all(), "logprobs<=inf %s" % str(logprobs[logprobs.le(float("-inf"))])
            no_nan(logprobs, "logprobs")
            
            probs = logprobs.exp()
            assert probs.ge(0).all(), "probs<= 0 %s\n%s" % (str(probs[probs.le(0)]), str(probs.le(0).nonzero()))
            assert probs.le(1).all(), "probs>= 1 %s\n%s" % (str(probs[probs.ge(1)]), str(probs.ge(1).nonzero()))
            no_nan(probs, "probs")
            log("probs", probs)
           
            # SUM over time and NORMALIZE
            tot_probs = (probs * maskYs.unsqueeze(2).float()).sum(0) / ntokYs.unsqueeze(1).float()
            log("tot_probs", tot_probs) 
            assert tot_probs.ge(0).all(), "tot_probs<= 0 %s\n%s" % (str(tot_probs[tot_probs.le(0)]), str(tot_probs.le(0).nonzero()))
            assert tot_probs.le(1).all(), "tot_probs>= 1 %s\n%s" % (str(tot_probs[tot_probs.ge(1)]), str(tot_probs.ge(1).nonzero()))
            no_nan(tot_probs, "tot_probs")
            log("tot_probs", tot_probs)
           
            # GATHER badcopy probs
            badcopy_probs = tot_probs.gather(1, Xtoks_notin_Ys.t()) 
            
            # badcopy_probs *= Xtoks_notin_Ys.t().gt(last_spetok).float()
            assert badcopy_probs.le(1).all()
            assert badcopy_probs.ge(0).all()
            log("badcopy_probs", badcopy_probs)

            # Penalty = sum(normalized) across X\Y*
            # badcopy_tot_probs = badcopy_probs.sum(1)
            normalize = X_notin_Ys.float().sum(0)
            zero_denum = normalize.eq(0.0)
            denum = normalize + zero_denum.float()
            # _denum = badcopy_probs.ne(0).float().sum(1)
            log("denum", denum)
            # assert denum.eq(_denum).all(), "DEN: %s\n_DEN%s" % (str(denum), str(_denum))
            
            # Remove token 0 scores
            badcopy_pen_probs = (1 - badcopy_probs) * Xtoks_notin_Ys.t().gt(last_spetok).float()

            badcopy_pen = badcopy_pen_probs.sum(1) 
            log("badcopy_pen(bef norm)", badcopy_pen)
            badcopy_pen = (badcopy_pen / denum)

            # add epsilon for 0 probs
            badcopy_pen += badcopy_pen.eq(0).float() * epsilon

            # remove epsilon for 1 probs
            badcopy_pen -= badcopy_pen.eq(1).float() * epsilon

            # badcopy_pen = badcopy_pen / denum
            # _badcopy_pen = (1 - _badcopy_tot_probs) / denum
            # assert badcopy_pen.eq(_badcopy_pen).all(), "BCP: %s\n_BCP: %s" % (str(badcopy_pen), str(_badcopy_pen))
            log("badcopy_pen", badcopy_pen)
            assert badcopy_pen.lt(1).all(), "badcopy_pen>=1 %s" % (str(badcopy_pen[badcopy_pen.ge(1)]))
            assert badcopy_pen.ge(0).all(), "badcopy_pen<=0 %s" % (str(badcopy_pen[badcopy_pen.le(0)]))
            
            badcopy_pen_logprobs = badcopy_pen.log()
            assert badcopy_pen_logprobs.lt(0).all()
            assert badcopy_pen_logprobs.gt(float("-inf")).all()

            badcopy_neglogprob = -badcopy_pen_logprobs

            
            loss = (badcopy_neglogprob * ntokYs * ntokX).sum()
            log("loss", loss)
            no_nan(loss, "loss (normalized by %s [%s])" % (str(X_notin_Ys.float().sum(0)), str(X_notin_Ys.float())))

        else:
            # scores [m, b, voc]
            # tot_pen_logprobs [b, voc]
            # min prob => max 1-prob => max log(1-prob) => min -log(1-prob)
            # we want to min probabilities of copy, therefore min:
            logprobs = scores - epsilon
            assert logprobs.lt(0).all(), "logprobs>= 0 %s\n%s" % (str(logprobs[logprobs.ge(0)]), str(logprobs.ge(0).nonzero()))
            assert logprobs.gt(float("-inf")).all(), "logprobs<=inf %s" % str(logprobs[logprobs.le(float("-inf"))])
            no_nan(logprobs, "logprobs")
            
            probs = logprobs.exp()
            assert probs.ge(0).all(), "probs<= 0 %s\n%s" % (str(probs[probs.le(0)]), str(probs.le(0).nonzero()))
            assert probs.le(1).all(), "probs>= 1 %s\n%s" % (str(probs[probs.ge(1)]), str(probs.ge(1).nonzero()))
            no_nan(probs, "probs")
            log("probs", probs)
            
            # => we want to max:
            penalty_probs = 1 - probs
            assert penalty_probs.ge(0).all(), "pen prob <= 0 %s" % str(penalty_probs[penalty_probs.le(0)])
            assert penalty_probs.le(1).all(), "pen prob >= 1 %s" % str(penalty_probs[penalty_probs.ge(1)])
            no_nan(penalty_probs, "penalty_probs")
            log("penalty_probs:", penalty_probs)
            # => max:
            # we sum (and norm) over all timesteps (be sure to do it on probs, not logprobs)
            # raise ValueError(penalty_probs.sum(0).size(), ntokX.size())
            tot_pen_probs = penalty_probs.sum(0) / ntokX.unsqueeze(1).float()
            log("tot_pen_probs", tot_pen_probs) 
            
            # => max
            tot_penalty_logprobs = tot_pen_probs.log() - epsilon
            assert tot_penalty_logprobs.lt(0).all(), "tot_penalty_logprobs>= 0 %s" % str(tot_penalty_logprobs[tot_penalty_logprobs.ge(0)])
            assert tot_penalty_logprobs.gt(float("-inf")).all(), "tot_penalty_logprobs<=inf %s" % str(tot_penalty_logprobs[tot_penalty_logprobs.le(float("-inf"))])
            no_nan(tot_penalty_logprobs, "tot_penalty_logprobs")

            # tot_pen_logprobs = penalty_logprobs.sum(0)
            # no_nan(tot_pen_logprobs, "tot_pen_logprobs")

            # Xtoks_notin_Ys.t() [b, n] w/ some 0 values that will gather scores of token 0
            # badcopy_logprobs [b, n]
            badcopy_logprobs = tot_penalty_logprobs.gather(1, Xtoks_notin_Ys.t()) 
            
            # clean 0-token scores gathered by "mistake"
            badcopy_logprobs *= Xtoks_notin_Ys.t().ne(0).float()

            # get *negative* log likelihood i.e. loss in [0, +inf[ 
            # so we want to minimize it, like common MaxLikelihood loss
            badcopy_neglogprobs = badcopy_logprobs * -1

            # print("badcopy_logprobs size:", badcopy_logprobs.size(), "X_notin_Ys size:", X_notin_Ys.size())
            log("badcopy_logprobs", badcopy_logprobs)
            assert badcopy_neglogprobs.lt(float("inf")).all(), "badcopy nelp inf, corresponding toks %s" % (str(Xtoks_notin_Ys.t()[badcopy_neglogprobs.lt.eq(float("inf"))]))
            assert badcopy_neglogprobs.gt(0).all(), "badcopy neglp 0, corresponding toks %s" % (str(Xtoks_notin_Ys.t()[badcopy_neglogprobs.le(0)]))
            no_nan(badcopy_logprobs, "badcopy_logprobs")
            # scores are neg log probs i.e. [0, +inf[
            # in general we want to max p => max log p => min -log p
            # here we want to min p => max (1-p) but our score is
            # s = -log p => -log[1-exp(-s)] = -log(1-p) giving neg log probs for 1-p
            

            # badcopy_neglogprobs = -torch.log(1-torch.exp(-badcopy_logprobs))
            # log("badcopy_neglogprobs", badcopy_neglogprobs)
            # or cheaper: just log prob instead of neglog
            # badcopy_neglogprobs = -badcopy_scores

             
            # normalize by #{X-Y*}
            # badcopy_neglogprobs = badcopy_neglogprobs.sum(1) / X_notin_Ys.float().sum(0)
            normalize = X_notin_Ys.float().sum(0)

            # remove zero denum and replace with 1.0
            src_len_norm = True 
            if src_len_norm:
                zero_denum = normalize.eq(0.0)
                denum = normalize + zero_denum.float()
                loss_norm = (badcopy_neglogprobs.sum(1) / denum).sum()
                loss = loss_norm
                no_nan(loss, "loss (normalized by %s [%s])" % (str(X_notin_Ys.float().sum(0)), str(X_notin_Ys.float())))
            else:
                loss = badcopy_neglogprobs.sum(1).sum()
                no_nan(loss, "loss (not normalized) %s" % (str(X_notin_Ys.float().sum(0))))

            # print("Loss Norm: %s\nLoss: %s" % (str(loss_norm), str(loss)))
        return loss





    def tokagg_mseloss(self, loss, stats, batch, src_embs, tgt_embs, output, target, tgt_lens, *args, **kwargs):
        src, tgt = batch.src, batch.tgt
        src, src_lens = src
 
        # tgt starts with <bos>
        tgt = tgt[1:]
        output = output
 
        m, b, feats = src.size()
        n, _b, _feats = tgt.size()
        __n, __b, dim = output.size()
        assert n == __n
        assert _b == b
        bottled_output = self._bottle(output)
        scores = self.generator(bottled_output).view(n, b, -1)
        # COPY_0 = 16
        # COPY_9 = 26
        ntoks = self.abstract_ntoks
        copy_offset = 16
        
        tgttok = tgt[0, :, 0] - copy_offset
        copytok_scores = scores[0, :, copy_offset:(copy_offset+ntoks)]
        assert_size(copytok_scores, [b, ntoks])

        copytok_probs = torch.nn.functional.softmax(copytok_scores)
        try:
            assert 0 <= tgttok.min().item() < ntoks, str(tgttok.min().item())
            assert 0 <= tgttok.max().item() < ntoks, str(tgttok.max().item())
        except AssertionError:
            print(tgt)
            print(tgttok)
            print("ntoks", ntoks)
            raise
        r = torch.arange(ntoks).to(scores.device).float()
        r = r.unsqueeze(0).expand_as(copytok_probs)
        

        copytok = (r * copytok_probs).sum(-1)
        
        # print("probs: ", copytok_probs[:5], "toks: ", copytok[:5], "tgt: ", tgttok[:5])
        
        # print(copytok_probs[:5], tgttok[:5])
        criterion = nn.MSELoss(reduce=True, reduction='sum')
        loss = criterion(copytok, tgttok.float())
        return loss

    

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

    def inv_abstrpen(self, loss, stats, batch, src_embs, tgt_embs, output, target, tgt_lens, *args, **kwargs):
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
        scores =  scores / penalties
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
               Return count of element of Y that is in X
               \sum{y \in Y}{1_{y \in X}}
            """
            def twodims(t):
                if len(t.size()) == 3:
                    assert t.size(2) == 1, "Invalid >1 on 3rd dim"
                    return t.squeeze(2)
                return t
            X = twodims(X)
            Y = twodims(Y)
            
                        
            n, b = X.size()
            m, b = Y.size()

            # Reshape both to [m x n x b]
            XX = X.unsqueeze(0).expand(m, n, b)
            YY = Y.unsqueeze(1).expand(m, n, b)
            
            # do not consider spe toks
            last_spetok = 3
            maskXX = XX.gt(last_spetok).long()
            maskYY = YY.gt(last_spetok).long()
            ntoky = maskYY.sum(1).gt(0).float().sum(0)

            # equality without mask
            E = XX.eq(YY) * maskXX * maskYY
            
            Y_in_X = E.sum(1).gt(0)

            count = Y_in_X.float().sum(0)

            return count, ntoky
 
        # assert nc == (n * b), "%s != %s" % (str(nc), str(n*b))

        gold_copy, ntoktgt = n_copy(src, tgt)
        pred_copy, ntokpred = n_copy(src, pred)
        # MSE-Sum divide by, n better implement our own
        # loss = torch.nn.functional.mse_loss(pred_copy, gold_copy, reduction='sum')
        rate_fct = "neglograte"
        rate_fct = "rate_mse"

        if rate_fct == "neglograte":
            pred_rate = pred_copy / ntokpred
            gold_rate = gold_copy / ntoktgt
            d_rate = torch.abs(pred_rate - gold_rate) # in [0, 1], 0 better
            l = locals()
            assert d_rate.le(1.0).all(), "\n"+"\n".join([
                "%s: %s" % (_, str(l[_])) 
                for _ in ["d_rate", "d_copy", "ntoktgt", "ntokpred"]
            ])
            d_lograte = torch.log(d_rate) # in ]-inf, 0], 0 better
            d_neglograte = -d_lograte # in [0, +inf[, 0 better
            loss = d_neglograte.sum()
        elif rate_fct == "rate_mse":
            pred_rate = 100 * (pred_copy / ntokpred)
            gold_rate = 100 * (gold_copy / ntoktgt)
            loss = (pred_rate - gold_rate) ** 2
            loss = loss.sum()
        else:
            raise ValueError()
        
        # mult by tgt tokens since its going to be normalized
        loss *= ntoktgt.sum()
        # loss = (((pred_copy - gold_copy)/ntoktgt) ** 2) 
        # loss = loss.sum()
        # print(loss)
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
       
