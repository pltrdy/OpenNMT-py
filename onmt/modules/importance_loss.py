import torch
import torch.nn as nn

from onmt.utils.misc import aeq 
# from onmt.utils.loss import NMTLossCompute
from onmt.modules.copy_generator import CopyGeneratorLossCompute, CopyGenerator, collapse_copy_scores

def tensor_in_range(t, ge=None, gt=None, le=None, lt=None):
    b = (t.eq(t))

    if ge is not None:
        b = b & t.ge(ge)
    if gt is not None:
        b = b & t.gt(gt)
    if le is not None:
        b = b & t.le(le)
    if lt is not None:
        b = b & t.lt(lt)
    return b

def assert_in_range(t, ge=None, gt=None, le=None, lt=None):
    b = tensor_in_range(t, ge, gt, le, lt)
    assert b.all(), (
            "Range error for tensor %s w/ %s <= %s < t < %s <= %s\n%s"
            % (str(t), str(ge), str(gt), str(lt), str(le), str(b)))

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


def cross_entropy(P, Q, name="noname", eps=1e-20):
    """
        P, Q, two distributions (normalized)
        P: [n x d]
        Q: [m x d]

    """
    PlogQ = P*(Q+eps).log()

    if not PlogQ.eq(PlogQ).all():
        print("P: ", torch.min(P), torch.max(P))
        print("Q: ", torch.min(Q), torch.max(Q))
        print("PlogQ: ", torch.min(PlogQ), torch.max(PlogQ))
        raise ValueError("NaN in cross_entropy %s" % str(name))
    # print("P: ", P.size())
    # print("Q: ", Q.size())
    # print("P*logQ: ", PlogQ)
    CE = -PlogQ.sum(dim=-1) 
    # print("CE: ", CE.size())
    return CE

def entropy(P, name="noname", eps=1e-20):
    return cross_entropy(P, P, eps=eps, name=name)


class ImportanceGenerator(CopyGenerator):
    def __init__(self, input_size, output_size, pad_idx):
        super(ImportanceGenerator, self).__init__(input_size, output_size, pad_idx)
        self.K = torch.nn.Parameter(torch.rand([input_size]))
        

class ImportanceLossCompute(CopyGeneratorLossCompute):
    IMPORTANCE_SUMMARY = ["pred", "state", "target"]
    def __init__(self, criterion, generator, tgt_vocab, normalize_by_length,
                 lambda_coverage=0.0, importance_lambda=0.5, importance_q=None,
                 importance_agg="sum",
                 importance_summary="pred",
                 importance_alpha=None,
                 importance_beta=None,
                 importance_gamma=None):
        super(ImportanceLossCompute, self).__init__(
            criterion, generator, tgt_vocab, normalize_by_length,
            lambda_coverage=lambda_coverage)
        print("importance_gamma: ",importance_gamma)
        self._embeddings = None

        if importance_summary not in ImportanceLossCompute.IMPORTANCE_SUMMARY:
            raise ValueError("Invalid importance_summary '%s', choices are %s"
                             % (importance_summary,
                                str(ImportanceLossCompute.IMPORTANCE_SUMMARY)))
        self.summary = importance_summary
        self.softmax = nn.Softmax(dim=-1)

        def softmax_agg(t, dim=0):
            t_norm = torch.softmax(t, dim=-1)
            return torch.sum(t_norm, dim=dim)

        self.importance_agg = importance_agg
        
        if importance_agg == "sum":
            agg = torch.sum
        elif importance_agg == "softmax":
            agg = softmax_agg
        else:
            raise ValueError("Unknown agg fct '%s'" % importance_agg)
        self.S_agg = agg 
        self.T_agg = agg 
        self.D_agg = agg 
        self.K_agg = agg 
           
        ###### v1 
        # a=2,b=0.5,g=0.10,l=0.5: is learning pretty much like l=0
        # good, 55.8929 val acc at 

        # acc: 2.99 2.95 2.99 3.08 3.13 3.44 3.41 
        # seems not to converge, nor diverge, max acc 4.55, 235 steps, ppl 10^25
        # acc: 2.96 2.91 2.91 2.93 2.93 3.21 3.15 
        # w/ lamba=1
        # conf = {"a": 2, "b": 0.5, "g": 0.10} 
        
        # w/ lambda=1
        # acc: 2.96 2.91 2.91 2.93
        # acc not rlly increasing in 390
        # conf = {"a": 1, "b": 1, "g": 1}


        ###### v2
        # after 3k steps acc=3.34, ppl=7.08
        # val 2700: acc 3.41894, ppl: 7.24065
        # lambda 1:
        # conf = {"a": 1, "b": 1, "g": 1}
        
        # lambda 0.9
        # acc improving
        # val acc: 20.735 at 600 step, ppl 9.49081
        conf = {"a": 1, "b": 1, "g": 1}

        # lambda=1, q=0.5
        # conf = {"a": 1, "b": 1, "g": 1}
        # val300: 3.54122 8.55397
        # val600: 3.51299 7.46816
        # => KO
        # same w/ agg=softmax,summary=state
        # ...

        # lambda=1, q=0.5
        # a 0.25 b 5.0
        # softmax state

        # relevance factor
        if importance_alpha is None:
            self.alpha = conf["a"]
        else:
            self.alpha = importance_alpha

        # informativeness factor
        if importance_beta is None:
            self.beta = conf["b"]
        else:
            self.beta = importance_beta

        # knowledge update factor (wrt importance = 1)
        if importance_gamma is None:
            self.gamma = conf["g"]
        else:
            self.gamma = importance_gamma

        # importance / ml loss balance
        # more => importance is more important than ml
        # self._lambda = 0.0010
        self._lambda = importance_lambda
        self.importance_q = importance_q

        self.v = 2
        print("Importance: a=%f, b=%f, g=%f, l=%f, v=%d, q=%s, agg=%s, summary=%s"
              % (self.alpha, self.beta, self.gamma, self._lambda, self.v, str(self.importance_q),
                 importance_agg, importance_summary))

    def embed(self, inp):
        if self._embeddings is None:
            raise ValueError("_embeddings should be set manually (e.g. in trainer)")
        else:
            return self._embeddings(inp)

    def _compute_loss(self, batch, src_embs, tgt_embs, output, target, copy_attn, align,
                      std_attn=None, coverage_attn=None):
        src, src_lengths = batch.src if isinstance(batch.src, tuple) \
            else (batch.src, None)

        verbose = False # True
        v = self.v
        def log(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)
        
        log("\n\n\n********************\nIMPORTANCE COMPUTE LOSS\n")
        target = target.view(-1)
        align = align.view(-1)
        scores = self.generator(
            self._bottle(output), self._bottle(copy_attn), batch.src_map
        )
        ml_loss = self.criterion(scores, align, target)

        if self.lambda_coverage != 0.0:
            raise ValueError("Not implemented")

        # this block does not depend on the loss value computed above
        # and is used only for stats
        scores_data = collapse_copy_scores(
            self._unbottle(scores.clone(), batch.batch_size),
            batch, self.tgt_vocab, None)
        
        tgt_len, bs, cvocab = scores_data.size()
        len_dim = 0

        log("scores_data(unbot): ", scores_data.size())
        scores_data = self._bottle(scores_data)

        eps = 1e-20
        if v == 3:
            # words as distrib, not using embeddings anymore
            # pred_scores = scores_data[:, :len(self.tgt_vocab)] + eps
            # 
            # if not pred_scores.eq(pred_scores).all():
            #     print("pred_scores: ", pred_scores)
            #     print("output: ", output)
            # log("pred_scores(pre-multinomial): ", pred_scores.size())
            # pred = torch.multinomial(pred_scores, 1).to(scores_data.device)


            # log("pred(bot): ", pred.size()) 
            # pred = pred.view(tgt_len, bs, 1)
            # log("pred(unbot): ", pred.size())

            # print(batch.__dict__.keys())
            pass
        else:
            if self.summary == "pred":
                pred_scores = scores_data[:, :len(self.tgt_vocab)] + eps
                log("min/max/sum scores: ", torch.min(pred_scores), torch.max(pred_scores), torch.sum(pred_scores))
                if not pred_scores.eq(pred_scores).all():
                    print("pred_scores: ", pred_scores)
                    print("output: ", output)
                log("pred_scores(pre-multinomial): ", pred_scores.size())
                pred = torch.multinomial(pred_scores, 1).to(scores_data.device)
                

                log("pred(bot): ", pred.size()) 
                pred = pred.view(tgt_len, bs, 1)
                log("pred(unbot): ", pred.size())

                S_emb = self.embed(pred)
            elif self.summary == "state":
                S_emb = output
            elif self.summary == "target":
                S_emb = tgt_embs

            D_emb = src_embs
            T_emb = tgt_embs
            # K_emb = torch.ones([T_emb.size(-1)]).to(T_emb.device) # self.generator.K
            K_emb = self.generator.K
           
            
            assert_size(S_emb, T_emb)
            # S = prediction embeddings
            # D = source embeddings
            # T = target embeddings
            # K = background knowledge

            log("std_attn: ", std_attn.size())
            log("S: ", S_emb.size())
            log("T: ", T_emb.size())
            log("D: ", D_emb.size())
            log("K: ", K_emb.size())
            log("std_attn: ", std_attn.size())
            
            
            nopad_mask = batch.tgt[:-1, :, 0].ne(self.padding_idx)
            tgt_lens = nopad_mask.sum(0).float()
            tgt_lens.detach().requires_grad_(False)

            # ==============================
            # Distributions
            if v == 2:
                # v2 mask Summary and Target
                try:
                    _nopad_mask = nopad_mask.unsqueeze(-1)
                    S_emb = S_emb * _nopad_mask.float()
                    T_emb = T_emb * _nopad_mask.float()
                except RuntimeError:
                    print("S_emb: ", S_emb.size())
                    print("T_emb: ", T_emb.size())
                    print("mask: ", nopad_mask.size())
                    raise

            S_dist = eps + self.softmax(self.S_agg(S_emb.double(), dim=len_dim))
            D_dist = eps + self.softmax(self.D_agg(D_emb.double(), dim=len_dim))
            T_dist = eps + self.softmax(self.T_agg(T_emb.double(), dim=len_dim))
            K_dist = eps + self.softmax(K_emb.double()).expand([bs, -1])
            

            log("S_dist: ", S_dist.size())
            log("D_dist: ", D_dist.size())
            log("T_dist: ", T_dist.size())
            log("K_dist: ", K_dist.size())

            # ==============================
            # Query
            # log("K_dist: ", K_dist) 
            # Q_dist = K_dist.clone().detach().requires_grad_(False)
            # Q_dist = K_dist
            # Q_dist -= T_dist
            Q_emb = K_emb.clone().detach().requires_grad_(False)
            
            if self.importance_q is not None:
                q = self.importance_q
                K_dist = self.softmax(K_emb.expand([bs, -1])).double()
                T_dist = self.softmax(self.T_agg(T_emb)).double()
                Q = K_dist - q * T_dist
                Q_dist = self.softmax(Q)
            else:
                Q_dist = eps + self.softmax(K_emb.expand([bs, -1])-self.T_agg(T_emb)).double()
           
            # ==============================
            # Red, Rel, Inf
            # Red(S) = -H(S)
            # Rel(S, D) = -CE(S, D)
            # Inf(S, K) = CE(S, K)
            red = -entropy(S_dist, "red") 
            rel = -cross_entropy(S_dist, D_dist, "rel")
            inf = cross_entropy(S_dist, Q_dist, "inf")
           


            if v == 1:
                """
                    Importance is in R
                """
                # importance is a score, higher is better
                tgt_lens.detach().requires_grad_(False)
                imp = -red + self.alpha * rel
                if self.beta is not None:
                    imp += self.beta * inf
                
                # update knowledge => minimize cross entropy D/K
                K_loss = cross_entropy(D_dist, K_dist)
                importance_loss = -imp.sum()

                gamma = self.gamma
                if gamma is not None:
                    _loss = importance_loss + self.gamma * K_loss
                else:
                    _loss = importance_loss
                _loss = ((_loss * tgt_lens)/bs).sum()

                
                # log("scores: ", scores)
                # log("ml_loss: ",ml_loss)
                ml_loss = ml_loss.sum()

                loss = self._lambda * _loss + (1-self._lambda) * ml_loss

                # loss = ml_loss - rel.sum()
                # loss = ml_loss - rel.sum() + red.sum()
                # loss = ml_loss - rel.sum() + red.sum() - inf.sum()
                # loss = ml_loss - inf.sum()

                log("red: ", red) 
                log("rel: ", rel) 
                log("inf: ", inf) 
                log("importance: ", imp)
                log("imp_loss: ", importance_loss)
                log("K loss: ",  K_loss)
                log("_loss: ", _loss)
                log("ml loss: ", ml_loss)
                log("loss: ", loss)
                # print(loss)
            if v == 2:
                """
                    All importance components in [0, 1] to converge to 0
                """
                tgt_lens = tgt_lens.double()
                n_tokens = tgt_lens.sum()
                red2 = 1 / (-red+1)
                rel2 = 1 - (1/ ((-self.alpha * rel) + 1))
                inf2 = 1 / ((self.beta * inf) +1)

                # try:
                #     assert_in_range(red2, ge=0, le=1)
                #     assert_in_range(rel2, ge=0, le=1)
                #     assert_in_range(inf2, ge=0, le=1)
                # except AssertionError:
                #     print("batch size: ", bs)
                #     print("red2: ", red2)
                #     print("red: ", red)
                #     print("rel2: ", rel2)
                #     print("rel: ", rel)
                #     print("inf2: ", inf2)
                #     print("inf: ", inf)
                #     print("S == D? ratio", (S_dist.eq(D_dist).sum()/n_tokens))
                #     print("S: ", S_dist)
                #     print("D: ", D_dist)
                #     raise

                red2 = (red2 * tgt_lens).sum().div(n_tokens)
                rel2 = (rel2 * tgt_lens).sum().div(n_tokens)
                inf2 = (inf2 * tgt_lens).sum().div(n_tokens)
                importance_loss = red2 + rel2 + inf2
                # try:
                #     assert_in_range(red2, ge=0, le=1)
                #     assert_in_range(rel2, ge=0, le=1)
                #     assert_in_range(inf2, ge=0, le=1)
                # except AssertionError:
                #     print("tgt_lens: ", tgt_lens)
                #     print("n_tokens: ", n_tokens)
                #     print("red2: ", red2)
                #     print("rel2: ", rel2)
                #     print("inf2: ", inf2)
                #     raise

                log("importance_loss: ", importance_loss)

                k_loss = 1 - (1/(1+cross_entropy(D_dist, K_dist)))
                _k_loss = k_loss
                k_loss = (k_loss * tgt_lens).sum().div(n_tokens)
                try:
                    assert 0 <= k_loss <= 1
                except AssertionError:
                    print("_kloss: ", _k_loss)
                    print("kloss: ", k_loss)
                    print("lengths: ", tgt_lens)
                    print("n_tokens: ", n_tokens)
                    raise
                log("k_loss: ", k_loss)

            full_importance_loss_per_tok = importance_loss + self.gamma * k_loss
            # not dividing by tokens, it's done later
            full_importance_loss = (full_importance_loss_per_tok * n_tokens).float()
            log("full_importance_loss: ", full_importance_loss)

            ml_loss = ml_loss.sum()
            log("ml_loss: ", ml_loss)

            loss = self._lambda * full_importance_loss + (1-self._lambda) * ml_loss
            log("loss: ", loss)
            
            ml_ratio = ml_loss / loss
            ml_ratio_norm = ml_ratio / (1-self._lambda)
            log("ml ratio: ", ml_ratio, " expected lambda ratio: ", (1-self._lambda), " it's %f%%" % (100*ml_ratio_norm))

        # this block does not depend on the loss value computed above
        # and is used only for stats
        # Correct target copy token instead of <unk>
        # tgt[i] = align[i] + len(tgt_vocab)
        # for i such that tgt[i] == 0 and align[i] != 0
        target_data = target.clone()
        unk = self.criterion.unk_index
        correct_mask = (target_data == unk) & (align != unk)
        offset_align = align[correct_mask] + len(self.tgt_vocab)
        target_data[correct_mask] += offset_align

        # Compute sum of perplexities for stats
        stats = self._stats(loss.sum().clone(), scores_data, target_data)

        if self.normalize_by_length:
            # just use normalization: tokens
            raise ValueError("Not implemented")
        else:
            # loss = loss.sum()
            pass
        return loss, stats
