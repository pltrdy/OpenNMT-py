#!/usr/bin/env python
from onmt.translate.beam_search import BeamSearch, GNMTGlobalScorer

from copy import deepcopy

import torch

BLOCKED_SCORE = -10e20

class GlobalScorerStub(object):
    alpha = 0
    beta = 0

    def __init__(self):
        self.length_penalty = lambda x, alpha: 1.
        self.cov_penalty = lambda cov, beta: torch.zeros(
            (1, cov.shape[-2]), device=cov.device, dtype=torch.float)
        self.has_cov_pen = False
        self.has_len_pen = False

    def update_global_state(self, beam):
        pass

    def score(self, beam, scores):
        return scores


def run(n_iterations=7, beam_sz=5, batch_sz=1, n_words=100, repeat_idx=47, ngram_repeat=-1,
        repeat_logprob=-0.2, no_repeat_logprob=-0.1,
        base_logprob=float("-inf"), verbose=True,
        ):
    """
        At each timestep `i`:
            - token `repeat_idx` get a logprob of `repeat_logprob`
            - token `repeat_idx+i` get `no_repeat_logprob`
            - other tokens get `base_logprob`

    """
    def log(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    device_init = torch.zeros(1, 1)
    
    beam = BeamSearch(
        beam_sz, batch_sz, 0, 1, 2, 4,
        GlobalScorerStub(), 0, 30,
        False, ngram_repeat, set(),
        False, 0.)
    beam.initialize(device_init, torch.randint(0, 30, (batch_sz,)))
    for i in range(n_iterations):
        # non-interesting beams are going to get dummy values
        word_logprobs = torch.full(
            (batch_sz * beam_sz, n_words), base_logprob)
        if i == 0:
            # on initial round, only predicted scores for beam 0
            # matter. Make two predictions. Top one will be repeated
            # in beam zero, second one will live on in beam 1.
            word_logprobs[0::beam_sz, repeat_idx] = repeat_logprob
            word_logprobs[0::beam_sz, repeat_idx +
                       i + 1] = no_repeat_logprob
        else:
            # predict the same thing in beam 0
            word_logprobs[0::beam_sz, repeat_idx] = repeat_logprob
            # continue pushing around what beam 1 predicts
            word_logprobs[0::beam_sz, repeat_idx + i + 1] = no_repeat_logprob
        attns = torch.randn(1, batch_sz * beam_sz, 0)
        beam.advance(word_logprobs, attns)
        if ngram_repeat > 0:
        # NOTE: IGNORE IT FOR NOW
            # if i < ngram_repeat:
            #     assertFalse(
            #         beam.topk_log_probs[0::beam_sz].eq(
            #             BLOCKED_SCORE).any())
            #     assertFalse(
            #         beam.topk_log_probs[1::beam_sz].eq(
            #             BLOCKED_SCORE).any())
            # elif i == ngram_repeat:
            #     assertFalse(
            #         beam.topk_log_probs[:, 0].eq(
            #             BLOCKED_SCORE).any())

            #     expected = torch.full([batch_sz, beam_sz], base_logprob)
            #     expected[:, 0] = (i+1) * no_repeat_logprob
            #     expected[:, 1] = BLOCKED_SCORE
            # else:
            #     expected = torch.full([batch_sz, beam_sz], base_logprob)
            #     expected[:, 0] = i * no_repeat_logprob
            #     expected[:, 1] = BLOCKED_SCORE
            pass
        # log("Iteration (%d): expected %s" % (i, str(expected)))
        log("Iteration (%d): logprobs %s" % (i, str(beam.topk_log_probs)))
        log("Iteration (%d): seq %s" % (i, str(beam.alive_seq)))
        log("Iteration (%d): indices %s" % (i, str(beam.topk_ids)))
        if ngram_repeat > 0:
            log("Iteration (%d): blocked %s" % (i, str(beam.forbidden_tokens)))
    return beam.topk_log_probs, beam.alive_seq



def main():
    def run_args(*args, **kwargs):
        print("\n=================")
        print("With parameters %s %s" % (str(args), str(kwargs)))
        lp, seq = run(*args, **kwargs)
        print(lp)
        print(seq)

    # w/ default params
    # repeat_logprob=-0.2, no_repeat_logprob=-0.1
    # therefore we get the best sequence by taking only 
    # no repeat tokens i.e. 
    # `[repeat_idx+1, repeat_idx+2, ..., repeat_idx+n_iterations]`
    # with a final logprob of `n_iterations * no_repeat_logprob = -0.7`
    run_args(verbose=True, base_logprob=float("-inf"))

    # # reproductible w/ other base_logprobs e.g.
    # run_args(verbose=False, base_logprob=-1e3)
    # 
    # run_args(verbose=True, beam_sz=1, base_logprob=-1e3)

    # run_args(verbose=True, no_repeat_logprob=-0.3, ngram_repeat=3)

if __name__ == "__main__":
    main()
