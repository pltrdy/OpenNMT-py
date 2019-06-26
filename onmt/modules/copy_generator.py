import torch
import torch.nn as nn

from onmt.utils.misc import aeq
from onmt.utils.loss import NMTLossCompute


def collapse_copy_scores(scores, batch, tgt_vocab, src_vocabs=None,
                         batch_dim=1, batch_offset=None):
    """
    Given scores from an expanded dictionary
    corresponeding to a batch, sums together copies,
    with a dictionary word when it is ambiguous.
    """
    def assert_size(t, sizes):
        ts = list(t.size())
        assert ts == sizes, "[%s] != [%s]" % (", ".join([str(_) for _ in ts]), ", ".join([str(_) for _ in sizes]))
    verbose = False
    experimental = False
    if experimental:
        src = batch.src[0]
        # print("src ", src.size())
        src_len = src.size(0)
        offset = len(tgt_vocab)
        _src_map = batch.src_map.float().data.cuda()
        _scores = scores.data.clone()

        # print("src_map ", _src_map.size())
        _src = src.clone().data
        src_l, bs, c_vocab = _src_map.size()
        assert src_l == src_len

        # [bs x src_len], mask of src_idx being in tgt_vocab
        src_invoc_mask = (_src.lt(offset) * _src.gt(1)).float().view(bs, src_len)
        assert_size(src_invoc_mask, [bs, src_len])

        # assert_size(scores, [bs, src_len])

        # [bs x c_voc], mask of cvocab_idx related to invoc src token
        cvoc_invoc_mask = src_invoc_mask.unsqueeze(1) \
                                        .bmm(_src_map.transpose(0, 1)) \
                                        .squeeze(1) \
                                        .gt(0)
        __bs, c_voc_size = cvoc_invoc_mask.size()

        # [bs x src_len], copy scores of in_voc src tokens
        # orig: [bs x 1 x cvocab] @bmm [bs x cvocab x src_len] = [bs x 1 x src_len]
        # [bs x tgt_len x cvocab] @    [bs x cvocab x src_len] = [bs x tgt_len x src_len]
        sc_offset = _scores[:, :, offset:].transpose(0, 1)
        # print("scores offset unsq ", sc_offset.size())
        # print("scores ", _scores.size())
        # print("_src_map tt ", _src_map.transpose(0, 1).transpose(1, 2).size())
        src_copy_scores = sc_offset \
                                             .bmm(_src_map.transpose(0, 1)
                                                          .transpose(1, 2))
        # [bs x src_len], invoc src tokens, or 1 (=pad)
        # print("_src ", _src.size())
        # print("src_invoc_mask ", src_invoc_mask.size())
        src_token_invoc = _src.clone().squeeze(2).transpose(0,1)
        # print("src_token_invoc ", src_token_invoc.size())
        src_token_invoc.masked_fill_(1-src_invoc_mask.byte(), 1)

        if verbose:
            print("cvoc_invoc_mask", cvoc_invoc_mask.size(),
                  cvoc_invoc_mask[0])
            print("src_invoc_mask", src_invoc_mask.size(),
                  src_invoc_mask[0])
            print("src_token_invoc", src_token_invoc.size(),
                  src_token_invoc[0])
            print("src_copy_scores", src_copy_scores.size(),
                  src_copy_scores[0])
            print(_src_map.size())
            print("src", src.size(), src[0])
            print("tgt", target.size(), target[0])
            print(src_copy_scores.size())
            print(src_token_invoc.size())

        src_copy_scores = src_copy_scores.transpose(0, 1)
        src_token_invoc = src_token_invoc.unsqueeze(0).expand_as(src_copy_scores)
        # src_token_invoc = src_token_invoc.view(bs, -1)
        try:
            _scores.scatter_add_(
                2, src_token_invoc.long(), src_copy_scores)
        except Exception as e:
            print("_scores ", _scores.size())
            print("src_token_invoc.size ", src_token_invoc.size())
            print("src_copy_scores.size ", src_copy_scores.size())
            print("scores ", scores)
            print("src_token_invoc ", src_token_invoc)
            print("src_copy_scores ", src_copy_scores)
            raise e


        # print("_scores ", _scores.size())
        # print("cvoc_invoc_mask ", cvoc_invoc_mask.size())
        _scores[:, :, offset:] *= (1-cvoc_invoc_mask.float())
        _scores[:, :, 1] = 0

        _scores_data = _scores
        scores = _scores
    else:
        offset = len(tgt_vocab)
        for b in range(scores.size(batch_dim)):
            blank = []
            fill = []

            if src_vocabs is None:
                src_vocab = batch.src_ex_vocab[b]
            else:
                batch_id = batch_offset[b] if batch_offset is not None else b
                index = batch.indices.data[batch_id]
                src_vocab = src_vocabs[index]

            for i in range(1, len(src_vocab)):
                sw = src_vocab.itos[i]
                ti = tgt_vocab.stoi[sw]
                if ti != 0:
                    blank.append(offset + i)
                    fill.append(ti)
            if blank:
                blank = torch.Tensor(blank).type_as(batch.indices.data)
                fill = torch.Tensor(fill).type_as(batch.indices.data)
                score = scores[:, b] if batch_dim == 1 else scores[b]
                score.index_add_(1, fill, score.index_select(1, blank))
                score.index_fill_(1, blank, 1e-10)

    # print(scores.size())
    return scores


class CopyGenerator(nn.Module):
    """An implementation of pointer-generator networks
    :cite:`DBLP:journals/corr/SeeLM17`.

    These networks consider copying words
    directly from the source sequence.

    The copy generator is an extended version of the standard
    generator that computes three values.

    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of copying a word from
      the source
    * :math:`p_{copy}` the probility of copying a particular word.
      taken from the attention distribution directly.

    The model returns a distribution over the extend dictionary,
    computed as

    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`


    .. mermaid::

       graph BT
          A[input]
          S[src_map]
          B[softmax]
          BB[switch]
          C[attn]
          D[copy]
          O[output]
          A --> B
          A --> BB
          S --> D
          C --> D
          D --> O
          B --> O
          BB --> O


    Args:
       input_size (int): size of input representation
       output_size (int): size of output vocabulary
       pad_idx (int)
    """

    def __init__(self, input_size, output_size, pad_idx):
        super(CopyGenerator, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.linear_copy = nn.Linear(input_size, 1)
        self.pad_idx = pad_idx

    def forward(self, hidden, attn, src_map):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by copying
        source words.

        Args:
           hidden (FloatTensor): hidden outputs ``(batch x tlen, input_size)``
           attn (FloatTensor): attn for each ``(batch x tlen, input_size)``
           src_map (FloatTensor):
               A sparse indicator matrix mapping each source word to
               its index in the "extended" vocab containing.
               ``(src_len, batch, extra_words)``
        """

        # CHECKS
        batch_by_tlen, _ = hidden.size()
        batch_by_tlen_, slen = attn.size()
        slen_, batch, cvocab = src_map.size()
        aeq(batch_by_tlen, batch_by_tlen_)
        aeq(slen, slen_)

        # Original probabilities.
        logits = self.linear(hidden)
        logits[:, self.pad_idx] = -float('inf')
        prob = torch.softmax(logits, 1)

        # Probability of copying p(z=1) batch.
        p_copy = torch.sigmoid(self.linear_copy(hidden))
        # Probability of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(prob, 1 - p_copy)
        mul_attn = torch.mul(attn, p_copy)
        copy_prob = torch.bmm(
            mul_attn.view(-1, batch, slen).transpose(0, 1),
            src_map.transpose(0, 1)
        ).transpose(0, 1)
        copy_prob = copy_prob.contiguous().view(-1, cvocab)
        return torch.cat([out_prob, copy_prob], 1)


class CopyGeneratorLoss(nn.Module):
    """Copy generator criterion."""
    def __init__(self, vocab_size, force_copy, unk_index=0,
                 ignore_index=-100, eps=1e-20):
        super(CopyGeneratorLoss, self).__init__()
        self.force_copy = force_copy
        self.eps = eps
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.unk_index = unk_index

    def forward(self, scores, align, target):
        """
        Args:
            scores (FloatTensor): ``(batch_size*tgt_len)`` x dynamic vocab size
                whose sum along dim 1 is less than or equal to 1, i.e. cols
                softmaxed.
            align (LongTensor): ``(batch_size x tgt_len)``
            target (LongTensor): ``(batch_size x tgt_len)``
        """
        # probabilities assigned by the model to the gold targets
        vocab_probs = scores.gather(1, target.unsqueeze(1)).squeeze(1)

        # probability of tokens copied from source
        copy_ix = align.unsqueeze(1) + self.vocab_size
        copy_tok_probs = scores.gather(1, copy_ix).squeeze(1)
        # Set scores for unk to 0 and add eps
        copy_tok_probs[align == self.unk_index] = 0
        copy_tok_probs += self.eps  # to avoid -inf logs

        # find the indices in which you do not use the copy mechanism
        non_copy = align == self.unk_index
        if not self.force_copy:
            non_copy = non_copy | (target != self.unk_index)

        probs = torch.where(
            non_copy, copy_tok_probs + vocab_probs, copy_tok_probs
        )

        loss = -probs.log()  # just NLLLoss; can the module be incorporated?
        # Drop padding.
        loss[target == self.ignore_index] = 0
        return loss


class CopyGeneratorLossCompute(NMTLossCompute):
    """Copy Generator Loss Computation."""
    def __init__(self, criterion, generator, tgt_vocab, normalize_by_length,
                 lambda_coverage=0.0):
        super(CopyGeneratorLossCompute, self).__init__(
            criterion, generator, lambda_coverage=lambda_coverage)
        self.tgt_vocab = tgt_vocab
        self.normalize_by_length = normalize_by_length

    def _make_shard_state(self, batch, output, range_, attns):
        """See base class for args description."""
        if getattr(batch, "alignment", None) is None:
            raise AssertionError("using -copy_attn you need to pass in "
                                 "-dynamic_dict during preprocess stage.")

        shard_state = super(CopyGeneratorLossCompute, self)._make_shard_state(
            batch, output, range_, attns)

        shard_state.update({
            "copy_attn": attns.get("copy"),
            "align": batch.alignment[range_[0] + 1: range_[1]]
        })
        return shard_state

    def _compute_loss(self, batch, output, target, copy_attn, align,
                      std_attn=None, coverage_attn=None):
        """Compute the loss.

        The args must match :func:`self._make_shard_state()`.

        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            copy_attn: the copy attention value.
            align: the align info.
        """
        target = target.view(-1)
        align = align.view(-1)
        scores = self.generator(
            self._bottle(output), self._bottle(copy_attn), batch.src_map
        )
        loss = self.criterion(scores, align, target)

        if self.lambda_coverage != 0.0:
            coverage_loss = self._compute_coverage_loss(std_attn,
                                                        coverage_attn)
            loss += coverage_loss

        # this block does not depend on the loss value computed above
        # and is used only for stats
        
        # print("scores ", str(scores.size()))
        # print("target ", str(target.size()))
        # print("align ", str(align.size()))
        scores_data = collapse_copy_scores(
            self._unbottle(scores.clone(), batch.batch_size),
            batch, self.tgt_vocab, None)
        scores_data = self._bottle(scores_data)

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

        # this part looks like it belongs in CopyGeneratorLoss
        if self.normalize_by_length:
            # Compute Loss as NLL divided by seq length
            tgt_lens = batch.tgt[:, :, 0].ne(self.padding_idx).sum(0).float()
            # Compute Total Loss per sequence in batch
            loss = loss.view(-1, batch.batch_size).sum(0)
            # Divide by length of each sequence and sum
            loss = torch.div(loss, tgt_lens).sum()
        else:
            loss = loss.sum()

        return loss, stats
