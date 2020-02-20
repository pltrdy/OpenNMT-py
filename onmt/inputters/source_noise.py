import math
import torch


class NoiseBase(object):
    def __init__(self, prob, subword=True, subword_prefix="▁", mask_token="<mask>", skip_first_n=1, **kwargs):
        self.prob = prob
        self.subword = subword
        self.subword_prefix = subword_prefix
        self.mask_token = mask_token
        self.skip_first_n = skip_first_n
        assert self.subword, "source noise only implemented for subwords"

    def word_starts(self, source):
        """ Bool tensor from list of tokens
        """
        return torch.tensor([t.startswith(self.subword_prefix) for t in source]).long()

    def noise_batch(self, batch):
        for i, seq in enumerate(batch.src):
            batch.src[i] = self.noise_sequence(seq)
        return batch

    def noise_sequence(self, seq):
        skipped_tokens = seq[:self.skip_first_n]
        seq = seq[self.skip_first_n:]

        noisy_seq = self.noise_tokens(seq)
        final_seq = skipped_tokens + noisy_seq

        return final_seq

    def noise_tokens(self, tokens):
        raise NotImplementedError()


class MaskNoise(NoiseBase):
    def noise_tokens(self, tokens):
        subword_prefix = self.subword_prefix
        prob = self.prob

        # useless token, it'll be replaced with UNK
        mask_tok = "<MASK>"

        r = torch.rand([len(tokens)])
        mask = False
        masked = []
        for i, tok in enumerate(tokens):
            if tok.startswith(subword_prefix):
                if r[i].item() <= prob:
                    masked.append(mask_tok)
                    mask = True
                else:
                    masked.append(tok)
                    mask = False
            else:
                if mask:
                    pass
                else:
                    masked.append(tok)
        return masked

class SenShufflingNoise(NoiseBase):
    def __init__(self, *args, **kwargs):
        super(SenShufflingNoise, self).__init__( *args, **kwargs)
        assert self.prob == 1.0, \
            "SenShuffling prob must be 1.0 (not %f)" % self.prob
        sentence_breaks = [".", "?", "!"]
        self.sentence_breaks = [self.subword_prefix + t for t in sentence_breaks]

    def split(self, tokens):
        """Return list of sentences from tokens
        """
        sentence_breaks = [".", "?", "!"]
        sentence_breaks = [self.subword_prefix + t for t in sentence_breaks]
        sentences = [[]]
        for tok in tokens:
            sentences[-1].append(tok)
            if tok in sentence_breaks:
                sentences.append([])
        if len(sentences[-1]) == 0:
            sentences.pop()
        return sentences

    def noise_tokens(self, tokens):
        sentences = self.split(tokens)
        shuffled_sentences = [
            sentences[i] for i in torch.randperm(len(sentences))
        ]
        shuffled_tokens = sum(shuffled_sentences, [])
        return shuffled_tokens

class InfillingNoise(NoiseBase):
    def __init__(self, *args, infilling_poisson_lambda=3.0, **kwargs):
        super(InfillingNoise, self).__init__(*args, **kwargs)
        self.poisson_lambda = infilling_poisson_lambda
        self.mask_span_distribution = self._make_poisson(self.poisson_lambda)
        self.mask_idx = 0
        
        # -1: keep everything (i.e. 1 mask per token)
        #  0: replace everything (i.e. no mask)
        #  1: 1 mask per span 
        self.replace_length = 1

    def _make_poisson(self, poisson_lambda):
        # fairseq/data/denoising_dataset.py
        _lambda = poisson_lambda

        lambda_to_the_k = 1
        e_to_the_minus_lambda = math.exp(-_lambda)
        k_factorial = 1
        ps = [] 
        for k in range(0, 128):
            ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
            lambda_to_the_k *= _lambda
            k_factorial *= (k + 1) 
            if ps[-1] < 0.0000001:
                break
        ps = torch.FloatTensor(ps)
        return torch.distributions.Categorical(ps)

    def noise_tokens(self, tokens):
        # we use indices as numerical representation of the source
        # using 0 as the mask value
        ids = self.text_infilling(tokens)
        noisy_tokens = [
            tokens[i-1] if i > 0 else self.mask_token
            for i in ids
        ]
        return noisy_tokens

    def text_infilling(self, tokens):
        """Text infilling
           based on fairseqdata/denoising_dataset.py
           commit 226c1f

        Args:
            tokens(list)
        Returns:
            source(LongTensor): source, with `self.mask_idx` as mask token
        """
        source = torch.tensor([i+1 for i in range(len(tokens))])
        is_word_start = self.word_starts(tokens)
        
        # we manually add this hypothesis since it's required for the rest
        # of the function and kindof make sense
        is_word_start[-1] = 0

        p = self.prob
        num_to_mask = int(math.ceil(is_word_start.float().sum() * p))
        num_inserts = 0
        if num_to_mask == 0:
            return source

        if self.mask_span_distribution is not None:
            lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))

            # Make sure we have enough to mask
            cum_length = torch.cumsum(lengths, 0)
            while cum_length[-1] < num_to_mask:
                lengths = torch.cat([lengths, self.mask_span_distribution.sample(sample_shape=(num_to_mask,))], dim=0)
                cum_length = torch.cumsum(lengths, 0)

            # Trim to masking budget
            i = 0
            while cum_length[i] < num_to_mask:
                i += 1
            lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
            num_to_mask = i + 1
            lengths = lengths[:num_to_mask]

            # Handle 0-length mask (inserts) separately
            lengths = lengths[lengths > 0]
            num_inserts = num_to_mask - lengths.size(0)
            num_to_mask -= num_inserts
            if num_to_mask == 0:
                return self.add_insertion_noise(source, num_inserts / source.size(0))
            assert (lengths > 0).all()
        else:
            raise ValueError("Not supposed to be there")
            lengths = torch.ones((num_to_mask,)).long()
        assert is_word_start[-1] == 0
        word_starts = is_word_start.nonzero()
        indices = word_starts[torch.randperm(word_starts.size(0))[:num_to_mask]].squeeze(1)
        
        # random ratio disabled
        # mask_random = torch.FloatTensor(num_to_mask).uniform_() < self.random_ratio

        source_length = source.size(0)
        assert source_length - 1 not in indices
        to_keep = torch.ones(source_length, dtype=torch.bool)
        

        is_word_start[-1] = 10e4 # acts as a long length, so spans don't go over the end of doc
        if self.replace_length == 0:
            to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            source[indices] = self.mask_idx
            # random ratio disabled
            # source[indices[mask_random]] = torch.randint(1, len(self.vocab), size=(mask_random.sum(),))

        # if self.mask_span_distribution is not None:
        assert len(lengths.size()) == 1
        assert lengths.size() == indices.size()
        lengths -= 1
        while indices.size(0) > 0:
            assert lengths.size() == indices.size()
            lengths -= is_word_start[indices + 1].long()
            uncompleted = lengths >= 0
            indices = indices[uncompleted] + 1
            # mask_random = mask_random[uncompleted]
            lengths = lengths[uncompleted]
            if self.replace_length != -1:
                # delete token
                to_keep[indices] = 0
            else:
                # keep index, but replace it with [MASK]
                source[indices] = self.mask_idx
                # random ratio disabled
                # source[indices[mask_random]] = torch.randint(1, len(self.vocab), size=(mask_random.sum(),))
        # else:
        #     # A bit faster when all lengths are 1
        #     while indices.size(0) > 0:
        #         uncompleted = is_word_start[indices + 1] == 0
        #         indices = indices[uncompleted] + 1
        #         mask_random = mask_random[uncompleted]
        #         if self.replace_length != -1:
        #             # delete token
        #             to_keep[indices] = 0
        #         else:
        #             # keep index, but replace it with [MASK]
        #             source[indices] = self.mask_idx
        #             source[indices[mask_random]] = torch.randint(1, len(self.vocab), size=(mask_random.sum(),))

        #         assert source_length - 1 not in indices

        source = source[to_keep]

        if num_inserts > 0:
            source = self.add_insertion_noise(source, num_inserts / source.size(0))

        return source


    def add_insertion_noise(self, tokens, p):
        if p == 0.0:
            return tokens

        num_tokens = len(tokens)
        n = int(math.ceil(num_tokens * p))

        noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
        noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
        noise_mask[noise_indices] = 1
        result = torch.LongTensor(n + len(tokens)).fill_(-1)

        # random ratio disabled
        # num_random = int(math.ceil(n * self.random_ratio))
        num_random = 0
        result[noise_indices[num_random:]] = self.mask_idx
        # result[noise_indices[:num_random]] = torch.randint(low=1, high=len(self.vocab), size=(num_random,))

        result[~noise_mask] = tokens

        assert (result >= 0).all()
        return result

class MultiNoise(NoiseBase): 
    NOISES = {
        "sen_shuffling": SenShufflingNoise,
        "infilling": InfillingNoise,
        "mask": MaskNoise
    }

    def __init__(self, noises=[], probs=[], **kwargs):
        assert len(noises) == len(probs)
        super(MultiNoise, self).__init__(probs, **kwargs)

        self.noises = []
        for i, n in enumerate(noises):
            cls = MultiNoise.NOISES.get(n)
            if n is None:
                raise ValueError("Unknown noise function '%s'" % n)
            else:
                noise = cls(probs[i], **kwargs)
                self.noises.append(noise)

    def noise_tokens(self, tokens):
        for noise in self.noises:
            tokens = noise.noise_tokens(tokens)
        return tokens



