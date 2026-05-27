from __future__ import annotations

import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path


class NGramModel:
    """A count-based next-token model for the Markov baseline.

    order=2 is a bigram model, order=3 is a trigram model, etc.
    """

    def __init__(
        self,
        order: int = 4,
        bos_id: int | None = None,
        eos_id: int | None = None,
    ) -> None:
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.context_size = order - 1
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        self.unigram_counts: Counter[int] = Counter()

    def fit(self, sequences: list[list[int]]) -> "NGramModel":
        """Count token transitions in the training sequences."""

        for seq in sequences:
            tokens = list(seq)
            if self.bos_id is not None and self.context_size > 0:
                tokens = [self.bos_id] * self.context_size + tokens
            if self.eos_id is not None:
                tokens = tokens + [self.eos_id]

            self.unigram_counts.update(tokens)

            if self.context_size == 0:
                for token in tokens:
                    self.counts[()].update([token])
                continue

            for i in range(self.context_size, len(tokens)):
                context = tuple(tokens[i - self.context_size : i])
                token = tokens[i]
                self.counts[context].update([token])

        return self

    def _counter_for_context(self, context: tuple[int, ...]) -> Counter[int]:
        """Use exact context when possible, otherwise back off to shorter suffixes."""

        if self.context_size == 0:
            return self.counts[()]

        for start in range(len(context) + 1):
            candidate = tuple(context[start:])
            if candidate in self.counts:
                return self.counts[candidate]
        return self.unigram_counts

    @staticmethod
    def _sample_from_counter(
        counter: Counter[int],
        rng: random.Random,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> int:
        """Sample one token from a count table."""

        if not counter:
            raise ValueError("Cannot sample from an empty counter.")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        items = counter.most_common(top_k)
        tokens = [token for token, _ in items]
        counts = [count for _, count in items]

        if temperature != 1.0:
            weights = [count ** (1.0 / temperature) for count in counts]
        else:
            weights = counts

        return rng.choices(tokens, weights=weights, k=1)[0]

    def generate(
        self,
        max_length: int = 512,
        seed: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = 20,
    ) -> list[int]:
        """Generate a token-id sequence from the learned distribution."""

        rng = random.Random(seed)
        generated: list[int] = []
        context = [self.bos_id] * self.context_size if self.bos_id is not None else []

        for _ in range(max_length):
            context_tuple = tuple(context[-self.context_size :]) if self.context_size else ()
            counter = self._counter_for_context(context_tuple)
            token = self._sample_from_counter(counter, rng, temperature=temperature, top_k=top_k)
            if token == self.eos_id:
                break
            generated.append(token)
            context.append(token)

        return generated

    def perplexity(self, sequences: list[list[int]]) -> float:
        """Compute a simple add-one-smoothed perplexity for validation sequences."""

        import math

        vocab_size = max(1, len(self.unigram_counts))
        total_log_prob = 0.0
        total_tokens = 0

        for seq in sequences:
            tokens = list(seq)
            if self.bos_id is not None and self.context_size > 0:
                tokens = [self.bos_id] * self.context_size + tokens
            if self.eos_id is not None:
                tokens = tokens + [self.eos_id]

            for i in range(self.context_size, len(tokens)):
                context = tuple(tokens[i - self.context_size : i])
                token = tokens[i]
                counter = self._counter_for_context(context)
                context_total = sum(counter.values())
                prob = (counter[token] + 1.0) / (context_total + vocab_size)
                total_log_prob += math.log(prob)
                total_tokens += 1

        if total_tokens == 0:
            return float("inf")
        return math.exp(-total_log_prob / total_tokens)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "NGramModel":
        with Path(path).open("rb") as f:
            return pickle.load(f)

