"""
Impostor Pair Generation
========================
Implements the Impostor Projection Method (Seidman, 2013).

During training, we inject "impostor pairs" (texts from different authors)
to force the Siamese network to learn a sharper decision boundary.

Corresponds to: Section 2.5 of the project document.
"""

import random
from typing import Dict, List, Optional, Tuple

from config import cfg


class ImpostorGenerator:
    """
    Generates impostor (negative) pairs from an author corpus.
    Creates balanced training data with genuine + impostor pairs.
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed if seed is not None else cfg.training.seed
        self.rng = random.Random(self.seed)

    def generate_pairs(
        self, author_texts: Dict[str, List[str]],
    ) -> List[Tuple[str, str, int]]:
        """
        Generate genuine + impostor pairs.

        Returns: List of (text_a, text_b, label) tuples
          label = 1 → same author   |   label = 0 → different author
        """
        pairs = []

        genuine_pairs = self._generate_genuine_pairs(author_texts)
        pairs.extend(genuine_pairs)

        num_impostor = int(len(genuine_pairs) * cfg.data.impostor_ratio)
        impostor_pairs = self._generate_impostor_pairs(author_texts, num_impostor)
        pairs.extend(impostor_pairs)

        self.rng.shuffle(pairs)
        return pairs

    def _generate_genuine_pairs(
        self, author_texts: Dict[str, List[str]],
    ) -> List[Tuple[str, str, int]]:
        """Same author, different texts. Label = 1."""
        pairs = []
        for author_id, texts in author_texts.items():
            if len(texts) < 2:
                continue
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    pairs.append((texts[i], texts[j], 1))
        return pairs

    def _generate_impostor_pairs(
        self, author_texts: Dict[str, List[str]], num_pairs: int,
    ) -> List[Tuple[str, str, int]]:
        """Cross-author pairs. Label = 0."""
        pairs = []
        authors = list(author_texts.keys())

        if len(authors) < 2:
            raise ValueError("Need at least 2 authors for impostor pairs")

        for _ in range(num_pairs):
            author_a, author_b = self.rng.sample(authors, 2)
            text_a = self.rng.choice(author_texts[author_a])
            text_b = self.rng.choice(author_texts[author_b])
            pairs.append((text_a, text_b, 0))

        return pairs

    def generate_adversarial_impostor_pairs(
        self,
        author_texts: Dict[str, List[str]],
        similarity_scores: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> List[Tuple[str, str, int]]:
        """
        Advanced: pick the *hardest* impostors based on style similarity.

        If similarity_scores are provided (from a previous training round),
        we preferentially sample impostors most similar to the target author.
        These "smart impostors" harden the decision boundary.
        """
        if similarity_scores is None:
            num_pairs = int(
                sum(len(t) for t in author_texts.values()) * cfg.data.impostor_ratio
            )
            return self._generate_impostor_pairs(author_texts, num_pairs)

        pairs = []
        authors = list(author_texts.keys())

        for target_author in authors:
            other_authors = [a for a in authors if a != target_author]
            other_authors.sort(
                key=lambda a: similarity_scores.get(
                    (target_author, a),
                    similarity_scores.get((a, target_author), 0.0)
                ),
                reverse=True,
            )

            hard_impostors = other_authors[:cfg.data.num_impostors_per_author]
            for imp_author in hard_impostors:
                if author_texts[target_author] and author_texts[imp_author]:
                    text_a = self.rng.choice(author_texts[target_author])
                    text_b = self.rng.choice(author_texts[imp_author])
                    pairs.append((text_a, text_b, 0))

        self.rng.shuffle(pairs)
        return pairs
