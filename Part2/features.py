from collections import ChainMap
from typing import Callable, Dict, Set
import math

import pandas as pd


class FeatureMap:
    name: str

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        pass

    @classmethod
    def prefix_with_name(self, d: Dict) -> Dict[str, float]:
        """just a handy shared util function"""
        return {f"{self.name}/{k}": v for k, v in d.items()}


class BagOfWords(FeatureMap):
    name = "bow"
    STOP_WORDS = set(pd.read_csv("stopwords.txt", header=None)[0])

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        # TODO: implement this! Expected # of lines: <5
        counts = {}
        words = text.lower().split()
        for word in words:
            if word in self.STOP_WORDS: continue
            if word in counts: continue
            counts[word] = 1.0
        return self.prefix_with_name(counts)


class SentenceLength(FeatureMap):
    name = "len"

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        """an example of custom feature that rewards long sentences"""
        if len(text.split()) < 10:
            k = "short"
            v = 1.0
        else:
            k = "long"
            v = 5.0
        ret = {k: v}
        return self.prefix_with_name(ret)


class ExclamationCount(FeatureMap):
    name = "exc_c"

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        """an example of custom feature that rewards long sentences"""
        return self.prefix_with_name({"count": text.count("!")})


class ExclamationPresence(FeatureMap):
    name = "exc_p"

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        """an example of custom feature that rewards long sentences"""
        return self.prefix_with_name({"present": int("!" in text)})


class LogWordCount(FeatureMap):
    name = "log_wc"

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        """an example of custom feature that rewards long sentences"""
        return self.prefix_with_name({"log_wc": math.log(len(text.split()))})



class PronounCount(FeatureMap):
    name = "pron_c"

    first_pronouns = set(["i", "me", "my", "myself"])
    second_pronouns = set(["you", "your", "yours", "yourself", "yourselves"])
    third_pronouns = set(["he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves"])
    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        """an example of custom feature that rewards long sentences"""
        counts = {}
        for pronoun in self.first_pronouns:
            counts["first"] = text.count(pronoun)
        for pronoun in self.second_pronouns:
            counts['second'] = text.count(pronoun)
        for pronoun in self.third_pronouns:
            counts['third'] = text.count(pronoun)
        return self.prefix_with_name(counts)

class PositiveWords(FeatureMap):
    name = "pos_c"
    POSITIVE_WORDS = ['amazing', 'awesome', 'best', 'fantastic', 'great', 'good', 'happy', 'love', 'wonderful']
    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        """an example of custom feature that rewards long sentences"""
        counts = 0
        for word in self.POSITIVE_WORDS:
            counts += text.count(word)
        return self.prefix_with_name({"count": counts})


class NegativeWords(FeatureMap):
    name = "neg_c"
    NEGATIVE_WORDS = ['bad', 'hate', 'terrible', 'worst', 'awful', 'terrible', 'horrible', 'dislike', 'disliked', 'dislikes']
    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        """an example of custom feature that rewards long sentences"""
        counts = 0
        for word in self.NEGATIVE_WORDS:
            counts += text.count(word)
        return self.prefix_with_name({"count": counts})


FEATURE_CLASSES_MAP = {c.name: c for c in [BagOfWords, SentenceLength, ExclamationCount, ExclamationPresence, PronounCount, LogWordCount, PositiveWords, NegativeWords]}


def make_featurize(
    feature_types: Set[str],
) -> Callable[[str], Dict[str, float]]:
    featurize_fns = [FEATURE_CLASSES_MAP[n].featurize for n in feature_types]

    def _featurize(text: str):
        f = ChainMap(*[fn(text) for fn in featurize_fns])
        return dict(f)

    return _featurize


__all__ = ["make_featurize"]

if __name__ == "__main__":
    text = "I love this movie"
    print(text)
    print(BagOfWords.featurize(text))
    featurize = make_featurize({"bow", "len"})
    print(featurize(text))
