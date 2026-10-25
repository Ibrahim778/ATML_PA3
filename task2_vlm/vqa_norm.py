"""VQA answer normalisation (after the official VQA evaluation code: punctuation, digits,
articles, contractions) and the local leave-one-out soft accuracy."""
import re

_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itll": "it'll", "lets": "let's",
    "mightnt": "mightn't", "mightve": "might've", "mustnt": "mustn't", "mustve": "must've",
    "neednt": "needn't", "shant": "shan't", "shed": "she'd", "shes": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "thats": "that's", "thered": "there'd", "theres": "there's",
    "theyd": "they'd", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "wasnt": "wasn't",
    "wed": "we'd", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're",
    "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's",
    "whereve": "where've", "whod": "who'd", "wholl": "who'll", "whos": "who's", "whove": "who've",
    "whyll": "why'll", "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've",
    "wouldnt": "wouldn't", "yall": "y'all", "youd": "you'd", "youll": "you'll", "youre": "you're",
    "youve": "you've",
}
_NUMBERS = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
            "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
_ARTICLES = {"a", "an", "the"}
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")
_PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`",
          ",", "?", "!"]


def _punct(text):
    out = text
    for p in _PUNCT:
        if (p + " " in text or " " + p in text) or (re.search(_COMMA_STRIP, text) is not None):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    return _PERIOD_STRIP.sub("", out, re.UNICODE)


def _digits_articles(text):
    words = []
    for w in text.lower().split():
        w = _NUMBERS.get(w, w)
        if w not in _ARTICLES:
            words.append(_CONTRACTIONS.get(w, w))
    return " ".join(words)


def normalize_answer(ans) -> str:
    if ans is None:
        return ""
    a = str(ans).replace("\n", " ").replace("\t", " ").strip()
    return _digits_articles(_punct(a)).strip()


def soft_accuracy(pred: str, answers) -> float:
    """Local VQA soft accuracy: mean over the 10 leave-one-out subsets of min(1, matches/3)."""
    p = normalize_answer(pred)
    gts = [normalize_answer(a) for a in answers]
    accs = []
    for i in range(len(gts)):
        others = gts[:i] + gts[i + 1:]
        accs.append(min(1.0, sum(g == p for g in others) / 3.0))
    return sum(accs) / len(accs)
