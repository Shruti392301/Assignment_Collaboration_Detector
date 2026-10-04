import re


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def jaccard_unigram(text_a, text_b):
    words_a = set(clean_text(text_a).split())
    words_b = set(clean_text(text_b).split())

    if not words_a and not words_b:
        return 1.0

    if not words_a or not words_b:
        return 0.0

    intersection = words_a.intersection(words_b)
    union = words_a.union(words_b)

    return len(intersection) / len(union)


def shared_word_count(text_a, text_b):
    words_a = set(clean_text(text_a).split())
    words_b = set(clean_text(text_b).split())

    return len(words_a.intersection(words_b))


def shared_phrase_count(text_a, text_b):
    words_a = clean_text(text_a).split()
    words_b = clean_text(text_b).split()

    phrases_a = set(zip(words_a, words_a[1:]))
    phrases_b = set(zip(words_b, words_b[1:]))

    return len(phrases_a.intersection(phrases_b))


def longest_copied_stretch(text_a, text_b):
    words_a = clean_text(text_a).split()
    words_b = clean_text(text_b).split()

    max_length = 0

    for i in range(len(words_a)):
        for j in range(len(words_b)):
            length = 0

            while (
                i + length < len(words_a)
                and j + length < len(words_b)
                and words_a[i + length] == words_b[j + length]
            ):
                length += 1

            max_length = max(max_length, length)

    return max_length


def get_lexical_features(text_a, text_b):
    return {
        "jaccard_unigram": jaccard_unigram(text_a, text_b),
        "shared_word_count": shared_word_count(text_a, text_b),
        "shared_phrase_count": shared_phrase_count(text_a, text_b),
        "longest_copied_stretch": longest_copied_stretch(text_a, text_b)
    }


if __name__ == "__main__":
    text_a = "Renewable energy reduces pollution and protects the environment."

    text_b = "Renewable energy helps reduce pollution and protect our environment."

    print("Clean text A:", clean_text(text_a))
    print("Clean text B:", clean_text(text_b))

    print("\nLexical Features:")

    features = get_lexical_features(text_a, text_b)

    for feature, value in features.items():
        print(f"{feature}: {value}")