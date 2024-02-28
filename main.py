import nltk
from nltk import word_tokenize

def EOS_disambiguer(period_chunk):
    before_word_shortenings = ["Dr", "Mr", "Mrs", "Ms", "Jr", "Sr", "Prof", "Dept", "Inc", "Corp", "Co",
                               "Ltd", "Gov"]
    after_word_shortenings = ["Ave.", "Blvd.", "Rd.", "Jan.", "Feb.", "Aug.", "Sept.", "Oct.", "Nov.", "Dec.", "i.e.",
                              "e.g.", "etc.", "vs.", "ft.", "in.", "lbs.", "mph", "kg", "min.", "sec."]
    feature_vector = [0, 0, 0]
    feature_weights = [2, -4, -4]
    likely_class = ["EOS", "Not_EOS"]
    tokenized_chunk = nltk.wordpunct_tokenize(period_chunk)
    period_position = tokenized_chunk.index(".")
    if tokenized_chunk[period_position - 1].islower():
        feature_vector[0] = 1
    else:
        feature_vector[0] = 0
    if tokenized_chunk[period_position - 1] in before_word_shortenings:
        feature_vector[1] = 1
    else:
        feature_vector[1] = 0
    if tokenized_chunk[period_position + 1] in after_word_shortenings and tokenized_chunk[period_position - 1].isupper():
        feature_vector[2] = 1
    else:
        feature_vector[2] = 0
    dot_product = sum(x * y for x, y in zip(feature_vector, feature_weights))
    z = dot_product + 0.33
    sigmoid = 1 / (1 + 2.718**(-z))
    complement = 1 - sigmoid
    if sigmoid > complement:
        return likely_class[0]
    if sigmoid < complement:
        return likely_class[1]


print(EOS_disambiguer("a man like Dr. Peterson"))

