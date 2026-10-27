"""Small scaling helpers shared by the reconstruction pipeline."""


def normalize_to_neg_one_to_one(values):
    return values * 2 - 1


def unnormalize_to_zero_to_one(values):
    return (values + 1) * 0.5
