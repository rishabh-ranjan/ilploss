import collections


def hparams(x):
    if isinstance(x, (bool, int, float, str)):
        return x
    elif isinstance(x, collections.abc.Mapping):
        return {k: hparams(v) for k, v in x.items()}
    elif isinstance(x, collections.abc.Iterable):
        return {str(i): hparams(xx) for i, xx in enumerate(x)}
    else:
        return getattr(x, "hparams", {})
