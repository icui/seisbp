from io import BytesIO
import numpy as np
import pickle

from obspy import Trace, Stream


def check(item):
    return isinstance(item, Trace)


def check_group(item):
    if isinstance(item, Stream):
        return item


def check_key(key: str):
    return key.count('.') == 3


def read(data: np.ndarray, header: bytes):
    with BytesIO(header) as b:
        stats = pickle.load(b)
        return Trace(data, stats)


def write(item: Trace):
    with BytesIO() as b:
        pickle.dump(item.stats, b)
        b.seek(0, 0)
        return item.data, np.frombuffer(b.read(), dtype=np.dtype('byte'))


def name(item):
    return f'{item.stats.network}.{item.stats.station}.{item.stats.location}.{item.stats.channel}'


def count(_):
    return 2
