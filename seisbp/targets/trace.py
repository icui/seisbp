from io import BytesIO
import numpy as np

from obspy.io.sac import SACTrace
from obspy import Trace, Stream


def check(item):
    return isinstance(item, Trace)


def check_group(item):
    if isinstance(item, Stream):
        return item


def check_key(key: str):
    return key.count('.') == 3 and ':' not in key


def read(data: np.ndarray, header: bytes):
    with BytesIO(header) as b:
        stats = SACTrace.read(b, headonly=True).to_obspy_trace().stats
        return Trace(data, stats)


def write(item: Trace):
    with BytesIO() as b:
        SACTrace.from_obspy_trace(item).write(b, headonly=True)
        b.seek(0, 0)
        return item.data, np.frombuffer(b.read(), dtype=np.dtype('byte'))


def name(item):
    return f'{item.stats.network}.{item.stats.station}.{item.stats.location}.{item.stats.channel}'


def count(_):
    return 2
