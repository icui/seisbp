from io import BytesIO
import numpy as np

from obspy import read as read_stream, Trace, Stream


def check(item):
    return isinstance(item, Trace)


def check_group(item):
    if isinstance(item, Stream):
        return item


def check_key(key: str):
    return key.count('.') == 3


def read(data: np.ndarray, header: bytes):
    with BytesIO(header) as b:
        tr = read_stream(b, format='sac')[0]
        tr.stats.npts = len(data)
        tr.data = data
        return tr


def write(item: Trace):
    with BytesIO() as b:
        tr = Trace(header=item.stats)
        tr.stats.npts = 0
        tr.write(b, format='sac')
        b.seek(0, 0)
        return item.data, np.frombuffer(b.read(), dtype=np.dtype('byte'))


def name(item):
    return f'{item.stats.network}.{item.stats.station}.{item.stats.location}.{item.stats.channel}'


def count(_):
    return 2
