from io import BytesIO
import numpy as np

from obspy import read_events, Catalog
from obspy.core.event import Event


def check(item) -> bool:
    return isinstance(item, Event)


def check_group(item):
    if isinstance(item, Catalog):
        return item


def check_key(key: str):
    return key.count('.') == 0


def read(data: bytes):
    with BytesIO(data) as b:
        return read_events(b)[0]


def write(item: Event):
    with BytesIO() as b:
        item.write(b, format='quakeml')
        b.seek(0, 0)
        return np.frombuffer(b.read(), dtype=np.dtype('byte'))


def name(item: Event):
    for d in item.event_descriptions:
        if d.type ==  'earthquake name':
            return d.text
    
    raise ValueError(f'{item} does not have a valid name')


count = None
