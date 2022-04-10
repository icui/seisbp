from io import BytesIO
import numpy as np

from obspy import read_inventory, Inventory


def check(item):
    return isinstance(item, Inventory)


check_group = None


def check_key(key: str):
    return key.count('.') == 1


def read(data: bytes):
    with BytesIO(data) as b:
        return read_inventory(b)


def write(item: Inventory):
    with BytesIO() as b:
        item.write(b, format='stationxml')
        b.seek(0, 0)
        return np.frombuffer(b.read(), dtype=np.dtype('byte'))


def name(item: Inventory):
    return f'{item.networks[0].code}.{item.networks[0].stations[0].code}'


count = None
