from __future__ import annotations
import adios2
from typing import overload, List, Literal, Dict, TypedDict, Callable, Any, TYPE_CHECKING
from io import BytesIO

import numpy as np
from obspy import read, read_events, read_inventory, Stream, Trace, Catalog, Inventory
from obspy.core.event import Event

if TYPE_CHECKING:
    from adios2 import File
    from mpi4py.MPI import Intracomm


class SeisBP:
    """Seismic data saved in adios2 binary pack format."""
    # file open mode
    _mode: Literal['r', 'w', 'a']

    # adios2 binary pack file
    _bp: File
    
    # index of entries
    _idx = {}

    # file closed
    _closed = False

    @property
    def events(self) -> List[str]:
        """Event list."""
        return self._idx['_events'].split(',')[1:]

    @property
    def stations(self) -> List[str]:
        """Station list."""
        return self._idx['_stations'].split(',')[1:]
    
    @property 
    def traces(self) -> List[str]:
        """Trace list."""
        return self._idx['_traces'].split(',')[1:]
    
    @property
    def channels(self) -> Dict[str, str]:
        """Dictionary of station names -> trace channels"""
        channels = {}

        for tr in self.traces:
            net, sta, cha = tr.split('.')
            sta = f'{net}.{sta}'
            
            if sta not in channels:
                channels[sta] = []
            
            channels[sta].append(cha)
        
        return channels

    def __init__(self, name: str, mode: Literal['r', 'w', 'a'], comm: Intracomm | None = None):
        self._bp = adios2.open(name, mode) if comm is None else adios2.open(name, mode, comm)
        self._mode = mode

        for target in _targets:
            self._idx[target['idx']] = self._bp.read(target['idx']).tostring().decode() if mode in ('r', 'a') else ''

    def __enter__(self):
        return self
    
    def __exit__(self, *_):
        self.close()
        return False

    def __del__(self):
        self.close()
    
    @overload
    def write(self, item: Stream | Catalog) -> List[str]: ...

    @overload
    def write(self, item: Trace | Event | Inventory) -> str: ...

    def write(self, item: Stream | Trace | Catalog | Event | Inventory) -> str | List[str]:
        """Add seismic data."""
        if self._mode not in ('w', 'a'):
            raise PermissionError('file not opened in write or append mode')

        for target in _targets:
            if target['item'](item):
                try:
                    key = target['name'](item)
                
                except:
                    raise ValueError(f'{item} does not have a valid name')
            
                with BytesIO() as b:
                    target['write'](item, b)
                    b.seek(0, 0)
                    data = np.frombuffer(b.read(), dtype=np.dtype('byte'))
                    self._bp.write(key, data, count=data.shape)
                
                self._idx[target['idx']] += ',' + key

                return key
            
            elif target['group'] and target['group'](item):
                return [self.write(i) for i in item] # type: ignore
        
        raise ValueError(f'unsupported data type: {item}')
    
    def read(self, key: str) -> Trace | Event | Inventory:
        """Read seismic data."""
        if self._mode != 'r':
            raise PermissionError('file not opened in read mode')

        for target in _targets:
            if target['key'](key):
                data = self._bp.read(key)
                
                if len(data) == 0:
                    raise KeyError(f'{key} does not exist')

                with BytesIO(self._bp.read(key)) as b:
                    return target['read'](b)
        
        raise ValueError(f'unsupported key type: {key}')

    def close(self):
        """Close file."""
        if not self._closed:
            if self._mode in ('w', 'a'):
                for target in _targets:
                    self._bp.write(target['idx'], self._idx[target['idx']])

            self._bp.close()
            self._closed = True


class _Target(TypedDict):
    # entry to save index
    idx: str

    # Check if item belongs to this target
    item: Callable[[Any], bool]

    # Check if item is a group of this target
    group: Callable[[Any], bool] | None

    # Check if key belongs to this target
    key: Callable[[str], bool]

    # read target from byte stream
    read: Callable[[BytesIO], Any]

    # write target to byte stream
    write: Callable[[Any, BytesIO], None]

    # get the name of an item
    name: Callable[[Any], str]


_targets: List[_Target] = [
     {
        'idx': '_events',
        'item': lambda item: isinstance(item, Event),
        'group': lambda item: isinstance(item, Catalog),
        'key': lambda key: key.count('.') == 0,
        'read': lambda  b: read_events(b)[0],
        'write': lambda item, b: item.write(b, format='quakeml'),
        'name': lambda item: [d.text for d in item.event_descriptions if d.type == 'earthquake name'][0]
    },

    {
        'idx': '_stations',
        'item': lambda item: isinstance(item, Inventory),
        'group': None,
        'key': lambda key: key.count('.') == 1,
        'read': lambda  b: read_inventory(b),
        'write': lambda item, b: item.write(b, format='stationxml'),
        'name': lambda item: f'{item.networks[0].code}.{item.networks[0].stations[0].code}'
    },

    {
        'idx': '_traces',
        'item': lambda item: isinstance(item, Trace),
        'group': lambda item: isinstance(item, Stream),
        'key': lambda key: key.count('.') == 2,
        'read': lambda  b: read(b),
        'write': lambda item, b: item.write(b, format='sac'),
        'name': lambda item: f'{item.stats.network}.{item.stats.station}.{item.stats.channel}'
    }
]


__all__ = ['SeisBP']
