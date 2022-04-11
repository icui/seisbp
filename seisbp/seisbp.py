from __future__ import annotations
import adios2
import typing as tp

import numpy as np
from obspy import Stream, Trace, Catalog, Inventory
from obspy.core.event import Event

from . import targets

if tp.TYPE_CHECKING:
    from adios2 import File
    from mpi4py.MPI import Intracomm


class SeisBP:
    """Seismic data saved in adios2 binary pack format."""
    # file open mode
    _mode: tp.Literal['r', 'w', 'a']

    # adios2 binary pack file
    _bp: File

    # index of all items
    _cache: dict

    # file closed
    _closed = False

    @property
    def events(self) -> tp.List[str]:
        """Event list."""
        if self._mode != 'r':
            raise PermissionError('file not opened in read mode')

        return self._cache['event']

    @property
    def stations(self) -> tp.List[str]:
        """Station list."""
        if self._mode != 'r':
            raise PermissionError('file not opened in read mode')

        return self._cache['station']
    
    @property 
    def traces(self) -> tp.List[str]:
        """Trace list."""
        if self._mode != 'r':
            raise PermissionError('file not opened in read mode')

        return self._cache['trace']
    
    @property
    def channels(self) -> tp.Dict[str, tp.List[str]]:
        """Dictionary of station names -> trace channels."""
        if '_channels' in self._cache:
            return self._cache['_channels']

        channels = {}

        for tr in self.traces:
            net, sta, loc, cha = tr.split('.')
            sta = f'{net}.{sta}'
            
            if sta not in channels:
                channels[sta] = []
            
            channels[sta].append(f'{loc}.{cha}')
        
        self._cache['_channels'] = channels
        
        return channels
    
    @property
    def keys(self) -> tp.List[str]:
        """List of saved arrays."""
        if self._mode != 'r':
            raise PermissionError('file not opened in read mode')

        return self._cache['#keys'] + self._cache['$keys']

    def __init__(self, name: str, mode: tp.Literal['r', 'w', 'a'], comm: Intracomm | bool = False):
        if comm == True:
            from mpi4py.MPI import COMM_WORLD
            comm = COMM_WORLD

        self._cache = { '#keys': [], '$keys': [] }
        self._bp = adios2.open(name, mode, comm) if comm else adios2.open(name, mode)
        self._mode = mode

        if mode == 'r':
            for name in _targets:
                self._cache[name] = []
            
            for key in self._bp.available_variables():
                if key.startswith('#'):
                    self._cache['#keys'].append(key[1:])
                
                elif key.startswith('$'):
                    self._cache['$keys'].append(key[1:])
                
                else:
                    for name, target in _targets.items():
                        if target.check_key(key):
                            self._cache[name].append(key)

    def __enter__(self):
        return self
    
    def __exit__(self, *_):
        self.close()
        return False

    def __del__(self):
        self.close()
    
    @tp.overload
    def write(self, item: Stream | Catalog) -> tp.List[str]: ...

    @tp.overload
    def write(self, item: Trace | Event | Inventory) -> str: ...

    def write(self, item: Stream | Trace | Catalog | Event | Inventory) -> str | tp.List[str]:
        """Add seismic data."""
        if self._mode not in ('w', 'a'):
            raise PermissionError('file not opened in write or append mode')

        for target in _targets.values():
            if target.check(item):
                try:
                    key = target.name(item)
                
                except:
                    raise ValueError(f'{item} does not have a valid name')

                data = target.write(item)
                
                if target.count and (count := target.count(item)):
                    if not isinstance(data, tuple):
                        raise ValueError(f'{data} should be a tuple of {count}')
                    
                    self._bp.write(key, np.array([count]))

                    for i in range(count):
                        self._bp.write(f'{key}:{i}', data[i], count=data[i].shape)

                else:
                    if isinstance(data, tuple):
                        raise ValueError(f'{data} should have type numpy.ndarray')

                    self._bp.write(key, data, count=data.shape)

                return key
            
            elif target.check_group and target.check_group(item):
                return [tp.cast(str, self.write(i)) for i in item]
        
        raise ValueError(f'unsupported data type: {item}')
    
    def read(self, key: str) -> Trace | Event | Inventory:
        """Read seismic data."""
        if self._mode != 'r':
            raise PermissionError('file not opened in read mode')

        for target in _targets.values():
            if target.check_key(key):
                if target.count:
                    count = int(self._bp.read(key))

                    if not isinstance(count, int):
                        raise KeyError(f'{key} does not exist')

                    args = []

                    for i in range(count):
                        if len(data := self._bp.read(f'{key}:{i}')) == 0:
                            raise KeyError(f'{key} does not exist')

                        args.append(data)
                    
                    return target.read(*args)
                
                else:
                    if len(data := self._bp.read(key)) == 0:
                        raise KeyError(f'{key} does not exist')

                    return target.read(data)
        
        raise ValueError(f'unsupported key type: {key}')
    
    def stream(self, sta: str):
        """Get Stream of a station."""
        traces = []

        for cha in self.channels[sta]:
            traces.append(self.read(f'{sta}.{cha}'))
        
        return Stream(traces)
    
    def trace(self, sta: str, cmp: str):
        """Get Trace of a station with channel or component code."""
        for cha in self.channels[sta]:
            if (len(cmp) == 1 and cha.endswith(cmp)) or cha.endswith(f'.{cmp}'):
                return self.read(f'{sta}.{cha}')
    
    def put(self, key: str, val: np.ndarray | str):
        """Save a numpy array or string."""
        if isinstance(val, str):
            self._bp.write(f'${key}', val)

        else:
            self._bp.write(f'#{key}', val, count=val.shape)
    
    def get(self, key: str):
        """Get a numpy array or string."""
        if key in self._cache['$keys']:
            return self._bp.read(f'${key}').tostring().decode()

        elif key in self._cache['#keys']:
            return self._bp.read(f'#{key}')
        
        raise KeyError(f'{key} is not a valid key')

    def close(self):
        """Close file."""
        if not self._closed:
            self._bp.close()
            self._closed = True


class _Target(tp.Protocol):
    """Protocol of data that can be saved."""
    # Check if item belongs to this target
    check: tp.Callable[[tp.Any], bool]

    # Check if item is a group of this target
    check_group: tp.Callable[[tp.Any], tp.Iterable | None] | None

    # Check if key belongs to this target
    check_key: tp.Callable[[str], bool]

    # read target from byte stream
    read: tp.Callable

    # write target to byte stream
    write: tp.Callable[[tp.Any], np.ndarray | tp.Tuple[np.ndarray, ...]]

    # get the name of an item
    name: tp.Callable[[tp.Any], str]

    # number of entries
    count: tp.Callable[[tp.Any], int | None] | None


_targets: tp.Dict[str, _Target] = dict(zip(
    targets.__all__, [getattr(targets, s) for s in targets.__all__]
))
