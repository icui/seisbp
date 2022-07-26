from __future__ import annotations
import typing as tp
import adios2, json
from io import BytesIO

import numpy as np
from obspy import Stream, Trace, Catalog, Inventory
from obspy.core.event import Event

if tp.TYPE_CHECKING:
    from adios2 import File # type: ignore
    from mpi4py.MPI import Intracomm


class SeisBP:
    """Seismic data saved in adios2 binary pack format."""
    # file open mode
    _mode: tp.Literal['r', 'w', 'a']

    # adios2 binary pack file
    _bp: File

    # size of the numpy arrays that awaits writting
    _nbytes: int = 0

    # maximum write size in MB before end_step
    _buffer_size: float = 1024.0

    # index of all events
    _events: tp.Set[str]

    # index of all stations
    _stations: tp.Set[str]

    # index of all traces
    _traces: tp.Dict[str, tp.Set[str]]

    # index of all auxiliary data
    _keys: tp.Set[str]

    # MPI communicator
    _comm: tp.Optional[Intracomm] = None

    # file closed
    _closed = False

    def __init__(self, name: str, mode: tp.Literal['r', 'w', 'a'], comm: Intracomm | bool = False):
        if comm == True:
            from mpi4py.MPI import COMM_WORLD
            comm = COMM_WORLD

        # open adios file
        self._bp = adios2.open(name, mode, comm) if comm else adios2.open(name, mode) # type: ignore
        self._mode = mode
        self._comm = comm or None

        # create indices in read mode
        if mode == 'r':
            self._events = set()
            self._stations = set()
            self._traces = {}
            self._keys = set()

            for key in self._bp.available_variables():
                if key.startswith('$'):
                    # auxiliary data
                    self._keys.add(key[1:])
                
                else:
                    # event, station or trace data
                    if key.endswith('#'):
                        # skip trace meta data because it always binds to a trace data
                        continue

                    ndots = key.split(':')[0].count('.')

                    if ndots == 0:
                        # event data
                        self._events.add(key)
                    
                    elif ndots == 1:
                        # station data
                        self._stations.add(key)
                    
                    elif ndots == 3:
                        # trace data
                        net, sta, loc, cha = key.split('.')

                        station = f'{net}.{sta}'
                        channel = f'{loc}.{cha}'

                        if station not in self._traces:
                            self._traces[station] = set()
                        
                        self._traces[station].add(channel)

    def __enter__(self):
        return self
    
    def __exit__(self, *_):
        self.close()
        return False
    
    @tp.overload
    def add(self, item: Stream | Catalog | Inventory, tag: str | None = None) -> tp.List[str]: ...

    @tp.overload
    def add(self, item: Trace | Event, tag: str | None = None) -> str: ...

    def add(self, item: Stream | Trace | Catalog | Event | Inventory, tag: str | None = None) -> str | tp.List[str]:
        """Add seismic data."""
        if self._mode not in ('w', 'a'):
            raise PermissionError('file not opened in write or append mode')
        
        if isinstance(item, (Stream, Catalog)):
            keys = []

            for it in item:
                keys.append(self.add(it, tag))
            
            return keys
        
        if isinstance(item, Event):
            return self._write_event(item, tag)
        
        if isinstance(item, Inventory):
            return self._write_station(item, tag)
        
        if isinstance(item, Trace):
            return self._write_trace(item, tag)
        
        raise TypeError(f'unsupported item {item}')
    
    def put(self, key: str, item: tp.Tuple[np.ndarray, dict] | dict | np.ndarray, tag: str | None = None):
        """Set auxiliary data."""
        key2 = key

        if ':' in key:
            raise KeyError('`:` is not allowed in data key')

        if tag:
            key2 += ':' + tag
        
        data: np.ndarray | None = None
        aux: dict | None = None

        if isinstance(item, tuple):
            data = item[0]
            aux = item[1]

            if not isinstance(item[0], np.ndarray) or not isinstance(item[1], dict):
                raise TypeError(f'unsupported item {item}')
        
        elif isinstance(item, np.ndarray):
            data = item
        
        elif isinstance(item, dict):
            aux = item
        
        else:
            raise TypeError(f'unsupported item {item}')
        
        if data is not None:
            self._write('$' + key2, data)
        
        if aux:
            self._write('$' + key2 + '#', np.frombuffer(json.dumps(aux).encode(), dtype=np.dtype('byte')))

        return key

    def events(self, tag: str | None = None) -> tp.List[str]:
        """List of event names."""
        return self._list(self._events, tag)

    def stations(self, tag: str | None = None) -> tp.List[str]:
        """List of station names with StationXML."""
        return self._list(self._stations, tag)
    
    def streams(self, tag: str | None = None) -> tp.List[str]:
        """List of station names with traces."""
        if self._mode != 'r':
            raise PermissionError('file not opened in read mode')

        stations = []

        for sta, chas in self._traces.items():
            for cha in chas:
                if tag:
                    if cha.endswith(':' + tag):
                        stations.append(sta)
                        continue
                
                else:
                    if ':' not in cha:
                        stations.append(sta)
                        continue
        
        return stations
    
    def traces(self, tag: str | None = None) -> tp.List[str]:
        """List of trace IDs."""
        if self._mode != 'r':
            raise PermissionError('file not opened in read mode')

        traces = []

        for sta, chas in self._traces.items():
            for cha in chas:
                if tag:
                    if cha.endswith(':' + tag):
                        traces.append(f'{sta}.{cha.split(":")[0]}')
                
                else:
                    if ':' not in cha:
                        traces.append(f'{sta}.{cha}')
        
        return traces
    
    def keys(self, tag: str | None = None) -> tp.List[str]:
        """List of auxiliary data keys."""
        return self._list(self._keys, tag)
    
    def event(self, event: str, tag: str | None = None) -> Event:
        if tag:
            event += ':' + tag
        
        return self._read_event(event)

    def station(self, station: str, tag: str | None = None) -> Inventory:
        if tag:
            station += ':' + tag
        
        return self._read_station(station)

    def stream(self, station: str, tag: str | None = None) -> Stream:
        traces = []

        for cha in self._traces[station]:
            if tag:
                if not cha.endswith(':' + tag):
                        continue
            
            else:
                if ':' in cha:
                    continue
            
            traces.append(self._read_trace(f'{station}.{cha}'))
        
        if len(traces) == 0:
            raise KeyError(f'{station} not found')
        
        return Stream(traces)

    def trace(self, station: str, cmp: str | None = None, tag: str | None = None) -> Trace:
        if cmp:
            for cha in self._traces[station]:
                if tag:
                    if not cha.endswith(':' + tag):
                        continue

                    cha = cha.split(':')[0]
                
                else:
                    if ':' in cha:
                        continue

                if len(cmp) == 1:
                    if cha[-1] != cmp:
                        continue
                
                elif cha != cmp:
                    continue

                if tag:
                    cha += ':' + tag

                return self._read_trace(f'{station}.{cha}')
            
            raise KeyError(f'trace {station}.{cmp} not found')
        
        return self.stream(station, tag)[0]
    
    def get(self, key: str, tag: str | None = None):
        """Get auxiliary data."""
        if tag:
            key += ':' + tag
        
        data = None
        pars = None

        if key in self._keys:
            data = self._bp.read(f'${key}')
        
        if f'{key}#' in self._keys:
            pars = json.loads(self._bp.read(f'${key}#').tobytes().decode())
        
        if data is not None and pars is not None:
            return data, pars
        
        if data is not None:
            return data
        
        if pars is not None:
            return pars
        
        raise KeyError(f'{key} not found')

    def close(self):
        """Close file."""
        if not self._closed:
            self._bp.close()
            self._closed = True
    
    def _list(self, target: tp.Set[str], tag: str | None) -> tp.List[str]:
        if self._mode != 'r':
            raise PermissionError('file not opened in read mode')

        keys = set()

        for key in target:
            if key.endswith('#'):
                # notation for auxiliary parameters
                key = key[:-1]

            if tag:
                if key.endswith(':' + tag):
                    keys.add(key.split(':')[0])
            
            else:
                if ':' not in key:
                    keys.add(key)
        
        return list(keys)
    
    def _write(self, key: str, data: np.ndarray):
        end_step = False
        self._nbytes += data.nbytes

        if self._nbytes >= self._buffer_size * 1024 ** 2:
            end_step=True
            self._nbytes = 0

        self._bp.write(key, data, count=data.shape, end_step=end_step)

    def _write_event(self, item: Event, tag: str | None) -> str:
        # event name
        key: str | None = None

        for d in item.event_descriptions:
            if d.type ==  'earthquake name':
                key = d.text
        
        if key is None:
            raise ValueError(f'{item} does not have a valid name')
        
        # event name with tag
        key2 = key

        if tag:
            key2 += ':' + tag

        with BytesIO() as b:
            item.write(b, format='quakeml')
            b.seek(0, 0)
            self._write(key2, np.frombuffer(b.read(), dtype=np.dtype('byte')))
        
        return key
    
    def _read_event(self, key: str) -> Event:
        from obspy import read_events

        with BytesIO(self._bp.read(key)) as b:
            return read_events(b, format='quakeml')[0]

    def _write_station(self, item: Inventory, tag: str | None) -> tp.List[str]:
        if len(item.networks) != 1 or len(item.networks[0].stations) != 1:
            keys = []

            for net in item.networks:
                for sta in net.stations:
                    keys += self._write_station(item.select(net.code, sta.code), tag)
            
            return keys
        
        key = key2 = f'{item.networks[0].code}.{item.networks[0].stations[0].code}'

        if tag:
            key2 += ':' + tag

        with BytesIO() as b:
            item.write(b, format='stationxml')
            b.seek(0, 0)
            self._write(key2, np.frombuffer(b.read(), dtype=np.dtype('byte')))

        return [key]
    
    def _read_station(self, key: str) -> Inventory:
        from obspy import read_inventory

        with BytesIO(self._bp.read(key)) as b:
            return read_inventory(b)

    def _write_trace(self, item: Trace, tag: str | None) -> str:
        from obspy.io.sac import SACTrace

        key = key2 = f'{item.stats.network}.{item.stats.station}.{item.stats.location}.{item.stats.channel}'

        if tag:
            key2 += ':' + tag

        with BytesIO() as b:
            SACTrace.from_obspy_trace(item).write(b, headonly=True)
            b.seek(0, 0)
            self._write(key2 + '#', np.frombuffer(b.read(), dtype=np.dtype('byte')))
            self._write(key2, item.data)

        return key
    
    def _read_trace(self, key: str):
        from obspy.io.sac import SACTrace

        with BytesIO(self._bp.read(key + '#')) as b:
            stats = SACTrace.read(b, headonly=True).to_obspy_trace().stats

        return Trace(self._bp.read(key), stats)
