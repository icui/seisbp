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
    from obspy.core.trace import Stats


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

    # set of event names
    _events: tp.Set[str]

    # set of station names
    _stations: tp.Set[str]

    # dict of trace stations -> trace components
    _traces: tp.Dict[str, tp.Set[str]]

    # set of auxiliary data keys
    _auxiliaries: tp.Set[str]

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

        # index of all entries in read mode
        if mode == 'r':
            # keys of event, station, trace and auxiliary data
            self._events = set()
            self._stations = set()
            self._traces = {}
            self._auxiliaries = set()

            for key in self._bp.available_variables():
                if key.startswith('$'):
                    # auxiliary data or parameters
                    self._auxiliaries.add(key[1:])
                
                else:
                    # event, station or trace data
                    if key.endswith('#'):
                        # skip trace header because it always binds to trace data
                        continue

                    ndots = key.split(':')[0].count('.')

                    if ndots == 0:
                        # event data (e.g. C051200D)
                        self._events.add(key)
                    
                    elif ndots == 1:
                        # station data (e.g. II.OBN)
                        self._stations.add(key)
                    
                    elif ndots == 3:
                        # trace data (e.g. IU.PET.S3.MXZ)
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
    def write(self, item: Stream | Catalog | Inventory, tag: str | None = None) -> tp.List[str]: ...

    @tp.overload
    def write(self, item: Trace | Event, tag: str | None = None) -> str: ...

    def write(self, item: Stream | Trace | Catalog | Event | Inventory, tag: str | None = None) -> str | tp.List[str]:
        """Write seismic auxiliary data."""
        if self._mode not in ('w', 'a'):
            raise PermissionError('file not opened in write or append mode')
        
        if isinstance(item, (Stream, Catalog)):
            keys = []

            for it in item:
                keys.append(self.write(it, tag))
            
            return keys
        
        if isinstance(item, Event):
            return self._write_event(item, tag)
        
        if isinstance(item, Inventory):
            return self._write_station(item, tag)
        
        if isinstance(item, Trace):
            return self._write_trace(item, tag)
        
        raise TypeError(f'unsupported item {item}')
    
    def write_auxiliary(self, key: str, item: tp.Tuple[np.ndarray, dict] | dict | np.ndarray, tag: str | None = None) -> str:
        """Write auxiliary data and/or parameters."""
        key_notag = key

        if ':' in key:
            raise KeyError('`:` is not allowed in data key')

        if tag:
            key += ':' + tag
        
        data: np.ndarray | None = None
        params: dict | None = None

        if isinstance(item, tuple):
            data = item[0]
            params = item[1]

            if not isinstance(item[0], np.ndarray) or not isinstance(item[1], dict):
                raise TypeError(f'unsupported item {item}')
        
        elif isinstance(item, np.ndarray):
            data = item
        
        elif isinstance(item, dict):
            params = item
        
        else:
            raise TypeError(f'unsupported item {item}')
        
        if data is not None:
            self._write('$' + key, data)
        
        if params is not None:
            self._write('$' + key + '#', np.frombuffer(json.dumps(params).encode(), dtype=np.dtype('byte')))

        return key_notag

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

        for sta in self._traces:
            traces += self.traces_of_station(sta, tag)
        
        return traces
    
    def traces_of_station(self, station: str, tag: str | None = None) -> tp.List[str]:
        """List of trace IDs in a station."""
        traces = []

        for cha in self._traces[station]:
            if tag:
                if cha.endswith(':' + tag):
                    traces.append(f'{station}.{cha.split(":")[0]}')
            
            else:
                if ':' not in cha:
                    traces.append(f'{station}.{cha}')
            
        return traces
    
    def auxiliaries(self, tag: str | None = None) -> tp.List[str]:
        """List of auxiliary data keys."""
        return self._list(self._auxiliaries, tag)

    def trace_id(self, station: str, cmp: str | None = None, tag: str | None = None) -> str:
        """Find the ID of a trace."""
        for cha in self._traces[station]:
            if tag:
                if not cha.endswith(':' + tag):
                    continue

            else:
                if ':' in cha:
                    continue

            if cmp is not None:
                # skip traces without matching component
                cha_notag = cha.split(':')[0]

                if len(cmp) == 1:
                    if cha_notag[-1] != cmp:
                        continue

                elif cha_notag != cmp:
                    continue

            return f'{station}.{cha}'

        raise KeyError(f'trace {station}.{cmp or ""} not found')

    def read_event(self, event: str, tag: str | None = None) -> Event:
        """Read an event."""
        from obspy import read_events

        if tag:
            event += ':' + tag

        with BytesIO(self._bp.read(event)) as b:
            return read_events(b, format='quakeml')[0]

    def read_station(self, station: str, tag: str | None = None) -> Inventory:
        """Read a station."""
        from obspy import read_inventory

        if tag:
            station += ':' + tag

        with BytesIO(self._bp.read(station)) as b:
            return read_inventory(b)

    def read_stream(self, station: str, tag: str | None = None) -> Stream:
        """Read a stream."""
        traces = []

        for key in self.traces_of_station(station):
            traces.append(self.read_trace(key, tag))
        
        if len(traces) == 0:
            raise KeyError(f'{station} not found')
        
        return Stream(traces)

    def read_trace(self, trace_id: str, tag: str | None = None) -> Trace:
        """Read a trace."""
        return Trace(self.read_trace_data(trace_id, tag), self.read_trace_header(trace_id, tag))

    def read_trace_data(self, trace_id: str, tag: str | None = None) -> np.ndarray:
        """Read a trace data."""
        if tag:
            trace_id += ':' + tag

        return self._bp.read(trace_id)

    def read_trace_header(self, trace_id: str, tag: str | None = None) -> Stats:
        """Read a trace header."""
        from obspy.io.sac import SACTrace

        if tag:
            trace_id += ':' + tag

        with BytesIO(self._bp.read(trace_id + '#')) as b:
            return SACTrace.read(b, headonly=True).to_obspy_trace().stats
    
    def read_auxiliary(self, key: str, tag: str | None = None) -> tp.Tuple[np.ndarray | None, dict | None]:
        """Read auxiliary data and parameters."""
        return self.read_auxiliary_data(key, tag), self.read_auxiliary_params(key, tag)
    
    def read_auxiliary_data(self, key: str, tag: str | None = None) -> np.ndarray | None:
        """Read auxiliary data."""
        if tag:
            key += ':' + tag

        if key in self._auxiliaries:
            # return data if it exists
            return self._bp.read(f'${key}')
        
        if f'{key}#' in self._auxiliaries:
            # return None if data does not exist, but parameter dict exists
            return None
        
        raise KeyError(f'{key} not found')
    
    def read_auxiliary_params(self, key: str, tag: str | None = None) -> dict | None:
        """Read auxiliary parameters."""
        if tag:
            key += ':' + tag
        
        if f'{key}#' in self._auxiliaries:
            # return parameter dict if it exists
            return json.loads(self._bp.read(f'${key}#').tobytes().decode())
        
        if key in self._auxiliaries:
            # return None if parameter dict does not exist, but data exists
            return None
        
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
                # notation for data parameters
                key = key[:-1]

            if tag is not None:
                # entries with tag
                if key.endswith(':' + tag):
                    keys.add(key.split(':')[0])
            
            else:
                # entries without tag
                if ':' not in key:
                    keys.add(key)
        
        return list(keys)
    
    def _write(self, key: str, data: np.ndarray):
        end_step = False
        self._nbytes += data.nbytes

        if self._nbytes >= self._buffer_size * 1024 ** 2:
            # end step when cache space is used up
            end_step=True
            self._nbytes = 0

        self._bp.write(key, data, count=data.shape, end_step=end_step)

    def _write_event(self, item: Event, tag: str | None) -> str:
        # get event name
        key: str | None = None

        for d in item.event_descriptions:
            if d.type ==  'earthquake name':
                key = d.text
        
        if key is None:
            raise ValueError(f'{item} does not have a valid name')
        
        # event name with tag
        key_notag = key

        if tag:
            key += ':' + tag

        with BytesIO() as b:
            item.write(b, format='quakeml')
            b.seek(0, 0)
            self._write(key, np.frombuffer(b.read(), dtype=np.dtype('byte')))
        
        return key_notag

    def _write_station(self, item: Inventory, tag: str | None) -> tp.List[str]:
        if len(item.networks) != 1 or len(item.networks[0].stations) != 1:
            # Inventory with multiple stations
            keys = []

            for net in item.networks:
                for sta in net.stations:
                    keys += self._write_station(item.select(net.code, sta.code), tag)
            
            return keys
        
        key = key_notag = f'{item.networks[0].code}.{item.networks[0].stations[0].code}'

        if tag:
            key += ':' + tag

        with BytesIO() as b:
            item.write(b, format='stationxml')
            b.seek(0, 0)
            self._write(key, np.frombuffer(b.read(), dtype=np.dtype('byte')))

        return [key_notag]

    def _write_trace(self, item: Trace, tag: str | None) -> str:
        from obspy.io.sac import SACTrace

        key = key_notag = f'{item.stats.network}.{item.stats.station}.{item.stats.location}.{item.stats.channel}'

        if tag:
            key += ':' + tag

        with BytesIO() as b:
            SACTrace.from_obspy_trace(item).write(b, headonly=True)
            b.seek(0, 0)
            self._write(key + '#', np.frombuffer(b.read(), dtype=np.dtype('byte')))
            self._write(key, item.data)

        return key_notag
