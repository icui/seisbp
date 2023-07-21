from __future__ import annotations
from typing import TYPE_CHECKING, Literal, List, Tuple, Set, Dict, overload
import adios2, json
from io import BytesIO

import numpy as np
from obspy import Stream, Trace, Catalog, Inventory
from obspy.core.event import Event

if TYPE_CHECKING:
    from adios2 import File # type: ignore
    from mpi4py.MPI import Intracomm
    from obspy.core.trace import Stats


class SeisBP:
    """Seismic data saved in adios2 binary pack format."""
    # file open mode
    _mode: Literal['r', 'w', 'a']

    # adios2 binary pack file
    _bp: File

    # size of the numpy arrays that awaits writting
    _nbytes: int = 0

    # maximum write size in MB before end_step
    _buffer_size: float = 1024.0

    # dict of tag -> event names
    _events: Dict[str, Set[str]]

    # dict of tag -> station names
    _stations: Dict[str, Set[str]]

    # dict of tag -> trace stations -> trace location and channel -> trace start and end time
    _traces: Dict[str, Dict[str, Dict[str, Set[str]]]]

    # dict of tag -> auxiliary data keys
    _auxiliaries: Dict[str, Set[str]]

    # MPI communicator
    _comm: Intracomm | None = None

    # file closed
    _closed = False

    def __init__(self, name: str, mode: Literal['r', 'w', 'a'], comm: Intracomm | bool = False):
        if comm == True:
            from mpi4py.MPI import COMM_WORLD
            comm = COMM_WORLD

        # open adios file
        self._bp = adios2.open(name, mode, comm) if comm else adios2.open(name, mode) # type: ignore
        self._mode = mode
        self._comm = comm or None

        # index of all items in read mode
        if mode == 'r':
            # keys of event, station, trace and auxiliary data
            self._events = {}
            self._stations = {}
            self._traces = {}
            self._auxiliaries = {}

            for key in self._bp.available_variables():
                if '#' not in key:
                    tag = ''

                else:
                    key, tag = key.split('#')

                if key.startswith('$'):
                    # auxiliary data or parameters
                    if tag not in self._auxiliaries:
                        self._auxiliaries[tag] = set()

                    self._auxiliaries[tag].add(key[1:])

                else:
                    # event, station or trace data
                    ndots = key.count('.')

                    if ndots == 0:
                        # event data (e.g. C051200D)
                        if tag not in self._events:
                            self._events[tag] = set()

                        self._events[tag].add(key)

                    elif ndots == 1:
                        # station data (e.g. II.OBN)
                        if tag not in self._stations:
                            self._stations[tag] = set()

                        self._stations[tag].add(key)

                    elif ndots == 4:
                        # trace data (e.g. HT.LIT.S3.MXN.13111035479249_13126035479249)
                        net, sta, loc, cha, ts = key.split('.')

                        station = f'{net}.{sta}'
                        channel = f'{loc}.{cha}'

                        if tag not in self._traces:
                            self._traces[tag] = {}

                        if station not in self._traces:
                            self._traces[tag][station] = {}

                        if channel not in self._traces[tag][station]:
                            self._traces[tag][station][channel] = set()

                        self._traces[tag][station][channel].add(ts)

    def __enter__(self):
        return self
    
    def __exit__(self, *_):
        self.close()
        return False

    def add(self, item: Stream | Trace | Catalog | Event | Inventory, *, tag: str = '') -> List[str]:
        """Write seismic data."""
        self._write_mode()

        if isinstance(item, (Stream, Catalog)):
            keys = []

            for it in item: # type: ignore
                keys.append(self.add(it, tag=tag))

            return keys

        if isinstance(item, Event):
            return [self._write_event(item, tag)]

        if isinstance(item, Inventory):
            return self._write_station(item, tag)

        if isinstance(item, Trace):
            return [self._write_trace(item, tag)]

        raise TypeError(f'unsupported item {item}')

    def add_events(self):
        pass

    def add_stations(self):
        pass

    def add_traces(self):
        pass

    def add_auxiliary(self, key: str, item: Tuple[np.ndarray, dict] | dict | np.ndarray, *, tag: str = '') -> str:
        """Write auxiliary data and/or parameters."""
        self._write_mode()

        if '#' in key:
            raise KeyError('`#` is not allowed in auxiliary key')

        data: np.ndarray | None = None
        params: dict | None = None

        if isinstance(item, tuple):
            # item is (data, params)
            data = item[0]
            params = item[1]

            if not isinstance(item[0], np.ndarray) or not isinstance(item[1], dict):
                raise TypeError(f'unsupported item {item}')
        
        elif isinstance(item, np.ndarray):
            # item contains data only
            data = item
        
        elif isinstance(item, dict):
            # item contains parameters only
            params = item
        
        else:
            raise TypeError(f'unsupported item {item}')

        # write data and parameters
        self._write('$' + key, data if data is not None else np.array([]), tag)
        self._write_params('$' + key, params or {}, tag)

        return key

    def event_ids(self, *, tag: str = '') -> Set[str]:
        """Get names of events."""
        self._read_mode()
        return set(self._events.get(tag) or [])

    def station_ids(self, has_meta: bool | None = True, has_trace: bool | None = True, *, tag: str = '') -> Set[str]:
        """Get names of stations with StationXML and/or traces."""
        self._read_mode()

        meta_set = lambda: set(self._stations.get(tag) or [])
        trace_set = lambda: set((self._traces.get(tag) or {}).keys())

        if has_meta and has_trace:
            return meta_set().intersection(trace_set())

        if has_meta:
            if has_trace is False:
                return meta_set().difference(trace_set())

            return meta_set()
        
        if has_trace:
            if has_meta is False:
                return trace_set().difference(meta_set())

            return trace_set()
        
        return set()

    def trace_ids(self, station: str, filt: str | None = None, *, tag: str = '') -> Set[str]:
        """Get IDs all traces in a station or channel."""
        self._read_mode()

        traces = set()

        for cha, tss in ((self._traces.get(tag) or {}).get(station) or {}).items():
            if filt is not None:
                # skip traces without matching channel
                if len(filt) == 1:
                    # filt is component code
                    if cha[-1] != filt:
                        continue
                
                elif '.' in filt:
                    # filt is f'{location}.{channel}'
                    if cha != filt:
                        continue
                
                else:
                    # filt is channel code
                    if cha.split('.')[-1] != filt:
                        continue
            
            for ts in tss:
                traces.add(f'{station}.{cha}.{ts}')

        return traces

    def channels(self, station: str, *, tag: str = '') -> Set[str]:
        """Get channels of a station."""
        self._read_mode()
        return set(((self._traces.get(tag) or {}).get(station) or {}).keys())

    def components(self, station: str, *, tag: str = '') -> Set[str]:
        """Get components of a station."""
        return {cha[-1] for cha in self.channels(station, tag=tag)}

    def auxiliary_ids(self, *, tag: str = '') -> Set[str]:
        """Get auxiliary data keys."""
        self._read_mode()
        return set(self._auxiliaries.get(tag) or [])

    def trace_id(self, trace: Trace | Stats) -> str:
        """Get the ID of a trace."""
        stats = trace.stats if isinstance(trace, Trace) else trace
        channel_id = f'{stats.network}.{stats.station}.{stats.location}.{stats.channel}'

        # start and endtime in microseconds
        stats = trace.stats
        s = int(round(stats.starttime.ns / 1000))
        e = int(round(stats.endtime.ns / 1000))

        return  f'{channel_id}.{s}_{e}'

    def event_tags(self, event: str | None = None) -> Set[str]:
        """Get tag names of an event."""
        return self._tags(self._events, event)

    def station_tags(self, station: str | None = None):
        """Get StationXML tag names of a station."""
        return self._tags(self._stations, station)

    def trace_tags(self, station: str) -> Set[str]:
        """Get trace tag names of a station."""
        return self._tags(self._traces, station)

    def auxiliary_tags(self, key: str):
        """Get tag names of auxiliary data."""
        return self._tags(self._auxiliaries, key)

    def read_event(self, event: str, *, tag: str = '') -> Event:
        """Read an event."""
        from obspy import read_events

        with BytesIO(self._read(event, tag)) as b:
            return read_events(b, format='quakeml')[0]

    def read_station(self, station: str, *, tag: str = '') -> Inventory:
        """Read a station."""
        from obspy import read_inventory

        with BytesIO(self._read(station, tag)) as b:
            return read_inventory(b)

    def read_stream(self, station: str, filt: str | None = None, *, tag: str = '') -> Stream:
        """Get a stream of traces in a channel."""
        traces = []

        for trace_id in self.trace_ids(station, filt, tag=tag):
            traces.append(self.read_trace(trace_id, tag=tag))

        return Stream(traces)

    @overload
    def read_trace(self, trace_id: str, header_only: Literal[True] = True, *, tag: str = '') -> Stats: ...

    @overload
    def read_trace(self, trace_id: str | Stats, header_only: Literal[False] = False, *, tag: str = '') -> Trace: ...

    def read_trace(self, trace_id: str | Stats, header_only: bool = False, *, tag: str = '') -> Trace | Stats:
        """Read a trace from its ID."""
        from obspy import UTCDateTime
        from obspy.core.trace import Stats

        if isinstance(trace_id, Stats):
            trace_id = self.trace_id(trace_id)

        # dict containing starttime and sampling_rate
        stats_dict = self._read_params(trace_id, tag)
        stats_dict['starttime'] = UTCDateTime(float(stats_dict['starttime'] / 1.0e9))

        # dict containing station code
        net, sta, loc, cha, _ = trace_id.split('.')
        sta_dict = {'network': net, 'station': sta, 'location': loc, 'channel': cha}

        # trace stats
        stats = Stats(stats_dict | sta_dict)

        if header_only:
            return stats

        return Trace(self._read(trace_id, tag), stats)

    @overload
    def read_auxiliary(self, key: str, header_only: Literal[True] = True, *, tag: str = '') -> dict: ...

    @overload
    def read_auxiliary(self, key: str, header_only: Literal[False] = False, *, tag: str = '') -> Tuple[np.ndarray, dict]: ...

    def read_auxiliary(self, key: str, header_only: bool = False, *, tag: str = '') -> Tuple[np.ndarray, dict] | dict:
        """Read auxiliary data and parameters."""
        params = self._read_params('$' + key, tag)

        if header_only:
            return params

        return self._read('$' + key, tag), params

    def close(self):
        """Close file."""
        if not self._closed:
            self._bp.close()
            self._closed = True

    def _read_mode(self):
        if self._mode != 'r':
            raise PermissionError('file not opened in read mode')

    def _write_mode(self):
        if self._mode not in ('w', 'a'):
            raise PermissionError('file not opened in write mode')
    
    def _tags(self, target: Dict, item: str | None) -> Set[str]:
        self._read_mode()

        if item is None:
            # get all tags in the dataset
            return set(target.keys())

        # get tags for a specific item
        tags = set()

        for tag, items in target.items():
            if item in items:
                tags.add(tag)

        return tags

    def _read(self, key: str, tag: str):
        self._read_mode()

        if tag:
            key += '#' + tag

        return self._bp.read(key)

    def _read_params(self, key: str, tag: str) -> dict:
        self._read_mode()

        if tag:
            key += '#' + tag

        return json.loads((self._bp.read_attribute_string('params', key) or ['{}'])[0])

    def _write(self, key: str, data: np.ndarray, tag: str):
        end_step = False
        self._nbytes += data.nbytes

        if self._nbytes >= self._buffer_size * 1024 ** 2:
            # end step when cache space is used up
            end_step=True
            self._nbytes = 0

        if tag:
            key += '#' + tag

        self._bp.write(key, data, count=data.shape, end_step=end_step)

    def _write_params(self, key: str, params: dict, tag: str):
        if tag:
            key += '#' + tag

        self._bp.write_attribute('params', json.dumps(params), key)

    def _write_event(self, item: Event, tag: str) -> str:
        # get event name
        key: str | None = None

        for d in item.event_descriptions:
            if d.type ==  'earthquake name':
                key = d.text
        
        if key is None:
            raise ValueError(f'{item} does not have a valid name')
        
        # event name with tag
        with BytesIO() as b:
            item.write(b, format='quakeml')
            b.seek(0, 0)
            self._write(key, np.frombuffer(b.read(), dtype=np.dtype('byte')), tag)
        
        return key

    def _write_station(self, item: Inventory, tag: str) -> List[str]:
        if len(item.networks) != 1 or len(item.networks[0].stations) != 1:
            # Inventory with multiple stations
            keys = []

            for net in item.networks:
                for sta in net.stations:
                    keys += self._write_station(item.select(net.code, sta.code), tag)

            return keys

        key = f'{item.networks[0].code}.{item.networks[0].stations[0].code}'

        with BytesIO() as b:
            item.write(b, format='stationxml')
            b.seek(0, 0)
            self._write(key, np.frombuffer(b.read(), dtype=np.dtype('byte')), tag)

        return [key]

    def _write_trace(self, trace: Trace, tag: str = '') -> str:
        trace_id = self.trace_id(trace)
        stats = trace.stats
        self._write(trace_id, trace.data, tag)
        self._write_params(trace_id, {'starttime': stats.starttime.ns, 'sampling_rate': stats.sampling_rate, 'npts': stats.npts}, tag)

        return self.trace_id(trace)
