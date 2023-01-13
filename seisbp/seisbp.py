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

    # dict of tag -> event names
    _events: tp.Dict[str, tp.Set[str]]

    # dict of tag -> station names
    _stations: tp.Dict[str, tp.Set[str]]

    # dict of tag -> trace stations -> trace location and channel -> trace start and end time
    _traces: tp.Dict[str, tp.Dict[str, tp.Dict[str, tp.Set[tp.Tuple[str, str]]]]]

    # dict of tag -> auxiliary data keys
    _auxiliaries: tp.Dict[str, tp.Set[str]]

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
                    k = key.split('#')
                    tag = k[-1]
                    key = '#'.join(k[:-1])

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
                    
                    else:
                        # trace data (e.g. HT.LIT.S3.MXN_1311103547.9249_1.6)
                        try:
                            tr, s, sr = key.split('_')
                            net, sta, loc, cha = tr.split('.')
                        
                        except:
                            continue

                        station = f'{net}.{sta}'
                        channel = f'{loc}.{cha}'

                        if tag not in self._traces:
                            self._traces[tag] = {}

                        if station not in self._traces:
                            self._traces[tag][station] = {}

                        if channel not in self._traces[tag][station]:
                            self._traces[tag][station][channel] = set()

                        self._traces[tag][station][channel].add((s, sr))

    def __enter__(self):
        return self
    
    def __exit__(self, *_):
        self.close()
        return False

    @tp.overload
    def write(self, item: Stream | Catalog | Inventory, tag: str = '') -> tp.List[str]: ...

    @tp.overload
    def write(self, item: Trace | Event, tag: str = '') -> str: ...

    def write(self, item: Stream | Trace | Catalog | Event | Inventory, tag: str = '') -> str | tp.List[str]:
        """Write seismic auxiliary data."""
        if self._mode not in ('w', 'a'):
            raise PermissionError('file not opened in write or append mode')

        if isinstance(item, (Stream, Catalog)):
            keys = []

            for it in item: # type: ignore
                keys.append(self.write(it, tag))

            return keys

        if isinstance(item, Event):
            return self._write_event(item, tag)

        if isinstance(item, Inventory):
            return self._write_station(item, tag)

        if isinstance(item, Trace):
            trace_id = self._trace_id(item)
            self._write(trace_id, item.data, tag)
            return trace_id

        raise TypeError(f'unsupported item {item}')

    def write_auxiliary(self, key: str, item: tp.Tuple[np.ndarray, dict] | dict | np.ndarray, tag: str = '') -> str:
        """Write auxiliary data and/or parameters."""
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

    def events(self, tag: str = '') -> tp.Set[str]:
        """Get names of events."""
        self._read_mode()
        return set(self._events.get(tag) or [])

    def stations(self, tag: str = '') -> tp.Set[str]:
        """Get names of stations with StationXML."""
        # stations with StationXML
        self._read_mode()
        return set(self._stations.get(tag) or [])

    def waveforms(self, tag: str = '') -> tp.Set[str]:
        """Get names of stations with traces."""
        self._read_mode()
        return set((self._traces.get(tag) or {}).keys())
    
    def auxiliaries(self, tag: str = '') -> tp.Set[str]:
        """Get auxiliary data keys."""
        self._read_mode()
        return set(self._auxiliaries.get(tag) or [])

    def traces(self, station: str, filt: str | None = None, tag: str = '') -> tp.Set[str]:
        """Get trace identifiers (location + channel) of a station."""
        self._read_mode()

        chas = set()

        for cha in (self._traces.get(tag) or {}).get(station) or {}:
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
            
            chas.add(cha)

        return chas

    def components(self, station: str, tag: str = '') -> tp.Set[str]:
        """Get components of a station."""
        return {cha[-1] for cha in self.traces(station, None, tag)}
    
    def event_tags(self, event: str | None = None) -> tp.Set[str]:
        """Get tag names of an event."""
        return self._tags(self._events, event)

    def station_tags(self, station: str | None = None):
        """Get StationXML tag names of a station."""
        return self._tags(self._stations, station)

    def waveform_tags(self, station: str) -> tp.Set[str]:
        """Get trace tag names of a station."""
        return self._tags(self._traces, station)

    def auxiliary_tags(self, key: str):
        """Get tag names of auxiliary data."""
        return self._tags(self._auxiliaries, key)

    def read_event(self, event: str, tag: str = '') -> Event:
        """Read an event."""
        from obspy import read_events

        with BytesIO(self._read(event, tag)) as b:
            return read_events(b, format='quakeml')[0]

    def read_station(self, station: str, tag: str = '') -> Inventory:
        """Read a station."""
        from obspy import read_inventory

        with BytesIO(self._read(station, tag)) as b:
            return read_inventory(b)

    def read_waveforms(self, station: str, tag: str = '') -> Stream:
        """Read a stream of a station."""
        traces: tp.List[Trace] = []

        for cha in self.traces(station, None, tag):
            for tr in self.read_traces(station, cha, tag):
                traces.append(tr)

        if len(traces) == 0:
            raise KeyError(f'{station} not found')

        return Stream(traces)

    def read_traces(self, station: str, filt: str | None = None, tag: str = '') -> Stream:
        """Read a stream of a station channel."""
        from obspy import UTCDateTime

        traces = []

        for cha in self.traces(station, filt, tag):
            for s, sr in self._traces[tag][station][cha]:
                stats = {'starttime': UTCDateTime(float(s)), 'sampling_rate': float(sr)}
                traces.append(Trace(self._read(f'{station}.{cha}_{s}_{sr}', tag), stats))

        return Stream(traces)

    def read_traces_data(self, station: str, cha: str | None = None, tag: str = '') -> tp.List[np.ndarray]:
        """Read the data of the first trace that matches the arguments."""
        data = []

        for cha in self.traces(station, cha, tag):
            for s, sr in self._traces[tag][station][cha]:
                data.append(self._read(f'{station}.{cha}_{s}_{sr}', tag))

        return data

    def read_traces_params(self, station: str, cha: str | None = None, tag: str = '') -> tp.List[dict]:
        """Read the data of the first trace that matches the arguments."""
        params = []

        for cha in self.traces(station, cha, tag):
            for s, sr in self._traces[tag][station][cha]:
                params.append(self._read_params(f'{station}.{cha}_{s}_{sr}', tag))

        return params

    def read_auxiliary(self, key: str, tag: str = '') -> tp.Tuple[np.ndarray | None, dict | None]:
        """Read auxiliary data and parameters."""
        return self.read_auxiliary_data(key, tag), self.read_auxiliary_params(key, tag)

    def read_auxiliary_data(self, key: str, tag: str = '') -> np.ndarray | None:
        """Read auxiliary data."""
        return self._read('$' + key, tag)

    def read_auxiliary_params(self, key: str, tag: str = '') -> dict | None:
        """Read auxiliary parameters."""
        return self._read_params('$' + key, tag)

    def close(self):
        """Close file."""
        if not self._closed:
            self._bp.close()
            self._closed = True

    def _read_mode(self):
        if self._mode != 'r':
            raise PermissionError('file not opened in read mode')
    
    def _tags(self, target: tp.Dict[str, tp.Any], item: str | None) -> tp.Set[str]:
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

    def _write_station(self, item: Inventory, tag: str) -> tp.List[str]:
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

    def _trace_id(self, trace: Trace) -> str:
        stats = trace.stats
        s = stats.starttime.timestamp
        sr = stats.sampling_rate
        return f'{stats.network}.{stats.station}.{stats.location}.{stats.channel}_{s}_{sr}'
