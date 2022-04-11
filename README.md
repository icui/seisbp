# SeisBP

A simple library to read and write seismic data using [ADIOS2](https://adios2.readthedocs.io/en/latest/index.html) binary pack format.


### Usage
Write
```py
from mpi4py.MPI import COMM_WORLD as comm  # optional
from obspy import read, read_events, read_inventory

event = read_events('C051200C')
station = read_inventory('II.OBN.xml')
trace = read('II.OBN.MXZ.sac')

with SeisBP('samle.bp', 'w', comm) as bp:
   bp.write(event)
   bp.write(station)
   bp.write(trace)
   bp.put('array_data', trace.data)
```

Read
```py
from mpi4py.MPI import COMM_WORLD as comm  # optional

with SeisBP('samle.bp', 'r', comm) as bp:
   # read data directly
   event = bp.read('C051200C') # Event
   station = bp.read('II.OBN')  # Inventory
   trace = bp.read('II.OBN..MXZ')  # Trace
   data = bp.get('array_data')  # numpy.ndarray

   # get stream or trace directly
   stream = bp.stream('II.OBN')
   trace = bp.trace('II.OBN', 'E')

   # get list of items
   bp.events  # ['C051200C']
   bp.stations  # ['II.OBN']
   bp.traces  # ['II.OBN..MXZ']
   bp.channels  # {'II.OBN': ['.MXZ']}
   bp.keys  # ['array_data']
```


### API
Write
```py
@overload
def write(self, item: Stream | Catalog) -> List[str]: ...

@overload
def write(self, item: Trace | Event | Inventory) -> str: ...

def write(self, item: Stream | Trace | Catalog | Event | Inventory) -> str | List[str]: ...
```

Read
```py
def read(self, key: str) -> Trace | Event | Inventory: ...
```

Stream / Trace
```py
def stream(self, sta: str): ...
def trace(self, sta: str, c: str): ...
```
