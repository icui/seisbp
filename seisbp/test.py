from os import chdir

from obspy import read, read_inventory, read_events
from seisbp import SeisBP


def test():
    test_write()
    test_read()


def test_write():
    with SeisBP('test.bp', 'w') as bp:
        # write event data
        bp.add(read_events('C201107191935A'))
        bp.add(read_events('C201107191935A.tagged'), 'tagged')

        # write station data
        bp.add(read_inventory('AZ.FRD.xml'))
        bp.add(read_inventory('AZ.FRD.tagged.xml'), 'tagged')

        # write trace data
        tr = read('AZ.GRD.BHZ.sac')
        tr_tagged = read('AZ.GRD.BHZ.tagged.sac')

        bp.add(tr)
        bp.add(tr_tagged, 'tagged')

        # write auxiliary data
        bp.set('aux', tr[0].data)
        bp.set('aux2', {'tagged': False})
        bp.set('aux', (tr_tagged[0].data, {'tagged': True}), 'tagged')


def test_read():
    from seisbp import SeisBP

    with SeisBP('test.bp', 'r') as bp:
        # read indexing
        assert bp.events() == bp.events('tagged') == ['C201107191935A']
        assert bp.stations() == bp.stations('tagged') == ['AZ.FRD']
        assert bp.streams() == bp.streams('tagged') == ['AZ.FRD']
        assert bp.traces() == bp.traces('tagged') == ['AZ.FRD..BHZ']
        assert set(bp.keys()) == set(('aux', 'aux2'))
        assert bp.keys('tagged') == ['aux']

        # read event data
        assert bp.event('C201107191935A') == read_events('C201107191935A')[0]
        assert bp.event('C201107191935A', 'tagged') == read_events('C201107191935A.tagged')[0]

        # read station data
        assert bp.station('AZ.FRD') == read_inventory('AZ.FRD.xml')
        assert bp.station('AZ.FRD', 'tagged') == read_inventory('AZ.FRD.tagged.xml')

        # read trace data
        tr = read('AZ.GRD.BHZ.sac')
        tr_tagged = read('AZ.GRD.BHZ.tagged.sac')

        assert bp.trace('AZ.FRD') == bp.stream('AZ.FRD')[0]
        assert all(bp.trace('AZ.FRD').data == tr[0].data)
        assert all(bp.trace('AZ.FRD', None, 'tagged').data == tr_tagged[0].data)

        # read auxiliary data
        assert all(bp.get('aux') == tr[0].data)
        assert bp.get('aux2') == {'tagged': False}

        assert all(bp.get('aux', 'tagged')[0] == tr_tagged[0].data)
        assert bp.get('aux', 'tagged')[1] == {'tagged': True}

        print('test complete')


if __name__ == '__main__':
    chdir('seisbp/test_data')
    test()
