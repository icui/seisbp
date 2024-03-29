from os import chdir

from obspy import read, read_inventory, read_events
from seisbp import SeisBP


def test():
    test_write()
    test_read()


def test_write():
    with SeisBP('test.bp', 'w') as bp:
        # write event data
        bp.write(read_events('C201107191935A'))
        bp.write(read_events('C201107191935A.tag_a'), 'tag_a')

        # write station data
        bp.write(read_inventory('AZ.FRD.xml'))
        bp.write(read_inventory('AZ.FRD.tag_c.xml'), 'tag_c')

        # write trace data
        tr = read('AZ.GRD.BHZ.sac')
        tr_tag_b = read('AZ.GRD.BHZ.tag_b.sac')

        bp.write(tr)
        bp.write(tr_tag_b, 'tag_b')

        # write auxiliary data
        bp.write_auxiliary('aux', tr[0].data)
        bp.write_auxiliary('aux2', {'param_a': 'a'})
        bp.write_auxiliary('aux3', (tr_tag_b[0].data, {'param_b': True}), 'tag_d')


def test_read():
    from seisbp import SeisBP

    with SeisBP('test.bp', 'r') as bp:
        # read indexing
        assert bp.events() == bp.events('tag_a') == {'C201107191935A'}
        assert bp.stations() == bp.stations('tag_c') == {'AZ.FRD'}
        assert bp.stations_with_trace() == bp.stations_with_trace('tag_b') == {'AZ.FRD'}
        assert bp.channels('AZ.FRD') == {'.BHZ'}
        assert bp.channels('AZ.FRD', tag='tag_b') == {'S3.BHZ'}
        assert bp.auxiliaries() == {'aux', 'aux2'}
        assert bp.auxiliaries('tag_d') == {'aux3'}

        # read tags
        assert bp.event_tags('C201107191935A') == {'', 'tag_a'}
        assert bp.station_tags('AZ.FRD') == {'', 'tag_c'}
        assert bp.trace_tags('AZ.FRD') == {'', 'tag_b'}
        assert bp.auxiliary_tags('aux') == {''}
        assert bp.auxiliary_tags('aux3') == {'tag_d'}

        # read event data
        assert bp.read_event('C201107191935A') == read_events('C201107191935A')[0]
        assert bp.read_event('C201107191935A', 'tag_a') == read_events('C201107191935A.tag_a')[0]

        # read station data
        assert bp.read_station('AZ.FRD') == read_inventory('AZ.FRD.xml')
        assert bp.read_station('AZ.FRD', 'tag_c') == read_inventory('AZ.FRD.tag_c.xml')

        # read trace data
        tr = read('AZ.GRD.BHZ.sac')
        tr_tag_b = read('AZ.GRD.BHZ.tag_b.sac')
        trace_id = bp.traces('AZ.FRD').pop()

        assert bp.components('AZ.FRD') == {'Z'}
        assert bp.read_stream('AZ.FRD', '.BHZ')[0].stats.endtime == bp.read_stream('AZ.FRD')[0].stats.endtime == bp.read_trace(trace_id, True).endtime
        assert all(bp.read_stream('AZ.FRD')[0].data == tr[0].data)
        assert all(bp.read_stream('AZ.FRD', tag='tag_b')[0].data == tr_tag_b[0].data)
        assert all(bp.read_stream('AZ.FRD', 'Z', tag='tag_b')[0].data == tr_tag_b[0].data)
        assert all(bp.read_stream('AZ.FRD', 'S3.BHZ', tag='tag_b')[0].data == tr_tag_b[0].data)
        assert all(bp.read_stream('AZ.FRD', 'BHZ', 'tag_b')[0].data == tr_tag_b[0].data)

        # read auxiliary data
        assert all(bp.read_auxiliary('aux')[0] == tr[0].data)
        assert bp.read_auxiliary('aux2')[1] == {'param_a': 'a'}

        assert all(bp.read_auxiliary('aux3', tag='tag_d')[0] == tr_tag_b[0].data)
        assert bp.read_auxiliary('aux3', tag='tag_d')[1] == bp.read_auxiliary('aux3', True, tag='tag_d') == {'param_b': True}

        print('test complete')


if __name__ == '__main__':
    chdir('seisbp/test_data')
    test()
