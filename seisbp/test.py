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
        bp.add(read_events('C201107191935A.tag_a'), tag='tag_a')

        # write station data
        bp.add(read_inventory('AZ.FRD.xml'))
        bp.add(read_inventory('AZ.FRD.tag_c.xml'), tag='tag_c')

        # write trace data
        tr = read('AZ.GRD.BHZ.sac')
        tr_tag_b = read('AZ.GRD.BHZ.tag_b.sac')

        bp.add(tr)
        bp.add(tr_tag_b, tag='tag_b')

        # write auxiliary data
        bp.add_auxiliary('aux', tr[0].data)
        bp.add_auxiliary('aux2', {'param_a': 'a'})
        bp.add_auxiliary('aux3', (tr_tag_b[0].data, {'param_b': True}), tag='tag_d')


def test_read():
    from seisbp import SeisBP

    with SeisBP('test.bp', 'r') as bp:
        # read indexing
        assert bp.event_ids() == bp.event_ids(tag='tag_a') == {'C201107191935A'}
        assert bp.station_ids() == bp.station_ids(tag='tag_c') == {'AZ.FRD'}
        assert bp.trace_station_ids() == bp.trace_station_ids(tag='tag_b') == {'AZ.FRD'}
        assert bp.channels('AZ.FRD') == {'.BHZ'}
        assert bp.channels('AZ.FRD', tag='tag_b') == {'S3.BHZ'}
        assert bp.auxiliary_ids() == {'aux', 'aux2'}
        assert bp.auxiliary_ids(tag='tag_d') == {'aux3'}

        # read tags
        assert bp.event_tags('C201107191935A') == {'', 'tag_a'}
        assert bp.station_tags('AZ.FRD') == {'', 'tag_c'}
        assert bp.trace_tags('AZ.FRD') == {'', 'tag_b'}
        assert bp.auxiliary_tags('aux') == {''}
        assert bp.auxiliary_tags('aux3') == {'tag_d'}

        # read event data
        assert bp.event('C201107191935A') == read_events('C201107191935A')[0]
        assert bp.event('C201107191935A', tag='tag_a') == read_events('C201107191935A.tag_a')[0]

        # read station data
        assert bp.station('AZ.FRD') == read_inventory('AZ.FRD.xml')
        assert bp.station('AZ.FRD', tag='tag_c') == read_inventory('AZ.FRD.tag_c.xml')

        # read trace data
        tr = read('AZ.GRD.BHZ.sac')
        tr_tag_b = read('AZ.GRD.BHZ.tag_b.sac')
        trace_id = bp.trace_ids('AZ.FRD').pop()

        assert bp.components('AZ.FRD') == {'Z'}
        assert bp.stream('AZ.FRD', '.BHZ')[0].stats.endtime == bp.stream('AZ.FRD')[0].stats.endtime == bp.trace_header(trace_id).endtime
        assert all(bp.stream('AZ.FRD')[0].data == tr[0].data)
        assert all(bp.stream('AZ.FRD', tag='tag_b')[0].data == tr_tag_b[0].data)
        assert all(bp.stream('AZ.FRD', 'Z', tag='tag_b')[0].data == tr_tag_b[0].data)
        assert all(bp.stream('AZ.FRD', 'S3.BHZ', tag='tag_b')[0].data == tr_tag_b[0].data)
        assert all(bp.stream('AZ.FRD', 'BHZ', tag='tag_b')[0].data == tr_tag_b[0].data)

        # read auxiliary data
        assert all(bp.auxiliary('aux')[0] == tr[0].data)
        assert bp.auxiliary('aux2')[1] == {'param_a': 'a'}

        assert all(bp.auxiliary('aux3', tag='tag_d')[0] == tr_tag_b[0].data)
        assert bp.auxiliary('aux3', tag='tag_d')[1] == bp.auxiliary_header('aux3', tag='tag_d') == {'param_b': True}

        print('test complete')


if __name__ == '__main__':
    chdir('seisbp/test_data')
    test()
