import os
import pytest
import thunderlab.configfile as cf
import thunderlab.tabledata as td


def test_config_file():
    cfg = cf.ConfigFile()
    td.add_write_table_config(cfg)

    wta = td.write_table_args(cfg)

    cfgfile = 'test.cfg'
    cfgdifffile = 'testdiff.cfg'

    # manipulate some values:
    cfg2 = cf.ConfigFile(cfg)
    cfg2.set('fileUnitStyle', 'header')
    cfg2.set('fileSections', '10')
    cfg2.set('fileCenterColumns', True)
    cfg3 = cf.ConfigFile(cfg2)

    assert 'fileUnitStyle' in cfg2, '__contains__'
    assert len(cfg2['fileUnitStyle']) == 4, '__getitem__'
    assert cfg2['fileUnitStyle'][0] == 'header', '__getitem__'

    with pytest.raises(IndexError):
        cfg2.set('xyz', 20)

    # write configurations to files:
    cfg.write(cfgfile, 'header', maxline=50)
    cfg2.write(cfgdifffile, diff_only=True)

    # test modified configuration:
    assert cfg != cfg2, 'cfg and cfg2 should differ'

    # read it in:
    cfg2.load(cfgfile)
    assert cfg == cfg2, 'cfg and cfg2 should be the same'

    # read manipulated values:
    cfg2.load(cfgdifffile)
    assert cfg2 == cfg3, 'cfg2 and cfg3 should be the same'

    # read it in:
    cfg3.load_files(cfgfile, 'data.dat', verbose=10)
    assert cfg == cfg3, 'cfg and cfg3 should be the same'

    # clean up:
    os.remove(cfgfile)
    os.remove(cfgdifffile)


def test_main():
    cf.main()

