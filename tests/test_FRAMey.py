from FRAMey.core import load_test_data, FRAMe, check_test_FRAMe

#import FRAMey.core


def test_sampledata():
    assert 1==1
    df = load_test_data()
    f = FRAMe(df, info_columns=['Info'])

    check_test_FRAMe(f)
