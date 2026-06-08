import pytest

from bioreactor.feeds import make_feed
from bioreactor.params import BioreactorParams


def test_make_feed_batch_returns_zero_feed_without_outflow():
    params = BioreactorParams()
    feed, outflow = make_feed("batch", params)

    assert outflow is False
    assert feed(0.0, params.V_0) == pytest.approx(0.0)


def test_make_feed_exponential_feed_increases_without_outflow():
    params = BioreactorParams(F=0.1, alpha=0.2, V_max=30.0)
    feed, outflow = make_feed("fedbatch_exp", params)

    assert outflow is False
    assert feed(10.0, params.V_0) > feed(0.0, params.V_0)


def test_make_feed_chemostat_has_outflow():
    params = BioreactorParams(F=0.1)
    feed, outflow = make_feed("chemostat", params)

    assert outflow is True
    assert feed(0.0, params.V_0) == pytest.approx(params.F)


def test_make_feed_rejects_unknown_mode():
    with pytest.raises(NotImplementedError, match="Unknown operating mode"):
        make_feed("not_a_mode", BioreactorParams())
