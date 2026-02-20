from upf.core import UPFConfig, PureUniversalCore

def test_smoke_step():
    core = PureUniversalCore(UPFConfig(), seed=1)
    S = core.step(drive=0.2, DOC=0.1)
    assert len(S) == core.cfg.levels
