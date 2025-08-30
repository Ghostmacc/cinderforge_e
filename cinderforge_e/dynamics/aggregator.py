# cinderforge_e/dynamics/aggregator.py
from dataclasses import dataclass

@dataclass
class Verdict:
    open_: bool     # gate intent before Schmitt
    score: float    # fused [0,1]
    reasons: dict   # {'plv':0.62,'sigma':0.97,'lambda':-0.03,'p':0.01}

class Aggregator:
    def __init__(self, cfg):
        self.open_th = cfg.gate.schmitt.open
        self.close_th = cfg.gate.schmitt.close
        self.state_open = False

    def fuse(self, plv, sigma, lam, p_rayleigh) -> Verdict:
        # Require both: decent PLV *and* significant non-uniform phase (Rayleigh)
        qual = 1.0 - min(1.0, p_rayleigh * 5.0)    # quick monotone squeeze
        base = 0.5*max(0.0, plv) + 0.5*max(0.0, qual)
        # Penalize chaotic expansion, prefer sigma≈1
        anti_chaos = 1.0 - max(0.0, lam)           # lam<=0 is good, lam>0 hurts
        crit = 1.0 - abs(sigma - 1.0)              # peak at 1.0
        score = 0.5*base + 0.25*anti_chaos + 0.25*crit

        if self.state_open and score < self.close_th:
            self.state_open = False
        elif not self.state_open and score >= self.open_th:
            self.state_open = True

        return Verdict(self.state_open, score, dict(plv=plv, sigma=sigma, lambda_=lam, p=p_rayleigh))
