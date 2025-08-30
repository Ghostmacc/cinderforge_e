# cinderforge_e/dynamics/penalty.py

def dynamics_penalty(meas, targets, w, mode, warmup_left):
    """
    Compute dynamics penalty loss term that respects mode & warm-up.
    
    Args:
        meas: dict with plv, sigma, lambda
        targets: config object with plv, sigma, lambda targets
        w: weights object with plv, sigma, lambda attributes
        mode: string - one of "off", "observe", "gentle", "hard", "auto"
        warmup_left: int - remaining warmup steps
        
    Returns:
        tuple: (loss_value, error_dict)
    """
    # meas: dict with plv, sigma, lambda
    if mode in ("off", "observe") or warmup_left > 0:
        return 0.0, dict(plv=0, sigma=0, lambda_=0)

    e_plv   = max(0.0, targets.plv - meas['plv'])           # below target hurts
    e_sigma = abs(meas['sigma'] - targets.sigma)
    e_lam   = max(0.0, meas['lambda'])                      # >0 hurts

    if mode == "gentle":
        alpha = 0.5
    else:  # "hard" or "auto" (weights already adapted upstream)
        alpha = 1.0

    loss = alpha * (w.plv*e_plv + w.sigma*e_sigma + w.lambda_*e_lam)
    return loss, dict(plv=e_plv, sigma=e_sigma, lambda_=e_lam)
