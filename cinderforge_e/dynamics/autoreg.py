# cinderforge_e/dynamics/autoreg.py

class AutoReg:
    """Bounded PI auto-correct system for weight adaptation."""
    
    def __init__(self, cfg):
        self.kp, self.ki = cfg.auto.k_p, cfg.auto.k_i
        self.max_step = cfg.auto.max_step
        self.hyst = cfg.auto.hysteresis

        self.I = dict(plv=0.0, sigma=0.0, lambda_=0.0)  # integrators

    def update(self, meas, targets, w):
        """
        Update weights using bounded PI control.
        
        Args:
            meas: dict with current plv, sigma, lambda measurements
            targets: config object with target values
            w: weights object to update in-place
            
        Returns:
            tuple: (updated_weights, error_dict)
        """
        # Signed "errors" (positive means "increase weight")
        err = dict(
            plv   = max(0.0, targets.plv - meas['plv']) - self.hyst,
            sigma = abs(meas['sigma'] - targets.sigma) - self.hyst,
            lambda_=max(0.0, meas['lambda']) - self.hyst
        )
        
        for k in err:
            if err[k] < 0: 
                err[k] = 0.0
            self.I[k] += err[k]
            delta = self.kp*err[k] + self.ki*self.I[k]
            delta = max(-self.max_step, min(self.max_step, delta))
            
            # Update weights with bounds check
            attr_name = 'lambda_' if k == 'lambda_' else k
            current_val = getattr(w, attr_name)
            new_val = max(0.0, current_val + delta)
            setattr(w, attr_name, new_val)
            
        return w, err
