import torch
class QuadrotorRandomizer:
    def __init__(self,
                 mass:      float = 1.21,  
                 inertia:   float = (0.007, 0.007, 0.013), 
                 length:    float = 0.15, 
                 angle:     float = 45.0,
                 torque_c:  float = 0.012, 
                 tau_m:     float = 0.05, 
                 eta:       float = 1.0,
                 Cd:        float = (0.28, 0.35, 0.7),
                 rho:       float = 1.225,
                 p:         float = 0.2, 
                 device='cuda',
                 ):
        
        self.device = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')
        self.p = p
        
        self.mass_nom    = torch.as_tensor(mass,     dtype=torch.float32, device=self.device)  
        self.inertia_nom = torch.as_tensor(inertia,  dtype=torch.float32, device=self.device)
        length_nom       = torch.as_tensor(length,   dtype=torch.float32, device=self.device)
        self.angle_nom   = torch.as_tensor(angle,    dtype=torch.float32, device=self.device)  
        torque_const_nom = torch.as_tensor(torque_c, dtype=torch.float32, device=self.device)
        tau_m_nom        = torch.as_tensor(tau_m,    dtype=torch.float32, device=self.device) 
        eta_nom          = torch.as_tensor(eta,      dtype=torch.float32, device=self.device)
        self.Cd_nom      = torch.as_tensor(Cd,       dtype=torch.float32, device=self.device)
        self.rho_nom     = torch.as_tensor(rho,      dtype=torch.float32, device=self.device)
        
        self.length_nom         = self._expand(length_nom)  
        self.torque_const_nom   = self._expand(torque_const_nom)
        self.tau_m_nom          = self._expand(tau_m_nom)  
        self.eta_nom            = self._expand(eta_nom)

    def _expand(self, value: torch.Tensor, n: int = 4) -> torch.Tensor:
        if value.ndim == 0:
            value = value.expand(n)
        return value.clone()
                       
    def _uniform_multiplicative(self, nominal: torch.Tensor, n: int, randomize: bool, p: float) -> torch.Tensor:
            out = nominal.expand((n,) + tuple(nominal.shape)).clone()
            if randomize and p > 0:
                scale = 1.0 + (2.0 * torch.rand_like(out) - 1.0) * p   # U[1-p, 1+p]
                out = out * scale
            return out
    
    def _uniform_additive(self, nominal: torch.Tensor, n: int, randomize: bool, p: float) -> torch.Tensor:
        out = nominal.expand((n,) + tuple(nominal.shape)).clone()
        if randomize and p > 0:
            delta = (2.0 * torch.rand_like(out) - 1.0) * p             # U[-p, +p]
            out = out + delta
        return out

    def sample(self, n: int, randomize: bool = True):
        mass         = self._uniform_multiplicative(self.mass_nom,         n, randomize, self.p).unsqueeze(-1)      
        inertia      = self._uniform_multiplicative(self.inertia_nom,      n, randomize, self.p)       
        length       = self._uniform_additive(self.length_nom,             n, randomize, 0.1*self.p)                  
        angle        = self._uniform_additive(self.angle_nom,              n, randomize, self.p).unsqueeze(-1)
        torque_const = self._uniform_multiplicative(self.torque_const_nom, n, randomize, self.p)                                 
        tau_m        = self._uniform_multiplicative(self.tau_m_nom,        n, randomize, self.p)                 
        eta          = self._uniform_multiplicative(self.eta_nom,          n, randomize, self.p) # NOTE: Maybe clamp to 1.0?                  
        Cd           = self._uniform_multiplicative(self.Cd_nom,           n, randomize, self.p)                   
        rho          = self._uniform_multiplicative(self.rho_nom,          n, randomize, self.p).unsqueeze(-1)                    
        return mass, inertia, length, angle, torque_const, tau_m, eta, Cd, rho