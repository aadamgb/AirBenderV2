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
                 Cd:        float = (0.28,0.35,0.70),
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
        self.tau_m_nom   = torch.as_tensor(tau_m,    dtype=torch.float32, device=self.device) # NOTE: Maybe have tau per env? test first with normal tau
        eta_nom          = torch.as_tensor(eta,      dtype=torch.float32, device=self.device)
        self.Cd_nom      = torch.as_tensor(Cd,       dtype=torch.float32, device=self.device)
        
        if length_nom.ndim == 0:
            length_nom = length_nom.expand(4)  
        self.length_nom = length_nom.clone()  
        
        if torque_const_nom.ndim == 0:
            torque_const_nom = torque_const_nom.expand(4)
        self.torque_const_nom = torque_const_nom.clone()  
          
        if eta_nom.ndim == 0:
            eta_nom = eta_nom.expand(4)
        self.eta_nom = eta_nom.clone()  
                       

    def _uniform_multiplicative(self, nominal: torch.Tensor, n: int, randomize: bool, p: float) -> torch.Tensor:
            out = nominal.expand((n,) + tuple(nominal.shape)).clone()
            if randomize and p > 0:
                scale = 1.0 + (2.0 * torch.rand_like(out) - 1.0) * p  # U[1-p, 1+p]
                out = out * scale
            return out
    
    def _uniform_additive(self, nominal: torch.Tensor, n: int, randomize: bool, p: float) -> torch.Tensor:
        out = nominal.expand((n,) + tuple(nominal.shape)).clone()
        if randomize and p > 0:
            delta = (2.0 * torch.rand_like(out) - 1.0) * p  # U[-p, +p]
            out = out + delta
        return out

    def sample(self, n: int, randomize: bool = True):
        mass         = self._uniform_multiplicative(self.mass_nom,         n, randomize, self.p).unsqueeze(-1)      
        inertia      = self._uniform_multiplicative(self.inertia_nom,      n, randomize, self.p)       
        length       = self._uniform_additive(self.length_nom,             n, randomize, 0.1*self.p)                  
        angle        = self._uniform_additive(self.angle_nom,              n, randomize, self.p).unsqueeze(-1)
        torque_const = self._uniform_multiplicative(self.torque_const_nom, n, randomize, self.p)                   
        tau_m        = self._uniform_multiplicative(self.tau_m_nom,        n, randomize, self.p)                   
        eta          = self._uniform_multiplicative(self.eta_nom,          n, randomize, self.p)                   
        Cd           = self._uniform_multiplicative(self.Cd_nom,           n, randomize, self.p)                   
        return mass, inertia, length, angle, torque_const, #tau_m, eta, Cd