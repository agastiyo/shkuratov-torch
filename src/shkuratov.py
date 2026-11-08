import torch
import numpy as np

class Material():
  def __init__(self, n:torch.Tensor, k:torch.Tensor):
    assert n.shape[0] == 2 #(n, lambda)
    assert k.shape[0] == 2 #(k, lambda)
    # TODO: assert that n and k wavelengths are sorted in ascending order
    # TODO: assert that n and k wavelengths have no duplicates
    self.n = n
    self.k = k


class ShkuratovModel():
  def __init__(self):
    # Cache Gauss-Legendre points and weights once
    self.gl_x,self.gl_w = np.polynomial.legendre.leggauss(16) #16 to 32 should work here, dim G
    self.gl_x = torch.tensor(self.gl_x).unsqueeze(0) # dim 1xG
    self.gl_w = torch.tensor(self.gl_w).unsqueeze(0) # dim 1xG
    
    # Linear stretch of points and weights for each integral
    Rb_interval = (0,torch.pi/4)
    Rf_interval = (torch.pi/4, torch.pi/2)
    
    self.Rb_theta = (self.gl_x*(Rb_interval[1] - Rb_interval[0]))/2 + (Rb_interval[0] + Rb_interval[1])/2 # all dim 1xG
    self.Rb_W = (self.gl_w*(Rb_interval[1] - Rb_interval[0]))/2
    self.Rf_theta = (self.gl_x*(Rf_interval[1] - Rf_interval[0]))/2 + (Rf_interval[0] + Rf_interval[1])/2
    self.Rf_W = (self.gl_w*(Rf_interval[1] - Rf_interval[0]))/2
  
  def simulate(self): # add OCs, fractions, grain sizes, porosity, wavelengths
    # Forward pass
    pass
  
  def simulate_one_component(self, mat:Material, size:float, porosity:float, lams:torch.Tensor):
    ''' A forward pass of the Shkuratov model with one component'''
    assert 0 <= porosity <= 1
    
    # Build the optical constant lists using the specified wavelengths.
    lams = lams.unsqueeze(1)
    n,k = self._build_oc(mat,lams)  # dim Lx1
    
    tau = (4*torch.pi*k*size)/lams # dim Lx1
    
    rho_b,rho_f = self.__indicatrix(n,tau,size,porosity) # both dim Lx1
    
    # Albedo at each requested wavelength
    A = ((1 + rho_b**2 - rho_f**2)/(2*rho_b)) - torch.sqrt(((1 + rho_b**2 - rho_f**2)/(2*rho_b))**2 - 1) # dim Lx1
    
    return A.flatten() # albedo of dim L
  
  def __indicatrix(self, n:torch.Tensor, tau:torch.Tensor, size:float, porosity:float):
    # Gauss-Legendre quadrature to solve for forward and backward reflectance coefficients
    Rb = torch.sum(self.Rb_W*self.__refl_coeff_integrand(n,self.Rb_theta),dim=1,keepdim=True) # both dim LxG, then summed over to Lx1
    Rf = torch.sum(self.Rf_W*self.__refl_coeff_integrand(n,self.Rf_theta),dim=1,keepdim=True)
    
    Re = Rb+Rf # all dim Lx1
    Ri = 1 - (1-Re)/(n**2)
    TeTi = (1 - Re) * (1 - Ri)
    
    # Cache repeated operations
    exp_neg_tau = torch.exp(-tau)
    a= ((0.5*TeTi*Ri*torch.exp(-2*tau))/(1 - Ri*exp_neg_tau))
    
    # Find fraction of rays scattered forward and backwards
    rb = Rb + a # all dim Lx1
    rf = Rf + (TeTi*exp_neg_tau) + a
    
    # Indicatrix using porosity
    q = 1 - porosity
    rho_b = q*rb # all dim Lx1
    rho_f = (q*rf) + 1 - q
    
    return rho_b,rho_f
    
  def __refl_coeff_integrand(self,n,theta):
    return 2 * self.__fresnel(n,theta) * torch.cos(theta) * torch.sin(theta) # dim LxG
    
  def __fresnel(self,n, theta):
    theta_t = torch.arcsin((1/n)*torch.sin(theta))
    
    r_s = (torch.cos(theta) - n*torch.cos(theta_t))/(torch.cos(theta) + n*torch.cos(theta_t))
    r_p = (n*torch.cos(theta) - torch.cos(theta_t))/(n*torch.cos(theta) + torch.cos(theta_t))
    
    return 0.5 * (r_s**2 + r_p**2)
  
  def _build_oc(self, mat:Material, lams:torch.Tensor):
    '''Building optical constants at the requested wavelengths using the known optical constants.
    
    If the requested wavelength is out of range, use the closest boundary value
    If it is in range, use linear interpolation with the two closest points.
    '''
    
    known_n = mat.n[0]
    known_k = mat.k[0]
    known_lams_n = mat.n[1]
    known_lams_k = mat.k[1]
    
    # Clamp out of range wavelengths to the min and max, so out of range wavelengths are just requesting the boundary values
    requested_lams_n = torch.clamp(lams, min=known_lams_n[0], max=known_lams_n[-1])
    requested_lams_k = torch.clamp(lams, min=known_lams_k[0], max=known_lams_k[-1])
    
    # Get the index of known wavelengths right above and below the requested wavelength
    lam_upper_idx_n = torch.searchsorted(known_lams_n, requested_lams_n).clamp(min=1)
    lam_upper_idx_k = torch.searchsorted(known_lams_k, requested_lams_k).clamp(min=1)
    lam_lower_idx_n = (lam_upper_idx_n - 1)
    lam_lower_idx_k = (lam_upper_idx_k - 1)
    
    # Get the known wavelengths right above and below the requested wavelengths
    upper_lams_n = known_lams_n[lam_upper_idx_n]
    lower_lams_n = known_lams_n[lam_lower_idx_n]
    upper_lams_k = known_lams_k[lam_upper_idx_k]
    lower_lams_k = known_lams_k[lam_lower_idx_k]
    
    # Linear interpolation formula: n = n_lower + (n1 - n0)*((requested wavelength - lambda0) / (lambda1 - lambda0))
    requested_n = known_n[lam_lower_idx_n] + (known_n[lam_upper_idx_n]-known_n[lam_lower_idx_n])*((requested_lams_n - lower_lams_n)/(upper_lams_n - lower_lams_n))
    requested_k = known_k[lam_lower_idx_k] + (known_k[lam_upper_idx_k]-known_k[lam_lower_idx_k])*((requested_lams_k - lower_lams_k)/(upper_lams_k - lower_lams_k))
    
    return requested_n,requested_k
