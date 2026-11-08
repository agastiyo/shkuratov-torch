import torch
from shkuratov import Material, ShkuratovModel
import matplotlib.pyplot as plt

wavelengths = torch.linspace(0.5, 5.0, 500)

# Refractive index: smooth baseline with a modest dispersion feature
n_values = 1.30 + 0.015 * torch.exp(
    -0.5 * ((wavelengths - 2.0) / 0.18) ** 2
)

# Extinction coefficient: clear absorption band at 2.0 microns
k_values = 0.001 + 0.08 * torch.exp(
    -0.5 * ((wavelengths - 2.0) / 0.12) ** 2
)

n = torch.stack([n_values, wavelengths])
k = torch.stack([k_values, wavelengths])

generator = ShkuratovModel()

test_mat = Material(n,k)

lams = torch.arange(0.4,5.6,0.01)
test = 1
for i in range(test):
  A = generator.simulate_one_component(test_mat, 5.0, 0.7, lams)
  print(f"Finished {i+1}/{test}", end='\r')

plt.plot(test_mat.n[1],test_mat.n[0],label='n',color='orange')
plt.plot(test_mat.k[1],test_mat.k[0],label='k',color='blue')
plt.plot(lams, A, label='Spectra')
plt.legend()
plt.show()