import sys, platform
import numpy, pandas, sklearn, matplotlib, torch

print("Python      :", sys.version.split()[0], "|", platform.system())
print("NumPy       :", numpy.__version__)
print("Pandas      :", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
print("Matplotlib  :", matplotlib.__version__)
print("PyTorch     :", torch.__version__)
print("CUDA        :", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

x = torch.randn(4, 6, 150)
print("Tensor test :", x.shape, "OK")