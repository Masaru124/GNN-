import sys

def main():
    print("--- CHECKING INSTALLED EXTERNAL GNN LIBRARIES ---")
    packages = ["alignn", "chnet", "matgl", "cgcnn", "pymatgen", "torch", "torch_geometric"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "installed")
            print(f"  [INSTALLED] {pkg}: {version}")
        except ImportError:
            print(f"  [NOT INSTALLED] {pkg}")

if __name__ == "__main__":
    main()
