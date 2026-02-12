import h5py

def print_h5_structure(file_path, indent=0):
    """递归打印h5文件结构"""
    with h5py.File(file_path, 'r') as f:
        def print_attrs(name, obj):
            print('  ' * indent + f"{name}: {type(obj)}")
            if isinstance(obj, h5py.Dataset):
                print('  ' * indent + f"  Shape: {obj.shape}, Dtype: {obj.dtype}")
        f.visititems(print_attrs)

print("METR-LA structure:")
print_h5_structure('/home/xiaoxiao/FlowGNN/data/metr-la.h5')

print("\nPEMS-BAY structure:")
print_h5_structure('/home/xiaoxiao/FlowGNN/data/pems-bay.h5')
