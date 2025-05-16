import h5py

# 打开 .h5 文件
file_path = '/Users/peony/Downloads/TrajCogn-5021/sample/small_chengdu.h5'  # 替换为你的文件路径
with h5py.File(file_path, 'r') as hdf:
    # 查看文件中的所有主键
    def print_structure(name, obj):
        print(f"{name}: {type(obj)}")
    
    # 遍历文件中的所有对象
    hdf.visititems(print_structure)