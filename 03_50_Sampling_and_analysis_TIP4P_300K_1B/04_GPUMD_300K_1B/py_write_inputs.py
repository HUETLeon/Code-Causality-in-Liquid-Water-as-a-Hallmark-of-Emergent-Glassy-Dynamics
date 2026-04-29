from ase.io import read, write

# Read all frames from your extended XYZ trajectory
atoms_list = read('dump.xyz', index=':')
ls = len(atoms_list)

# Write them back to a new Extended XYZ file
for i, frame in enumerate(atoms_list[::ls//50]):
    write(f'run_{i+1}/model.xyz', frame, format='extxyz')

