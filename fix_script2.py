import sys

with open('/home/laky/Desktop/proj/major_project/comparative_study.py', 'r') as f:
    content = f.read()

old_code = """    with torch.no_grad():
        Z_train, _, _ = vae(torch.tensor(X_train, dtype=torch.float32))
        Z_test, _, _ = vae(torch.tensor(X_test, dtype=torch.float32))"""

new_code = """    with torch.no_grad():
        Z_train = vae.get_latent(torch.tensor(X_train, dtype=torch.float32), deterministic=True)
        Z_test = vae.get_latent(torch.tensor(X_test, dtype=torch.float32), deterministic=True)"""

content = content.replace(old_code, new_code)

with open('/home/laky/Desktop/proj/major_project/comparative_study.py', 'w') as f:
    f.write(content)
