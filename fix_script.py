import sys

with open('/home/laky/Desktop/proj/major_project/comparative_study.py', 'r') as f:
    content = f.read()

content = content.replace("trainer = VAETrainer(vae, learning_rate=1e-3, beta=0.005)", "trainer = VAETrainer(vae, lr=1e-3, beta=0.005)")
content = content.replace("history = trainer.train(X_train, epochs=150, batch_size=64, validation_split=0.1)", "history = trainer.fit(X_train, epochs=150, batch_size=64)")

with open('/home/laky/Desktop/proj/major_project/comparative_study.py', 'w') as f:
    f.write(content)
