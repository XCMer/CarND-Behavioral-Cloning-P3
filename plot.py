import csv
import matplotlib.pyplot as plt

with open('training_log.csv', 'r') as f:
    reader = csv.DictReader(f)
    epochs = []
    training = []
    validation = []

    for row in reader:
        epochs.append(int(row['epoch']))
        training.append(float(row['loss']))
        validation.append(float(row['val_loss']))


fig = plt.figure()
plt.plot(epochs, training, label='Training loss')
plt.plot(epochs, validation, label='Validation loss')
plt.legend(loc="upper left", bbox_to_anchor=[0, 1],
           ncol=2, shadow=True, title="Legend", fancybox=True)
fig.savefig('loss_plot.png')
plt.close()
