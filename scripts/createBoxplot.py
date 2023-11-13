import pickle
import matplotlib.pyplot as plt
import seaborn as sns 

# Load the probabilities from the pickle file
with open("bioProjectIds/probability_data.pkl", "rb") as file:
    data = pickle.load(file)

prob_0_when_true_0 = data["prob_0_when_true_0"]
prob_0_when_true_1 = data["prob_0_when_true_1"]
prob_1_when_true_1 = data["prob_1_when_true_1"]
prob_1_when_true_0 = data["prob_1_when_true_0"]

combine = [prob_0_when_true_0 + prob_1_when_true_0, prob_0_when_true_1 + prob_1_when_true_1]

plt.figure(figsize=(8, 6))
boxplot = plt.boxplot(
    combine,
    patch_artist=True,
    labels=['True 0s', 'True 1s']
)

colors = ['lightblue', 'lightgreen']
for box, color in zip(boxplot['boxes'], colors):
    box.set_facecolor(color)

plt.ylabel("Probability")
plt.title("Probability Distribution")
plt.savefig("/bioProjectIds/probabilityDistro.png")
plt.show()
