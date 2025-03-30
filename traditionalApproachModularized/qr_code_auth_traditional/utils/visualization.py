import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(feature_names, importances):
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importances)
    plt.xlabel("Importance Score")
    plt.title("Feature Importances")
    plt.show()

def plot_feature_distributions(features, labels, feature_name):
    plt.figure(figsize=(8, 6))
    sns.histplot(features[labels==0], label='Original', kde=True)
    sns.histplot(features[labels==1], label='Counterfeit', kde=True)
    plt.xlabel(feature_name)
    plt.title(f"Distribution of {feature_name}")
    plt.legend()
    plt.show()
