import matplotlib.pyplot as plt

numerical_features = [
    "Area",
    "Perimeter",
    "Major_Axis_Length",
    "Minor_Axis_Length",
    "Eccentricity",
    "Convex_Area",
    "Extent"
]

#Phase 2- Understanding the data


#examine distribution of each numerical feature
def plot_feature_distributions(df):
    df.hist(bins=30,figsize=(12,8),edgecolor='black')
    plt.tight_layout()
    plt.show()


#examine outliers in numerical features
def plot_outliers(df):

    fig,axes = plt.subplots(4,2,figsize=(14,16))

    for ax,feature in zip(axes.flatten(),numerical_features):
        ax.boxplot(df[feature])
        ax.set_title(f"{feature} Distribution")
        ax.set_ylabel(feature)

    # Remove the unused eighth subplot
    fig.delaxes(axes.flatten()[7])
    plt.tight_layout()
    plt.show()


#correlation matrix
def plot_correlations(df):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax)
    #-- Set axis ticks and labels
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    #-- Annotate each cell with its correlation value
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")
    plt.tight_layout()
    plt.show()


#Distribution of numeric features for each class
def plot_class_distributions(df):
        numerical_features=[
            "Area",
            "Perimeter",
            "Major_Axis_Length",
            "Minor_Axis_Length",
            "Eccentricity",
            "Convex_Area",
            "Extent"
        ]

        fig,axes = plt.subplots(4,2,figsize=(14,16))

        for ax, feature in zip(axes.flatten(),numerical_features):
        #separate measurements by rice variety
            cammeo = df[df["Class"] == "Cammeo"][feature]
            osmancik = df[df["Class"] == "Osmancik"][feature]

        #Generate comparative box plots
            ax.boxplot(
                [cammeo,osmancik],
                labels=["Cammeo","Osmancik"]
    )

        #Add plot title and axis labels
            ax.set_title(f"{feature} by Rice Variety")
            ax.set_ylabel(feature)

        #Remove unused subplot
        fig.delaxes(axes.flatten()[7])

        plt.tight_layout()
        plt.show()

#investigate whether classes are distinguishable by features
def plot_class_separability(df):
    feature_pairs=[
        ("Area","Perimeter"),
        ("Major_Axis_Length", "Minor_Axis_Length"),
        ("Eccentricity", "Extent")
    ]
    fig,axes = plt.subplots(1,3, figsize=(18,5))
    for ax, (feature_x, feature_y) in zip(axes,feature_pairs):
        for variety in ["Cammeo", "Osmancik"]:
            subset = df[df["Class"] == variety]
            ax.scatter(
                subset[feature_x],
                subset[feature_y],
                label=variety,
                alpha=0.5
            )
        ax.set_xlabel(feature_x)
        ax.set_ylabel(feature_y)
        ax.set_title(f"{feature_x} vs {feature_y}")
        ax.legend()

    plt.tight_layout()
    plt.show()
