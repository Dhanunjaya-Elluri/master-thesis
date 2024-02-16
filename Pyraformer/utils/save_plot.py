import matplotlib.pyplot as plt
import seaborn as sns


def save_matching_distribution_plot(result_df, folder_path, data_name):
    # Save Matching distribution plot to folder_path
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    # Group by 'Character Distance' and count the occurrences
    character_distance_counts = (
        result_df["Character Distance"].value_counts().reset_index().astype(int)
    )
    character_distance_counts.columns = ["Character Distance", "Count"]
    character_distance_counts = character_distance_counts.sort_values(
        by="Character Distance"
    )

    # Calculate the cumulative count and find the index for the 95% quantile
    total_count = character_distance_counts["Count"].sum()
    cumulative_count = character_distance_counts["Count"].cumsum()
    quantile_95_value = total_count * 0.95
    quantile_index = cumulative_count[cumulative_count >= quantile_95_value].index[0]

    accepted_count = character_distance_counts.iloc[: quantile_index + 1]["Count"].sum()
    denied_count = character_distance_counts.iloc[quantile_index + 1 :]["Count"].sum()

    # Get the character distance value at the 95% quantile
    # quantile_value = character_distance_counts.iloc[quantile_index]['Character Distance']

    # Plotting
    plt.figure(figsize=(12, 8))
    palette = [
        "green" if i <= quantile_index else "orange"
        for i in range(len(character_distance_counts))
    ]
    barplot = sns.barplot(
        x="Character Distance",
        y="Count",
        data=character_distance_counts,
        palette=palette,
    )

    # Adding count values on top of the bars
    for p in barplot.patches:
        # Calculate the x-coordinate for each bar's center
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        barplot.text(x, y, int(y), color="black", ha="center", va="bottom")

    # Marking the 95% quantile
    plt.axvline(
        x=quantile_index,
        color="red",
        linestyle="--",
        label="95% Quantile Line",
        linewidth=1,
    )
    plt.text(
        quantile_index,
        plt.ylim()[1] * 0.9,
        f"95% Quantile\n({quantile_index:.2f})",
        color="red",
        ha="right",
    )

    plt.xlabel("Character Distance between True and Predicted Values")
    plt.ylabel("Count")
    plt.title(f"Pyraformer: Char Distance Distribution for {data_name} dataset")
    plt.xticks()  # Rotate x-axis labels for better readability

    # Adding legend for the colors of the bars
    blue_patch = plt.Rectangle(
        (0, 0), 1, 1, color="blue", label=f"Within 95% (Count: {accepted_count})"
    )
    red_patch = plt.Rectangle(
        (0, 0), 1, 1, color="red", label=f"Outside 95% (Count: {denied_count})"
    )
    total_patch = plt.Rectangle(
        (0, 0), 1, 1, color="gray", label=f"Total: {total_count}"
    )
    # Add total count and 95% quantile to the legend
    plt.legend(
        handles=[
            blue_patch,
            red_patch,
            total_patch,
            plt.Line2D(
                [0], [0], color="green", linestyle="--", label="95% Quantile Line"
            ),
        ]
    )
    plt.savefig(folder_path + "matching_distribution.png")
