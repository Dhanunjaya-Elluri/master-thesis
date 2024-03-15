import matplotlib.pyplot as plt
import seaborn as sns


def save_matching_distribution_plot(
    result_df, folder_path, data_name, model_name, embed_type
):
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
    mean = result_df["Character Distance"].mean()

    # Calculate the cumulative count and find the index for the 95% quantile
    total_count = character_distance_counts["Count"].sum()
    cumulative_count = character_distance_counts["Count"].cumsum()
    quantile_95_value = total_count * 0.95
    quantile_85_value = total_count * 0.85
    quantile_95_index = cumulative_count[cumulative_count >= quantile_95_value].index[0]
    quantile_85_index = cumulative_count[cumulative_count >= quantile_85_value].index[0]

    accepted_95_count = character_distance_counts.iloc[: quantile_95_index + 1][
        "Count"
    ].sum()
    denied_95_count = character_distance_counts.iloc[quantile_95_index + 1 :][
        "Count"
    ].sum()

    accepted_85_count = character_distance_counts.iloc[: quantile_85_index + 1][
        "Count"
    ].sum()
    denied_85_count = character_distance_counts.iloc[quantile_85_index + 1 :][
        "Count"
    ].sum()

    # Plotting
    plt.figure(figsize=(12, 8))
    palette = [
        "green"
        if i <= quantile_85_index
        else ("yellow" if i <= quantile_95_index else "orange")
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

    # Marking the mean
    plt.axvline(
        x=mean,
        color="black",
        linestyle="--",
        label="Avg",
        linewidth=1,
    )
    # Marking the 95% quantile
    plt.axvline(
        x=quantile_85_index,
        color="blue",
        linestyle="--",
        label="85% Quantile Line",
        linewidth=1,
    )
    plt.axvline(
        x=quantile_95_index,
        color="red",
        linestyle="--",
        label="95% Quantile Line",
        linewidth=1,
    )
    plt.text(
        mean,
        plt.ylim()[1] * 0.9,
        f"Mean\n({mean:.2f})",
        color="black",
        ha="left",
    )
    plt.text(
        quantile_85_index,
        plt.ylim()[1] * 0.8,
        f"85% Quantile\n({quantile_85_index:.2f})",
        color="blue",
        ha="right",
    )
    plt.text(
        quantile_95_index,
        plt.ylim()[1] * 0.6,
        f"95% Quantile\n({quantile_95_index:.2f})",
        color="red",
        ha="right",
    )

    plt.xlabel("Character Distance between True and Predicted Values", fontsize=16)
    plt.ylabel("Count", fontsize=16)
    if embed_type:
        if embed_type == 1:
            plt.title(
                f"{model_name}: Char Distance Distribution for {data_name} with Embed Type: Token + Temporal + Positional",
                fontsize=16,
            )
        elif embed_type == 2:
            plt.title(
                f"{model_name}: Char Distance Distribution for {data_name} with Embed Type: Token + Temporal",
                fontsize=16,
            )
        elif embed_type == 3:
            plt.title(
                f"{model_name}: Char Distance Distribution for {data_name} with Embed Type: Token + Positional",
                fontsize=16,
            )
        elif embed_type == 4:
            plt.title(
                f"{model_name}: Char Distance Distribution for {data_name} with Embed Type: Token",
                fontsize=16,
            )
    else:
        plt.title(
            f"{model_name}: Char Distance Distribution for {data_name} dataset",
            fontsize=16,
        )
    plt.xticks()  # Rotate x-axis labels for better readability

    # Adding legend for the colors of the bars
    green_patch = plt.Rectangle(
        (0, 0), 1, 1, color="green", label=f"Within 85% (Count: {accepted_85_count})"
    )
    yellow_patch = plt.Rectangle(
        (0, 0),
        1,
        1,
        color="yellow",
        label=f"Between 85% and 95% (Count: {accepted_95_count - accepted_85_count})",
    )
    orange_patch = plt.Rectangle(
        (0, 0), 1, 1, color="orange", label=f"Above 95% (Count: {denied_95_count})"
    )
    total_patch = plt.Rectangle(
        (0, 0), 1, 1, color="gray", label=f"Total: {total_count}"
    )
    # blue_patch = plt.Rectangle(
    #     (0, 0), 1, 1, color="green", label=f"Within 95% (Count: {accepted_95_count})"
    # )
    # red_patch = plt.Rectangle(
    #     (0, 0), 1, 1, color="orange", label=f"Outside 95% (Count: {denied_95_count})"
    # )
    # total_patch = plt.Rectangle(
    #     (0, 0), 1, 1, color="gray", label=f"Total: {total_count}"
    # )
    # Add total count and 95% quantile to the legend
    plt.legend(
        handles=[
            green_patch,
            yellow_patch,
            orange_patch,
            total_patch,
            plt.Line2D(
                [0], [0], color="red", linestyle="--", label="95% Quantile Line"
            ),
            plt.Line2D(
                [0], [0], color="blue", linestyle="--", label="85% Quantile Line"
            ),
            plt.Line2D([0], [0], color="black", linestyle="--", label="Avg"),
        ]
    )
    plt.savefig(folder_path + "matching_distribution.png")
