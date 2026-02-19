import matplotlib.pyplot as plt
import pandas as pd
import sacrebleu
import seaborn as sns

split = "test"

model_names = {
    0: "no_context_empty_sys",
    2: "full_context_empty_sys_2_turns",
    6: "full_context_empty_sys_6_turns",
    10: "full_context_empty_sys_10_turns",
    15: "full_context_empty_sys_15_turns",
    20: "full_context_empty_sys_20_turns",
    100: "full_context_empty_sys",
}

all_scores = []
full_df = pd.read_csv("data.csv")
for lang in ["de", "fr", "pt", "ko", "nl"]:
    df_lang = lang if lang != "pt" else "pt-br"
    df = full_df[
        (full_df["source_language"] == df_lang)
        | (full_df["target_language"] == df_lang)
    ]
    df.fillna("", inplace=True)

    for lp, lp_df in df.groupby("lp"):
        for t, mname in model_names.items():
            all_scores.append(
                [
                    lp,
                    t,
                    sacrebleu.corpus_chrf(
                        lp_df[mname].to_list(), [lp_df["reference"].to_list()]
                    ).score,
                ]
            )

scores_df = pd.DataFrame(all_scores)
scores_df.columns = ["LP", "# of Turns", "CHRF"]

scores_df["# of Turns"] = scores_df["# of Turns"].astype(str)
scores_df["# of Turns"] = scores_df["# of Turns"].replace("100", "Full")
scores_df["# of Turns"] = scores_df["# of Turns"].replace("0", "No")
scores_df["LP"] = scores_df["LP"].replace("en-pt-br", "en-pt")
scores_df["LP"] = scores_df["LP"].replace("pt-br-en", "pt-en")
scores_df["LP"] = scores_df["LP"].apply(lambda x: x.upper())
scores_df["Language"] = scores_df["LP"].apply(
    lambda x: x.split("-")[1].upper() if x.startswith("EN") else x.split("-")[0].upper()
)
scores_df["Direction"] = scores_df["LP"].apply(
    lambda x: r"EN$\rightarrow$XX" if x.startswith("EN") else r"XX$\rightarrow$EN"
)

sns.set_style(rc={"patch.force_edgecolor": True, "patch.edgecolor": "black"})
sns.set_style("ticks")

sns.set_style(
    rc={"patch.force_edgecolor": True, "patch.edgecolor": "black", "font": "Palatino"}
)

plt.rcParams["font.family"] = "Palatino"  # change to palatino
plt.rcParams["font.size"] = 22  # change to palatino

cmap = sns.color_palette(["#3953BF", "#b54343"])

for metric in ["CHRF"]:
    plt.figure(figsize=(10, 8))

    ax = sns.lineplot(
        data=scores_df,
        x="# of Turns",
        y=metric,
        hue="Language",
        style="Direction",
        linewidth=2.5,
        palette="colorblind",
        markers=["s", "o"],
        # size=10000
        markersize=10,
    )
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    for spine in ax.spines.values():
        spine.set_visible(False)  # Hide all spines

    # Show only x and y axes
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)

    # Increase line width for the visible spines
    ax.spines["bottom"].set_linewidth(2)  # Adjust as needed
    ax.spines["left"].set_linewidth(2)  # Adjust as needed

    ax.set_yticks([60, 65, 70, 75, 80])
    ax.set_xlim(-0.2, 6.1)
    plt.grid(alpha=0.25)
    plt.savefig("plot.png", bbox_inches="tight", dpi=1000)
    plt.show()
