import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['text.usetex'] = False
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# Function to plot a bar graph with corrected x-ticks
def plot_bar_graph_corrected(ax, categories, data, hatchs, colors):
    x = np.arange(len(categories))  # label locations
    total_width, n = 0.8, 4  # total width and number of bars in each group
    width = total_width / n  # width of each bar

    for i, method in enumerate(data.keys()):
        bars = ax.bar(x + i * width, data[method], width=width, label=method, ec='white', hatch=hatchs[i % len(hatchs)], color=colors[i])
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 3), ha='center', va='bottom', fontsize=12)

    ax.set_xticks(x + total_width / 2 - width / 2)
    ax.set_xticklabels(categories, fontsize=12)  # Increased fontsize for x-tick labels
    ax.set_xlabel('Query Category', fontweight='bold', fontsize=14)  # Increased fontsize for x-axis label
    ax.set_ylabel('MAP', fontweight='bold', fontsize=14)  # Increased fontsize for y-axis label
    ax.tick_params(axis='both', which='major', labelsize=12)  # Increased fontsize for tick labels
    ax.legend(loc='best', fontsize=11)  # Increased fontsize for legend

# Data for first plot
categories1 = ['Common', 'Controversial']
bm25_scores1 = [0.482, 0.451]
bert_scores1 = [0.578, 0.438]
sailer_scores1 = [0.640, 0.520]
keller_scores1 = [0.678, 0.645]
data1 = {
    'BM25': bm25_scores1,
    'BERT': bert_scores1,
    'SAILER': sailer_scores1,
    'KELLER': keller_scores1
}
hatchs1 = ['/', '.', '-', '\\']
colors1 = ['#FFC300', '#FF5733', '#C70039', '#900C3F']

# Data for second plot
categories2 = ['Common', 'Controversial']
bm25_scores2 = [0.544, 0.617]
bert_scores2 = [0.595, 0.671]
sailer_scores2 = [0.622, 0.731]
keller_scores2 = [0.670, 0.829]
data2 = {
    'BM25': bm25_scores2,
    'BERT': bert_scores2,
    'SAILER': sailer_scores2,
    'KELLER': keller_scores2
}
hatchs2 = ['/', '.', '-', '\\']
colors2 = ['#FFC300', '#FF5733', '#C70039', '#900C3F']

# Create subplots for each graph
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plotting first graph with corrected x-ticks and adjusted (a) label
plot_bar_graph_corrected(ax1, categories1, data1, hatchs1, colors1)
ax1.set_xlabel('Query Category', fontweight='bold')
ax1.set_ylabel('MAP', fontweight='bold')
ax1.set_ylim(0.4, 0.7)
ax1.text(0.5, -0.15, '(a)', ha='center', va='center', transform=ax1.transAxes, fontsize=15)

# Plotting second graph with corrected x-ticks and adjusted (b) label
plot_bar_graph_corrected(ax2, categories2, data2, hatchs2, colors2)
ax2.set_xlabel('Query Category', fontweight='bold')
ax2.set_ylabel('MAP', fontweight='bold')
ax2.set_ylim(0.5, 0.85)
ax2.text(0.5, -0.15, '(b)', ha='center', va='center', transform=ax2.transAxes, fontsize=15)

plt.tight_layout()
plt.savefig('combined.pdf')
