import numpy as np
import matplotlib.pyplot as plt

# 设置参数
plt.rcParams['text.usetex'] = False
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

categories = ['Common', 'Controversial']
labels = ['BM25', 'BERT', 'SAILER', 'KELLER']
hatchs = ['/', '.', '-', '\\']
colors = ['#FFC300', '#FF5733', '#C70039', '#FF2347']

# MAP
bm25_scores = [0.544, 0.617]
bert_scores = [0.595, 0.671]
sailer_scores = [0.622, 0.731]
keller_scores = [0.670, 0.829]
data = {
    'BM25': bm25_scores,
    'BERT': bert_scores,
    'SAILER': sailer_scores,
    'KELLER': keller_scores
}

# 设置柱状图参数
x = np.arange(len(categories))  # the label locations
total_width, n = 0.8, 4  # 修改总宽度以减少空隙，每组4个柱子
width = total_width / n  # 柱子宽度
x = x - (total_width - width) / 2  # 调整x轴位置

colors = ['#FFC300', '#FF5733', '#C70039', '#900C3F']  # 使用类似原始颜色的新颜色组

# 创建图表
fig, ax = plt.subplots(figsize=(6, 4))
for i, method in enumerate(data.keys()):
    bars = ax.bar(x + i * width, data[method], width=width, label=method, ec='white', hatch=hatchs[i % len(hatchs)], color=colors[i])
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 3), ha='center', va='bottom', fontsize=8)


# 设置图表属性
ax.set_xlabel('Query Category', fontweight='bold')
ax.set_ylabel('MAP', fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(categories)
ax.legend(loc='best')
ax.set_ylim(0.5, 0.85)

plt.savefig('LeCaRDv2.pdf')