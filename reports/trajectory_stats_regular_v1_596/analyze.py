"""Read-only analysis of the verified regular-v1 release; rerun with python analyze.py."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path('/mnt/shared-storage-user/sdpdev-fs/sunxiangyu/drug_wd/drug_pipe_regular_v1_20260908')
OUT = Path(__file__).resolve().parent

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read_rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]

def table(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def stats(values):
    a = np.asarray(values)
    return dict(count=len(a), total=int(a.sum()), min=int(a.min()), mean=float(a.mean()),
                median=float(np.median(a)), p90=float(np.percentile(a, 90)),
                p95=float(np.percentile(a, 95)), max=int(a.max()))

manifest = json.loads((ROOT / 'release_manifest.json').read_text())
files = ['training/qwen35_sft_train.jsonl', 'semantic_trajectories.jsonl']
hashes = {f: digest(ROOT / f) for f in files}
for f in files:
    assert hashes[f] == manifest['files'][f], f
assert digest(ROOT / 'training/context_gate_manifest.json') == manifest['training_manifest_sha256']
gate = json.loads((ROOT / 'training/context_gate_manifest.json').read_text())
assert hashes[files[0]] == gate['output_sha256']
training = read_rows(ROOT / files[0])
semantic = {r['id']: r for r in read_rows(ROOT / files[1])}
lengths = {r['id']: r for r in gate['record_lengths']}
assert len(training) == manifest['training_count'] == gate['accepted_count'] == 596
assert len({r['id'] for r in training}) == 596
assert set(semantic) - {r['id'] for r in training} == {r['id'] for r in gate['excluded']}
records, calls, reach, skills = [], Counter(), Counter(), Counter()
for r in training:
    s = semantic[r['id']]
    decisions = [m for m in r['messages'] if m['role'] == 'assistant']
    tool_calls = [t['function'] for m in decisions for t in m.get('tool_calls', [])]
    names = [t['name'] for t in tool_calls]
    observations = [e for e in s['events'] if e['type'] == 'tool_observation']
    assert len(decisions) == sum(e['type'] == 'assistant_decision' for e in s['events'])
    assert len(tool_calls) == len(observations)
    assert len(tool_calls) == sum(m['role'] == 'tool' for m in r['messages'])
    calls.update(names)
    reach.update(set(names))
    for t in tool_calls:
        if t['name'] == 'skill':
            args = t['arguments']
            if isinstance(args, str):
                args = json.loads(args)
            skills.update([args['name']])
    records.append(dict(id=r['id'], task_type=s['metadata']['task_type'].upper(),
        steps=len(decisions), tool_rounds=sum(bool(m.get('tool_calls')) for m in decisions),
        tool_calls=len(names), unique_tools=len(set(names)), skill_calls=names.count('skill'),
        parallel_rounds=sum(len(m.get('tool_calls', [])) > 1 for m in decisions),
        max_parallel_calls=max([len(m.get('tool_calls', [])) for m in decisions]),
        explicit_error_observations=sum(e.get('is_error') is True or e.get('status') == 'error' for e in observations),
        tokens=lengths[r['id']]['tokens'], trainable_tokens=lengths[r['id']]['trainable_tokens']))
assert all(0 < r['trainable_tokens'] <= r['tokens'] <= gate['max_tokens'] for r in records)
table('trajectory_metrics.csv', records)
order = ['AC', 'PF', 'VS', 'KG', 'E2E']
colors = dict(zip(order, ['#3978b7', '#42a58b', '#e5a13c', '#8a68b2', '#db706b']))
summary = dict(release=str(ROOT), verified_sha256=hashes, step_definition='assistant decision rounds, including final answer; parallel calls count as one round',
    metrics={k: stats([r[k] for r in records]) for k in ['steps', 'tool_calls', 'unique_tools', 'tokens', 'trainable_tokens']},
    tasks=[], trajectories_with_parallel_calls=sum(r['parallel_rounds'] > 0 for r in records),
    explicit_error_observations=sum(r['explicit_error_observations'] for r in records),
    trajectories_with_explicit_errors=sum(r['explicit_error_observations'] > 0 for r in records),
    unique_tools=len(calls), unique_loaded_skills=len(skills),
    context_coverage={str(n): sum(r['tokens'] <= n for r in records) for n in [16384, 32768, 65536, 131072, 245760]},
    steps_tokens_pearson=float(np.corrcoef([r['steps'] for r in records], [r['tokens'] for r in records])[0, 1]))
for task in order:
    subset = [r for r in records if r['task_type'] == task]
    summary['tasks'].append(dict(task_type=task, count=len(subset), mean_steps=float(np.mean([r['steps'] for r in subset])),
        median_steps=float(np.median([r['steps'] for r in subset])), tokens=sum(r['tokens'] for r in subset),
        trainable_tokens=sum(r['trainable_tokens'] for r in subset)))
summary['trainable_token_fraction'] = sum(r['trainable_tokens'] for r in records) / sum(r['tokens'] for r in records)
summary['longest_10_percent_token_share'] = sum(sorted([r['tokens'] for r in records], reverse=True)[:60]) / sum(r['tokens'] for r in records)
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
table('task_summary.csv', summary['tasks'])
table('tool_usage.csv', [dict(tool=k, calls=v, trajectories=reach[k]) for k,v in calls.most_common()])
table('skill_usage.csv', [dict(skill=k, calls=v) for k,v in skills.most_common()])
font_manager.fontManager.addfont('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
plt.rcParams.update({'font.family':['DejaVu Sans', 'Droid Sans Fallback'], 'axes.unicode_minus':False, 'font.size':11,
                     'axes.spines.top':False, 'axes.spines.right':False, 'savefig.facecolor':'white'})
def save(fig, name):
    fig.savefig(OUT / (name + '.png'), dpi=180, bbox_inches='tight')
    fig.savefig(OUT / (name + '.svg'), bbox_inches='tight')
    plt.close(fig)
def discrete(ax, key, title, label):
    freq = Counter(r[key] for r in records)
    x = np.arange(min(freq), max(freq)+1)
    ax.bar(x, [freq[i] for i in x], width=.82, color='#3978b7')
    ax.set(title=title, xlabel=label, ylabel='轨迹数量')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=.18)
    return x, freq
fig, ax = plt.subplots(figsize=(16,6))
x,freq = discrete(ax, 'steps', '596 条训练轨迹：步数分布', '轨迹步数（assistant 决策轮数，包含最终回答）')
ax.set_xticks(x)
ax.tick_params(axis='x', labelsize=7)
for i in x:
    if freq[i]: ax.text(i, freq[i]+.45, str(freq[i]), ha='center', fontsize=7)
ax.set_ylim(0,max(freq.values())*1.18)
s = summary['metrics']['steps']
ax.text(.99,.96, f"N = 596    均值 = {s['mean']:.2f}    中位数 = {s['median']:g}    P95 = {s['p95']:g}\n每根柱对应一个整数长度；零频长度保留空位", transform=ax.transAxes,ha='right',va='top')
save(fig, 'steps_histogram')
table('step_frequency.csv', [dict(steps=int(i), trajectories=freq[i]) for i in x])
fig, axes = plt.subplots(2,2,figsize=(15,11),layout='constrained')
ax=axes[0,0]
counts=[sum(r['task_type']==t for r in records) for t in order]
bars=ax.bar(order,counts,color=[colors[t] for t in order]);ax.bar_label(bars,labels=[f'{n} ({n/596:.1%})' for n in counts],padding=4)
ax.set(title='任务类型构成',ylabel='轨迹数量',ylim=(0,max(counts)*1.2))
ax=axes[0,1]
ax.boxplot([[r['steps'] for r in records if r['task_type']==t] for t in order],tick_labels=order, showmeans=True)
ax.set(title='不同任务的步数（箱线图）',ylabel='assistant 决策轮数')
ax=axes[1,0]
for t in order:
    subset=[r for r in records if r['task_type']==t]
    ax.scatter([r['steps'] for r in subset],[r['tokens']/1024 for r in subset],s=16,alpha=.55,label=t,color=colors[t])
ax.set(title=f"步数与上下文长度（Pearson r = {summary['steps_tokens_pearson']:.2f}）",xlabel='assistant 决策轮数',ylabel='完整序列 token 数 / 1024');ax.legend()
ax=axes[1,1]
positions=np.arange(5);width=.25
for i,(key,label) in enumerate([('count','轨迹数占比'),('tokens','全部 token 占比'),('trainable_tokens','监督 token 占比')]):
    vals=[r[key] for r in summary['tasks']];ax.bar(positions+(i-1)*width,np.array(vals)/sum(vals)*100,width,label=label)
ax.set_xticks(positions,order);ax.set(title='样本数占比与 token 占比',ylabel='占比 (%)');ax.legend()
save(fig,'task_and_length_dashboard')
fig,axes=plt.subplots(2,2,figsize=(16,11),layout='constrained')
discrete(axes[0,0],'tool_calls','每条轨迹的工具调用次数','工具调用次数（并行调用分别计数）')
discrete(axes[0,1],'unique_tools','每条轨迹使用的不同工具数','不同工具名称数量（包含 skill 等基础工具）')
ax=axes[1,0]
ax.hist([r['tokens']/1024 for r in records],bins=np.arange(0,257,8),color='#42a58b',edgecolor='white')
ax.set(title='真实 tokenizer 的完整序列长度',xlabel='token 数 / 1024（每箱 8K）',ylabel='轨迹数量')
ax=axes[1,1]
top=calls.most_common(10)[::-1]
ax.barh([k.removeprefix('mcp__molclaw-scp__') for k,v in top],[v for k,v in top],color='#8a68b2')
ax.set(title='调用次数最多的 10 个工具',xlabel='调用次数');ax.tick_params(axis='y',labelsize=9)
save(fig,'tools_and_tokens_dashboard')
lines=['# 596 条训练轨迹统计', '', f'数据源：`{ROOT}`。训练文件、semantic 文件和 token gate manifest 的 SHA-256 均已核对发布清单。',
'', '## 统计口径', '',
'- 主图每一个整数步数对应一根柱。一步为一条 assistant 消息，与 semantic assistant_decision 事件逐条核对；包含 skill 调用及最终回答，不计 system、user、tool 返回。同一轮多个工具调用算一步。',
'- 工具调用数按实际 tool_calls 元素计数，包含 skill/read/write 等基础工具；同时核对 semantic 观察数和 SFT tool 消息数。',
'- token 来自发布时真实 Qwen tokenizer / qwen3_5 loss mask 的 record_lengths，只选择本训练集 596 个 ID；没有重新 tokenize。完整序列含系统、catalog、工具返回等，监督 token 以 loss mask 为准。',
'- 分析对象为清洗、技能补充和长度门控后的训练轨迹，不代表原始 teacher 执行长度。599 条 semantic 中的 3 条超长轨迹不计入统计。',
'- 分位数采用 NumPy 默认线性插值；工具报错仅统计 semantic 的 is_error=true 或 status=error，可能漏掉嵌在正文内的失败，不能当作科学答案正确率。',
'', '## 步数', '', '![步数分布](steps_histogram.png)', '', '|指标|数值|','|---|---:|']
for k,v in s.items(): lines.append(f'|{k}|{v:.2f}|')
lines += ['', '## 任务与长度', '', '![任务与长度](task_and_length_dashboard.png)', '', '|任务|轨迹数|平均步数|中位步数|总 token|监督 token|','|---|---:|---:|---:|---:|---:|']
for r in summary['tasks']: lines.append(f"|{r['task_type']}|{r['count']}|{r['mean_steps']:.2f}|{r['median_steps']:g}|{r['tokens']:,}|{r['trainable_tokens']:,}|")
lines += ['', '## 工具与 token', '', '![工具与 token](tools_and_tokens_dashboard.png)', '',
 f"- 工具调用共 {sum(calls.values()):,} 次，覆盖 {len(calls)} 种工具；加载 {len(skills)} 种 skill。",
 f"- {summary['trajectories_with_parallel_calls']} 条轨迹存在一轮多个工具调用。",
 f"- 全部 token {summary['metrics']['tokens']['total']:,}；监督 token {summary['metrics']['trainable_tokens']['total']:,}，加权监督占比 {summary['trainable_token_fraction']:.2%}。",
 f"- 最长的 60 条（约 10%）轨迹占全部 token 的 {summary['longest_10_percent_token_share']:.2%}。",
 f"- 显式报错观察 {summary['explicit_error_observations']:,} 条，分布于 {summary['trajectories_with_explicit_errors']} 条轨迹；报错可能被后续修复，不等于轨迹失败。", '', '|上下文容量|可完整容纳轨迹数|覆盖率|','|---|---:|---:|']
for n,v in summary['context_coverage'].items():lines.append(f'|{int(n):,}|{v}|{v/596:.2%}|')
lines += ['', '## 进一步可做', '',
'- 对训练采样检查任务数量与监督 token 占比的差异；按轨迹均匀采样不等于各任务监督量均衡。',
'- 利用逐条 CSV 选择短、中、长轨迹及不同工具覆盖的开发验证样本；本报告未修改数据划分。',
'- 若要评估科学质量，需要独立核验最终答案、证据支持度与失败恢复；不能从轨迹长度或工具是否返回推断正确率。',
'', '## 复现与文件', '', '`python analyze.py`（需要 numpy、matplotlib 和系统 Droid Sans Fallback 字体）。',
'`trajectory_metrics.csv` 为逐轨迹明细，`step_frequency.csv` 为每个整数步数的频数，`task_summary.csv`、`tool_usage.csv`、`skill_usage.csv` 为分类汇总，`summary.json` 包含数值和来源哈希。所有图同时导出 PNG/SVG。', '']
(OUT/'report.md').write_text('\n'.join(lines))
print(json.dumps(summary,ensure_ascii=False,indent=2))
