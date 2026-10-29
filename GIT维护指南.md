# Git 日常维护

## 正式位置和范围

服务器工作目录是 `/home/luorongwei/pdsch_lte`，通过 VS Code Remote SSH 直接编辑。
只维护这一套源码，不另建阅读版、旧版目录或链接别名。Git 历史保存在 `.git`。
执行 `git remote -v` 查看实际绑定的 GitHub 仓库。

Git 跟踪源码、说明、依赖声明和 `reference/synthetic*.mat` 等小型运行资产。
数据、权重、日志、安装的依赖留在服务器，GitHub 不备份这些内容。
数据目录 `/data/luorongwei/pdsch_lte` 需要单独备份。

实测派生统计、源路径、采集元数据、实测分析报告及对话对接记录仅留在服务器，
不纳入代码仓库或推送历史。需要额外发布时先明确资料范围与授权。

`configs/*.json` 自动生成，默认忽略；生成规则 `tools/make_assets.py` 纳入 Git。
当次实际执行计划随数据保存在 `generation_plan.json`，应保留以供复现。
`.gitattributes` 对 MATLAB 关闭换行转换，避免改变已有生成器的原始字节摘要。

## 修改、检查、提交、上传

在已连接服务器的终端执行：

```bash
cd /home/luorongwei/pdsch_lte
git status --short
git diff
```

一次提交围绕一个明确问题。修改后做相应检查，再选择文件提交：

```bash
git add tools/make_assets.py
git diff --cached
git commit -m "调整场景采样规则"
git push
```

`add` 选择改动；`commit` 保存服务器历史；`push` 才上传 GitHub。
VS Code 左侧“源代码管理”也能完成差异查看、暂存和提交。
提交前检查清单，不要强制加入 IQ、依赖、模型权重或认证文件。

## 功能分支与合并

没有任务依赖当前源码、且工作目录干净时，创建开发分支：

```bash
git switch -c codex/improve-acquisition
```

完成修改和检查后提交，首次上传分支执行：

```bash
git push -u origin codex/improve-acquisition
```

通过 GitHub Pull Request 检查后合并到 main。分支使用同一工作目录，不复制源码文件夹。
拉回合并结果前确认任务已结束、没有未提交修改，再执行：

```bash
git switch main
git pull --ff-only
```

出现分叉或冲突时，先检查状态与历史，不用强制推送或 `reset --hard` 清掉问题。

## 运行期间与实验追溯

查看状态、差异、历史以及提交现有文件，不会自行重启仿真。
运行期间保持 MATLAB 源码、资产和任务配置稳定，不切换版本、拉取覆盖或编辑运行依赖。
一个进程可能已加载部分函数，后续又读取其他文件，中途更改可能造成版本混用。

每次新实验先提交，确认 `git status --short` 没有输出，再记录提交号与输入清单。
例如下面仅记录版本，不会启动训练：

```bash
PYTHONPATH=/home/luorongwei/pdsch_lte/.deps python3 tools/record_run.py \
  --manifest /data/luorongwei/pdsch_lte/production/manifest.csv \
  --output /data/luorongwei/pdsch_lte/training_runs/experiment_001/run_info.json
```

每次使用新的实验目录。工具默认拒绝未提交状态和覆盖历史记录。
`--allow-dirty` 只供调试，会明确记录状态，不能声称完全由提交号复现。
新版生成启动器只在阶段首次启动时冻结计划，续跑只读批次内的 `generation_plan.json`。
默认第三阶段 2,000 份；新批次在 `/data/luorongwei/pdsch_lte/batches/<批次名>/`。
任务参数摘要与生成源码/运行环境一起核对，变化时使用新批次名，禁止覆盖旧批次。
Git 纳管前已经开始的批次，不能事后声称由新提交启动。

## 查看和恢复历史

```bash
git log --oneline -n 10
git show 提交号 --stat
git diff 旧提交号 新提交号 -- matlab/lte_dataset_generate.m
```

对已推送的错误提交，优先使用 `git revert 提交号` 新增反向修改，保留历史。
先保持工作目录干净，确认没有任务依赖将被修改的源码。
`git restore 路径` 会丢弃文件的未提交修改，只在确定不需要时使用。

## GitHub 认证

电脑到服务器的 Remote SSH，与服务器到 GitHub 的认证是两条独立连接。
连接工具代为上传，不会自动让服务器终端获得日后的 `git push` 权限。
若提示未认证，使用平台官方登录流程；不要把密码、令牌或私钥写入源码、远程 URL、聊天或日志。
网络或认证临时不可用时，先保留已提交历史，恢复后再推送。

## 离线交付

普通维护使用 Git。确实需要压缩包时，显式指定工程外路径：

```bash
python3 tools/package_project.py \
  --output /data/luorongwei/pdsch_lte/pdsch_code_delivery.zip
```

工具用 `git archive` 导出当前干净提交并拒绝覆盖已有文件。
不包含 `.git`、未跟踪数据、忽略的依赖，也不会自动另复制一套训练源码。
