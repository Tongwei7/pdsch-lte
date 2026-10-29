param(
    [string]$Server = "luorongwei@10.161.4.39"
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path $PSScriptRoot -Parent
$bundlePath = Join-Path (Split-Path $projectRoot -Parent) ".runtime\first_training_update.zip"
if (-not (Test-Path -LiteralPath $bundlePath)) { throw "找不到更新包：$bundlePath" }
$bundleHash = (Get-FileHash -LiteralPath $bundlePath -Algorithm SHA256).Hash.ToLowerInvariant()
$remoteArchive = "/tmp/pdsch_first_training_$bundleHash.zip"
Write-Host "上传项目更新包。SSH 若询问服务器密码，请在你自己的终端输入。"
& scp $bundlePath "${Server}:$remoteArchive"
if ($LASTEXITCODE -ne 0) { throw "上传失败，未修改服务器源码。" }
# 仅传递固定程序和十六进制摘要；不保存密码、私钥或修改 SSH 认证配置。
$remoteCommand = "python3 -c 'import sys,zipfile; exec(compile(zipfile.ZipFile(sys.argv[1]).read(`"tools/deploy_first_training.py`"),`"deploy_first_training.py`",`"exec`"))' $remoteArchive $bundleHash"
& ssh $Server $remoteCommand
if ($LASTEXITCODE -ne 0) { throw "部署或启动未完成，请查看上面的具体错误。" }
Write-Host "部署命令执行完成。后台任务不依赖本机终端保持连接，实际结果请查看输出的日志和状态文件。"
