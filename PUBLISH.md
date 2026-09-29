# 推送方法：把镜头 MTF 仿真台发布到 GitHub

写给以后接手的人和 agent。照着做就能把本地的新版本推到线上，不用再摸索。

## 仓库里放的是什么

- **GitHub 仓库** `anvcor/lens-bench`（公开，开了 GitHub Pages）放的是**构建产物**，不是源码。
- **源码**在本机 `E:\Download\lens-bench`（`src/`、`tools/`、`lenses/`、`docs/`），这个目录**不是** git 仓库。
- **站点仓库的本地副本**在 `E:\Download\lens-bench-site`，`tools/publish.py` 会自动维护；没有的话它会自己 clone。

两边文件的对应关系：

| 站点仓库 | 来自本地 | 说明 |
|---|---|---|
| `index.html` | `dist/web/index.html` | 三行 Google Fonts `<link>` 换成自托管字体块 |
| `js/*`、`data/*` | `dist/web/js`、`dist/web/data` | 原样镜像，下架的镜头会被删掉 |
| `standalone.html` | `dist/lens-mtf-bench.html` | 单文件版，字体换成内嵌 base64 |
| `fonts/*.woff2` | 只在站点仓库里有 | 不动 |
| `README.md` | `README.md` | 原样 |
| `PUBLISH.md` | `docs/推送方法.md` | 就是本文 |

## 一条命令推送

```bash
node tools/build.js
python tools/publish.py -m "新增 X 颗：……"
```

`publish.py` 依次做这几件事：

1. 拉取站点仓库最新版本（`pull --ff-only`）。
2. 从站点现有的 `index.html` 和 `standalone.html` 里取出字体块。
3. 用 `dist/` 覆盖站点仓库，并把字体块贴回去。
4. 提交，然后推送。

先想看改了什么，可以这样跑：

- `--dry-run`：只同步文件并列出 `git status`，不提交。
- `--no-push`：只提交，不推送。

**本机的特殊情况**（agent 注意）：

- `node` 和 `git` 都**不在 PATH 上**。
  - node：`E:\Visual Studio\MSBuild\Microsoft\VisualStudio\NodeJs\node.exe`。
  - git：`E:\Visual Studio\Common7\IDE\CommonExtensions\Microsoft\TeamFoundation\Team Explorer\Git\cmd\git.exe`。`publish.py` 会自己找到它。
- Python 在 PATH 上。

## 登录（只需要一次）

- 这份 git 配了 **Git Credential Manager**（`credential.helper = manager`）。
- **第一次推送时会弹出 GitHub 登录窗口**，由用户本人登录 anvcor 账号。登录之后凭据存在 Windows 凭据管理器里，以后推送就不再问了。
- agent 不要替用户输入密码或 token，也不要设置 `GIT_TERMINAL_PROMPT=0`。设了这个变量，GCM 就弹不出登录窗口，推送会直接失败，报「could not read Username」。
- **如果推送失败**，把报错原文告诉用户，请用户在自己的终端里跑一次 `git push`（在 `E:\Download\lens-bench-site` 目录下）完成登录。

## 备用办法：Chrome 网页上传（git 用不了时）

用 Claude in Chrome 打开用户已登录的 GitHub，页面地址是 `https://github.com/anvcor/lens-bench/upload/main/<目录>`，逐个目录上传。几个坑：

- **一次最多 100 个文件。** 文件选择框会丢掉子目录，所以要按目录分开传：`data/lenses`、`data`、根目录各传一次。
- **提交说明别手打带「/」的内容。** 「/」会触发 GitHub 的搜索快捷键，要用 JS 直接给输入框赋值。
- **点完 Commit changes 后别马上跳页。** 页面显示「Processing your files…」时离开，这次提交就没了。要等页面跳回仓库首页再继续。
- **按钮点不动时换一种点法。** 用元素 ref 点 Commit 按钮有时没反应，改用坐标点。
- **传完要核对。** 在站点仓库里 `git fetch`，然后 `git diff HEAD origin/main` 应该为空。

## 字体块

- `tools/build.js` 产出的页面用的是 Google Fonts；线上换成了自托管字体。
- 字体块以 `/* 自托管字体` 开头，放在 `<style>` 里 `:root{` 的前面。
- `publish.py` 每次都从站点**当前**的文件里把这段原样取出来、再贴回新页面，所以源码里不需要保存字体文件。
- 找不到这段，或者 `dist/` 的 `<head>` 结构变了，脚本都会**直接停下、不覆盖**，这时先看是谁改了模板。
- 验证方法：`dist/` 的 `index.html` 没有改动时，贴回字体块后的 `index.html` 应该和线上**逐字节相同**。

## 推送前后的检查

- **推送前**：`node tools/import.js` 的几何自检不报新的越界；`lenses/meta.json` 里补好 brand、name、origin、originNote；跑过 `node tools/build.js`。
- **推送后**：GitHub Pages 一两分钟后更新。打开页面，确认新镜头出现在下拉里、控制台没有报错。
