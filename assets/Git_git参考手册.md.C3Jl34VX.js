import{_ as i,c as a,o as t,ae as e}from"./chunks/framework.BHrE6nLq.js";const g=JSON.parse('{"title":"git 参考手册","description":"","frontmatter":{},"headers":[],"relativePath":"Git/git参考手册.md","filePath":"Git/git参考手册.md"}'),h={name:"Git/git参考手册.md"};function n(l,s,p,k,d,r){return t(),a("div",null,s[0]||(s[0]=[e(`<h1 id="git-参考手册" tabindex="-1">git 参考手册 <a class="header-anchor" href="#git-参考手册" aria-label="Permalink to &quot;git 参考手册&quot;">​</a></h1><h2 id="基础指令" tabindex="-1">基础指令 <a class="header-anchor" href="#基础指令" aria-label="Permalink to &quot;基础指令&quot;">​</a></h2><h3 id="创建分支" tabindex="-1">创建分支 <a class="header-anchor" href="#创建分支" aria-label="Permalink to &quot;创建分支&quot;">​</a></h3><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">git</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> branch</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> newBranch</span></span></code></pre></div><h3 id="查看当前绑定的远程仓库" tabindex="-1">查看当前绑定的远程仓库 <a class="header-anchor" href="#查看当前绑定的远程仓库" aria-label="Permalink to &quot;查看当前绑定的远程仓库&quot;">​</a></h3><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">git</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> remote</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> -v</span></span></code></pre></div><h3 id="修改当前绑定的远程仓库" tabindex="-1">修改当前绑定的远程仓库 <a class="header-anchor" href="#修改当前绑定的远程仓库" aria-label="Permalink to &quot;修改当前绑定的远程仓库&quot;">​</a></h3><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">git</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> remote</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> set-url</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> origin</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> &lt;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">新远程仓库UR</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">L</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span></span></code></pre></div><h3 id="删除当前绑定的远程仓库" tabindex="-1">删除当前绑定的远程仓库 <a class="header-anchor" href="#删除当前绑定的远程仓库" aria-label="Permalink to &quot;删除当前绑定的远程仓库&quot;">​</a></h3><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">git</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> remote</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> remove</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> origin</span></span></code></pre></div><h3 id="查看当前-git-绑定的邮箱和用户名-全局和当前环境" tabindex="-1">查看当前 Git 绑定的邮箱和用户名（全局和当前环境） <a class="header-anchor" href="#查看当前-git-绑定的邮箱和用户名-全局和当前环境" aria-label="Permalink to &quot;查看当前 Git 绑定的邮箱和用户名（全局和当前环境）&quot;">​</a></h3><ul><li><strong>查看全局邮箱和用户名</strong>：</li></ul><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">git</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> config</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> --global</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> user.name</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">git</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> config</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> --global</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> user.email</span></span></code></pre></div><ul><li><strong>查看当前项目的邮箱和用户名</strong>：</li></ul><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">git</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> config</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> user.name</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">git</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> config</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> user.email</span></span></code></pre></div><h3 id="修改当前-git-绑定的邮箱和用户名-全局和当前环境" tabindex="-1">修改当前 Git 绑定的邮箱和用户名（全局和当前环境） <a class="header-anchor" href="#修改当前-git-绑定的邮箱和用户名-全局和当前环境" aria-label="Permalink to &quot;修改当前 Git 绑定的邮箱和用户名（全局和当前环境）&quot;">​</a></h3><ul><li><strong>修改全局邮箱和用户名</strong>：</li></ul><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">git</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> config</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> --global</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> user.name</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Your Global Name&quot;</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">git</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> config</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> --global</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> user.email</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;your-global-email@example.com&quot;</span></span></code></pre></div><ul><li><strong>修改当前项目邮箱和用户名</strong>：</li></ul><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">git</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> config</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> user.name</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Your Local Name&quot;</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">git</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> config</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> user.email</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;your-local-email@example.com&quot;</span></span></code></pre></div><h3 id="提交当前主分支到远程-一般为-main-或-master" tabindex="-1">提交当前主分支到远程（一般为 <code>main</code> 或 <code>master</code>） <a class="header-anchor" href="#提交当前主分支到远程-一般为-main-或-master" aria-label="Permalink to &quot;提交当前主分支到远程（一般为 \`main\` 或 \`master\`）&quot;">​</a></h3><ul><li><strong>提交更改</strong>（添加所有更改并提交）：</li></ul><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">git</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> add</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> .</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">git</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> commit</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> -m</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Your commit message&quot;</span></span></code></pre></div><ul><li><strong>推送主分支到远程</strong>：</li></ul><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">git</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> push</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> origin</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> main</span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">  # 如果主分支是 main</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">git</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> push</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> origin</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> master</span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">  # 如果主分支是 master</span></span></code></pre></div><h3 id="推送当前新分支到远程" tabindex="-1">推送当前新分支到远程 <a class="header-anchor" href="#推送当前新分支到远程" aria-label="Permalink to &quot;推送当前新分支到远程&quot;">​</a></h3><ul><li><strong>检查当前所在分支</strong>：</li></ul><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">git</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> branch</span></span></code></pre></div><ul><li><strong>如果需要推送当前分支到远程</strong>：</li></ul><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">git</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> push</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> --set-upstream</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> origin</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> &lt;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">当前分支</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">名</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span></span></code></pre></div><p><strong>示例</strong>：当前分支名为 <code>new-feature</code>，将其推送到远程：</p><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">git</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> push</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> --set-upstream</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> origin</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> new-feature</span></span></code></pre></div><ul><li>推送完成后，后续只需用 <code>git push</code> 即可更新该分支。</li></ul><h3 id="拉取远程更新" tabindex="-1">拉取远程更新 <a class="header-anchor" href="#拉取远程更新" aria-label="Permalink to &quot;拉取远程更新&quot;">​</a></h3><p>执行以下命令以拉取远程分支的更新并与本地分支合并：</p><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">git</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> pull</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> origin</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> &lt;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">branch_nam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">e</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span></span></code></pre></div><ul><li>可能情况： <ul><li><strong>无冲突</strong>： Git 会自动合并远程和本地的更改。</li><li><strong>有冲突</strong>： 如果本地和远程对同一文件的相同部分进行了修改，Git 会提示冲突，需要手动解决。</li></ul></li></ul><h3 id="使用-ssh-管理仓库" tabindex="-1">使用 SSH 管理仓库 <a class="header-anchor" href="#使用-ssh-管理仓库" aria-label="Permalink to &quot;使用 SSH 管理仓库&quot;">​</a></h3><p>机器和 git 仓库账号中需要有公私钥配对，需要在本地机器上生成密钥对，然后将公钥内容添加到账户中。</p><p>在本地机器生成密钥对：</p><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">ssh-keygen</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> -t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> rsa</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> -b</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 4096</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> -C</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;your-email@qq.com&quot;</span></span></code></pre></div><p>这回生成 <code>id_rsa</code> 和 <code>id_rsa.pub</code> 文件，其中 <code>id_rsa.pub</code> 是公钥，将其复制到账户下。</p><p>查看公钥内容：</p><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">cat</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> ~/.ssh/id_rsa.pub</span></span></code></pre></div><h2 id="克隆远程仓库" tabindex="-1">克隆远程仓库 <a class="header-anchor" href="#克隆远程仓库" aria-label="Permalink to &quot;克隆远程仓库&quot;">​</a></h2><h3 id="使用代理克隆远程仓库" tabindex="-1">使用代理克隆远程仓库 <a class="header-anchor" href="#使用代理克隆远程仓库" aria-label="Permalink to &quot;使用代理克隆远程仓库&quot;">​</a></h3><p>要在使用 <code>git clone</code> 命令时通过代理指定端口（例如7890端口），你可以通过设置 <code>git</code> 的代理配置来实现。可以在命令行中设置代理，方法如下：</p><h4 id="临时设置代理" tabindex="-1">临时设置代理 <a class="header-anchor" href="#临时设置代理" aria-label="Permalink to &quot;临时设置代理&quot;">​</a></h4><p>在 <code>git clone</code> 命令之前，临时设置 HTTP 和 HTTPS 代理。例如使用 <code>7897</code> 端口代理：</p><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">git</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> -c</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> http.proxy=http://127.0.0.1:7897</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> -c</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> https.proxy=http://127.0.0.1:7897</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> clone</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> --recurse-submodules</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> https://github.com/mikel-brostrom/Yolov7_StrongSORT_OSNet.git</span></span></code></pre></div><h4 id="永久设置代理" tabindex="-1">永久设置代理 <a class="header-anchor" href="#永久设置代理" aria-label="Permalink to &quot;永久设置代理&quot;">​</a></h4><p>如果你希望在所有 Git 操作中都使用该代理，可以配置 Git 的全局代理：</p><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">git</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> config</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> --global</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> http.proxy</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> http://127.0.0.1:7890</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">git</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> config</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> --global</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> https.proxy</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> http://127.0.0.1:7890</span></span></code></pre></div><p>这样，你的所有 Git 操作都会自动通过该代理。若要取消代理设置：</p><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">git</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> config</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> --global</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> --unset</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> http.proxy</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">git</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> config</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> --global</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> --unset</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> https.proxy</span></span></code></pre></div><p>这些命令应该能帮助你在执行 <code>git clone</code> 时通过指定的代理端口进行操作。</p><h2 id="其他功能指令" tabindex="-1">其他功能指令 <a class="header-anchor" href="#其他功能指令" aria-label="Permalink to &quot;其他功能指令&quot;">​</a></h2><h3 id="查看仓库大小" tabindex="-1">查看仓库大小 <a class="header-anchor" href="#查看仓库大小" aria-label="Permalink to &quot;查看仓库大小&quot;">​</a></h3><p>一般情况下，可以直接到网页端的仓库管理界面查询到仓库的信息，包含仓库大小，如果无法查到仓库大小，可以将仓库克隆到本地，然后运行：</p><div class="language-cmd vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">cmd</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">git count</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">objects </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">vH</span></span></code></pre></div><p>查看仓库大小。</p><p>输出形式：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>count: 123</span></span>
<span class="line"><span>size: 1.23 MiB</span></span>
<span class="line"><span>in-pack: 0</span></span>
<span class="line"><span>packs: 0</span></span>
<span class="line"><span>size-pack: 0 bytes</span></span>
<span class="line"><span>prune-packable: 0</span></span>
<span class="line"><span>garbage: 0</span></span>
<span class="line"><span>size-garbage: 0 bytes</span></span></code></pre></div>`,63)]))}const c=i(h,[["render",n]]);export{g as __pageData,c as default};
