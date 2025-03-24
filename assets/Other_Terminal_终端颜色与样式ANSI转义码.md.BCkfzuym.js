import{_ as t,c as a,o as i,ag as d}from"./chunks/framework.Cizge8DQ.js";const c=JSON.parse('{"title":"终端颜色与样式 ANSI 转义码总结","description":"","frontmatter":{},"headers":[],"relativePath":"Other/Terminal/终端颜色与样式ANSI转义码.md","filePath":"Other/Terminal/终端颜色与样式ANSI转义码.md","lastUpdated":1742805761000}'),l={name:"Other/Terminal/终端颜色与样式ANSI转义码.md"};function e(n,s,h,p,r,k){return i(),a("div",null,s[0]||(s[0]=[d(`<h1 id="终端颜色与样式-ansi-转义码总结" tabindex="-1">终端颜色与样式 ANSI 转义码总结 <a class="header-anchor" href="#终端颜色与样式-ansi-转义码总结" aria-label="Permalink to &quot;终端颜色与样式 ANSI 转义码总结&quot;">​</a></h1><p>本文档总结了终端（如 Linux、macOS、WSL 等）中使用的 ANSI 转义码及其各种变种，包括基本颜色、亮色、背景色、256 色模式、RGB 颜色以及其他文本样式和重置颜色的用法。</p><h2 id="_1-基本颜色" tabindex="-1">1. 基本颜色 <a class="header-anchor" href="#_1-基本颜色" aria-label="Permalink to &quot;1. 基本颜色&quot;">​</a></h2><p>ANSI 颜色代码格式为：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>\\033[&lt;参数&gt;m</span></span></code></pre></div><p>其中 <code>\\033</code>（或 <code>\\e</code>）是 ESC 转义符。</p><h3 id="_1-1-前景色-文本颜色" tabindex="-1">1.1 前景色（文本颜色） <a class="header-anchor" href="#_1-1-前景色-文本颜色" aria-label="Permalink to &quot;1.1 前景色（文本颜色）&quot;">​</a></h3><table tabindex="0"><thead><tr><th>颜色</th><th>ANSI 代码</th></tr></thead><tbody><tr><td>黑色 <span style="color:black;">⬛</span></td><td><code>\\033[30m</code></td></tr><tr><td>红色 <span style="color:red;">🟥</span></td><td><code>\\033[31m</code></td></tr><tr><td>绿色 <span style="color:green;">🟩</span></td><td><code>\\033[32m</code></td></tr><tr><td>黄色 <span style="color:gold;">🟨</span></td><td><code>\\033[33m</code></td></tr><tr><td>蓝色 <span style="color:blue;">🟦</span></td><td><code>\\033[34m</code></td></tr><tr><td>洋红 <span style="color:magenta;">🟪</span></td><td><code>\\033[35m</code></td></tr><tr><td>青色 <span style="color:cyan;">🟦</span></td><td><code>\\033[36m</code></td></tr><tr><td>白色 <span style="color:lightgray;">⬜</span></td><td><code>\\033[37m</code></td></tr></tbody></table><h3 id="_1-2-背景色" tabindex="-1">1.2 背景色 <a class="header-anchor" href="#_1-2-背景色" aria-label="Permalink to &quot;1.2 背景色&quot;">​</a></h3><table tabindex="0"><thead><tr><th>颜色</th><th>ANSI 代码</th></tr></thead><tbody><tr><td>黑色 <span style="background-color:black;">⬛</span></td><td><code>\\033[40m</code></td></tr><tr><td>红色 <span style="background-color:red;">🟥</span></td><td><code>\\033[41m</code></td></tr><tr><td>绿色 <span style="background-color:green;">🟩</span></td><td><code>\\033[42m</code></td></tr><tr><td>黄色 <span style="background-color:gold;">🟨</span></td><td><code>\\033[43m</code></td></tr><tr><td>蓝色 <span style="background-color:blue;">🟦</span></td><td><code>\\033[44m</code></td></tr><tr><td>洋红 <span style="background-color:magenta;">🟪</span></td><td><code>\\033[45m</code></td></tr><tr><td>青色 <span style="background-color:cyan;">🟦</span></td><td><code>\\033[46m</code></td></tr><tr><td>白色 <span style="background-color:lightgray;">⬜</span></td><td><code>\\033[47m</code></td></tr></tbody></table><h2 id="_2-亮色-高亮-加粗" tabindex="-1">2. 亮色（高亮 / 加粗） <a class="header-anchor" href="#_2-亮色-高亮-加粗" aria-label="Permalink to &quot;2. 亮色（高亮 / 加粗）&quot;">​</a></h2><p>在基本颜色的代码基础上，还可以使用亮色（也可通过粗体 <code>\\033[1m</code> 来实现一定程度的加亮效果）。</p><p>下面的亮色代码通常也被称作“明亮”或“浅色”版本。</p><h3 id="_2-1-前景亮色" tabindex="-1">2.1 前景亮色 <a class="header-anchor" href="#_2-1-前景亮色" aria-label="Permalink to &quot;2.1 前景亮色&quot;">​</a></h3><table tabindex="0"><thead><tr><th>颜色</th><th>ANSI 代码</th></tr></thead><tbody><tr><td>亮黑色 <span style="color:dimgray;">⬛</span></td><td><code>\\033[90m</code></td></tr><tr><td>亮红色 <span style="color:#ff6666;">🟥</span></td><td><code>\\033[91m</code></td></tr><tr><td>亮绿色 <span style="color:lightgreen;">🟩</span></td><td><code>\\033[92m</code></td></tr><tr><td>亮黄色 <span style="color:khaki;">🟨</span></td><td><code>\\033[93m</code></td></tr><tr><td>亮蓝色 <span style="color:lightskyblue;">🟦</span></td><td><code>\\033[94m</code></td></tr><tr><td>亮洋红 <span style="color:violet;">🟪</span></td><td><code>\\033[95m</code></td></tr><tr><td>亮青色 <span style="color:paleturquoise;">🟦</span></td><td><code>\\033[96m</code></td></tr><tr><td>亮白色 <span style="color:white;">⬜</span></td><td><code>\\033[97m</code></td></tr></tbody></table><h3 id="_2-2-背景亮色" tabindex="-1">2.2 背景亮色 <a class="header-anchor" href="#_2-2-背景亮色" aria-label="Permalink to &quot;2.2 背景亮色&quot;">​</a></h3><table tabindex="0"><thead><tr><th>颜色</th><th>ANSI 代码</th></tr></thead><tbody><tr><td>亮黑色 <span style="background-color:dimgray;">⬛</span></td><td><code>\\033[100m</code></td></tr><tr><td>亮红色 <span style="background-color:#ff6666;">🟥</span></td><td><code>\\033[101m</code></td></tr><tr><td>亮绿色 <span style="background-color:lightgreen;">🟩</span></td><td><code>\\033[102m</code></td></tr><tr><td>亮黄色 <span style="background-color:khaki;">🟨</span></td><td><code>\\033[103m</code></td></tr><tr><td>亮蓝色 <span style="background-color:lightskyblue;">🟦</span></td><td><code>\\033[104m</code></td></tr><tr><td>亮洋红 <span style="background-color:violet;">🟪</span></td><td><code>\\033[105m</code></td></tr><tr><td>亮青色 <span style="background-color:paleturquoise;">🟦</span></td><td><code>\\033[106m</code></td></tr><tr><td>亮白色 <span style="background-color:white;">⬜</span></td><td><code>\\033[107m</code></td></tr></tbody></table><h2 id="_3-其他文本样式" tabindex="-1">3. 其他文本样式 <a class="header-anchor" href="#_3-其他文本样式" aria-label="Permalink to &quot;3. 其他文本样式&quot;">​</a></h2><table tabindex="0"><thead><tr><th>样式</th><th>ANSI 代码</th><th>说明</th></tr></thead><tbody><tr><td>复位</td><td><code>\\033[0m</code></td><td>重置所有样式</td></tr><tr><td>粗体</td><td><code>\\033[1m</code></td><td>加粗 / 高亮文本</td></tr><tr><td>暗色</td><td><code>\\033[2m</code></td><td>使文本颜色暗淡</td></tr><tr><td>斜体</td><td><code>\\033[3m</code></td><td>斜体（部分终端支持）</td></tr><tr><td>下划线</td><td><code>\\033[4m</code></td><td>添加下划线</td></tr><tr><td>闪烁</td><td><code>\\033[5m</code></td><td>使文本闪烁</td></tr><tr><td>反色</td><td><code>\\033[7m</code></td><td>前景色与背景色交换</td></tr><tr><td>隐藏文本</td><td><code>\\033[8m</code></td><td>隐藏文本</td></tr><tr><td>删除线</td><td><code>\\033[9m</code></td><td>文本显示删除线</td></tr></tbody></table><h2 id="_4-256-色模式" tabindex="-1">4. 256 色模式 <a class="header-anchor" href="#_4-256-色模式" aria-label="Permalink to &quot;4. 256 色模式&quot;">​</a></h2><p>如果终端支持 256 色，可以使用以下格式指定颜色：</p><ul><li><p>前景色：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>\\033[38;5;&lt;颜色代码&gt;m</span></span></code></pre></div></li><li><p>背景色：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>\\033[48;5;&lt;颜色代码&gt;m</span></span></code></pre></div></li></ul><p>示例：</p><ul><li><p>设置前景色为绿色（颜色代码 82）：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>\\033[38;5;82m</span></span></code></pre></div></li><li><p>设置背景色为红色（颜色代码 196）：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>\\033[48;5;196m</span></span></code></pre></div></li></ul><h2 id="_5-true-color-24-位颜色-rgb" tabindex="-1">5. True Color（24 位颜色 / RGB） <a class="header-anchor" href="#_5-true-color-24-位颜色-rgb" aria-label="Permalink to &quot;5. True Color（24 位颜色 / RGB）&quot;">​</a></h2><p>如果终端支持 True Color，可使用 24 位 RGB 颜色。格式如下：</p><ul><li><p>前景色：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>\\033[38;2;&lt;R&gt;;&lt;G&gt;;&lt;B&gt;m</span></span></code></pre></div></li><li><p>背景色：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>\\033[48;2;&lt;R&gt;;&lt;G&gt;;&lt;B&gt;m</span></span></code></pre></div></li></ul><p>示例：</p><ul><li><p>设置前景色为橙色 (RGB: 255,165,0)：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>\\033[38;2;255;165;0m</span></span></code></pre></div></li><li><p>设置背景色为青色 (RGB: 0,255,255)：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>\\033[48;2;0;255;255m</span></span></code></pre></div></li></ul><h2 id="_6-复位颜色" tabindex="-1">6. 复位颜色 <a class="header-anchor" href="#_6-复位颜色" aria-label="Permalink to &quot;6. 复位颜色&quot;">​</a></h2><p>无论使用哪种颜色或样式，都可以使用以下 ANSI 转义码来重置终端样式：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>\\033[0m</span></span></code></pre></div><h2 id="_7-示例代码" tabindex="-1">7. 示例代码 <a class="header-anchor" href="#_7-示例代码" aria-label="Permalink to &quot;7. 示例代码&quot;">​</a></h2><p>下面是一个示例程序，展示了如何在终端中使用各种 ANSI 转义码：</p><div class="language-c++ vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">c++</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">#include</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &lt;iostream&gt;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">int</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() {</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    std</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">::cout </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;&lt;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\033</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">[31m红色文本</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\033</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">[0m&quot;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> &lt;&lt;</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> std</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">::endl;</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    std</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">::cout </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;&lt;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\033</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">[32m绿色文本</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\033</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">[0m&quot;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> &lt;&lt;</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> std</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">::endl;</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    std</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">::cout </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;&lt;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\033</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">[34m蓝色文本</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\033</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">[0m&quot;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> &lt;&lt;</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> std</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">::endl;</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    std</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">::cout </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;&lt;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\033</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">[1;33m亮黄色文本</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\033</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">[0m&quot;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> &lt;&lt;</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> std</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">::endl;</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    std</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">::cout </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;&lt;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\033</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">[7m反色文本</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\033</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">[0m&quot;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> &lt;&lt;</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> std</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">::endl;</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    std</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">::cout </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;&lt;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\033</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">[4m下划线文本</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\033</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">[0m&quot;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> &lt;&lt;</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> std</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">::endl;</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    std</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">::cout </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;&lt;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\033</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">[38;5;82m256 色绿色文本</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\033</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">[0m&quot;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> &lt;&lt;</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> std</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">::endl;</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    std</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">::cout </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;&lt;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\033</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">[38;2;255;165;0mTrue Color 橙色文本</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\033</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">[0m&quot;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> &lt;&lt;</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> std</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">::endl;</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    std</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">::cout </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;&lt;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\033</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">[0m恢复正常文本&quot;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> &lt;&lt;</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> std</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">::endl;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span></code></pre></div>`,35)]))}const g=t(l,[["render",e]]);export{c as __pageData,g as default};
