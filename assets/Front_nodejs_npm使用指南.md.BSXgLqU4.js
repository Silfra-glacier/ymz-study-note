import{_ as i,c as a,o as t,ae as p}from"./chunks/framework.DI3LEz4j.js";const c=JSON.parse('{"title":"npm 使用指南","description":"","frontmatter":{},"headers":[],"relativePath":"Front/nodejs/npm使用指南.md","filePath":"Front/nodejs/npm使用指南.md","lastUpdated":1740398526000}'),n={name:"Front/nodejs/npm使用指南.md"};function e(h,s,l,d,k,o){return t(),a("div",null,s[0]||(s[0]=[p(`<h1 id="npm-使用指南" tabindex="-1">npm 使用指南 <a class="header-anchor" href="#npm-使用指南" aria-label="Permalink to &quot;npm 使用指南&quot;">​</a></h1><h2 id="镜像源" tabindex="-1">镜像源 <a class="header-anchor" href="#镜像源" aria-label="Permalink to &quot;镜像源&quot;">​</a></h2><h3 id="查看是否使用镜像源" tabindex="-1">查看是否使用镜像源 <a class="header-anchor" href="#查看是否使用镜像源" aria-label="Permalink to &quot;查看是否使用镜像源&quot;">​</a></h3><p>可以通过以下命令检查当前的 npm 配置的 <strong>registry（镜像源）</strong>：</p><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">npm</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> config</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> get</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> registry</span></span></code></pre></div><p><strong>返回示例</strong>：</p><ul><li><p>默认情况下，官方源是：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>https://registry.npmjs.org/</span></span></code></pre></div></li><li><p>如果使用淘宝镜像源，返回会是：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>https://registry.npmmirror.com/</span></span></code></pre></div></li></ul><hr><h3 id="设置淘宝镜像源" tabindex="-1">设置淘宝镜像源 <a class="header-anchor" href="#设置淘宝镜像源" aria-label="Permalink to &quot;设置淘宝镜像源&quot;">​</a></h3><p>使用以下命令将 npm 的镜像源设置为淘宝镜像：</p><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">npm</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> config</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> set</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> registry</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> https://registry.npmmirror.com/</span></span></code></pre></div><p><strong>验证设置</strong>： 执行 <code>npm config get registry</code>，确保输出为：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>https://registry.npmmirror.com/</span></span></code></pre></div><hr><h2 id="代理" tabindex="-1">代理 <a class="header-anchor" href="#代理" aria-label="Permalink to &quot;代理&quot;">​</a></h2><h3 id="设置-7897-端口代理" tabindex="-1">设置 7897 端口代理 <a class="header-anchor" href="#设置-7897-端口代理" aria-label="Permalink to &quot;设置 7897 端口代理&quot;">​</a></h3><p>要让 npm 使用 7897 端口代理（比如一个 HTTP/SOCKS 代理），需要配置代理信息。</p><h4 id="临时设置-当前会话有效" tabindex="-1">临时设置（当前会话有效） <a class="header-anchor" href="#临时设置-当前会话有效" aria-label="Permalink to &quot;临时设置（当前会话有效）&quot;">​</a></h4><p>在运行 npm 命令时设置代理，可以使用以下命令：</p><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">npm</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> --proxy</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> http://127.0.0.1:7897</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> --https-proxy</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> http://127.0.0.1:7897</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> install</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> &lt;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">package-nam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">e</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span></span></code></pre></div><p><strong>解释</strong>：</p><ul><li><code>--proxy</code>：设置 HTTP 代理。</li><li><code>--https-proxy</code>：设置 HTTPS 代理。</li><li><code>http://127.0.0.1:7890</code>：本地代理的地址和端口。</li></ul><hr><h4 id="全局设置-默认都走-7897-端口代理" tabindex="-1">全局设置（默认都走 7897 端口代理） <a class="header-anchor" href="#全局设置-默认都走-7897-端口代理" aria-label="Permalink to &quot;全局设置（默认都走 7897 端口代理）&quot;">​</a></h4><p>通过 <code>npm config</code> 设置全局代理：</p><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">npm</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> config</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> set</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> proxy</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> http://127.0.0.1:7897</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">npm</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> config</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> set</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> https-proxy</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> http://127.0.0.1:7897</span></span></code></pre></div><p><strong>验证设置</strong>： 运行以下命令检查代理设置是否生效：</p><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">npm</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> config</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> get</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> proxy</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">npm</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> config</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> get</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> https-proxy</span></span></code></pre></div><hr><h3 id="取消代理" tabindex="-1">取消代理 <a class="header-anchor" href="#取消代理" aria-label="Permalink to &quot;取消代理&quot;">​</a></h3><p>如果想移除代理设置，执行：</p><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">npm</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> config</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> delete</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> proxy</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">npm</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> config</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> delete</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> https-proxy</span></span></code></pre></div><h3 id="pnpm-代理" tabindex="-1">pnpm 代理 <a class="header-anchor" href="#pnpm-代理" aria-label="Permalink to &quot;pnpm 代理&quot;">​</a></h3><p><code>pnpm</code> 命令不直接支持 <code>--proxy</code> 和 <code>--https-proxy</code> 参数，因为这些参数在 <code>npm</code> 中是可用的，<code>pnpm</code> 需要使用全局配置的方法。</p><p>你需要在 <strong>pnpm 配置文件</strong> 中设置代理。</p><h4 id="设置全局代理" tabindex="-1">设置全局代理 <a class="header-anchor" href="#设置全局代理" aria-label="Permalink to &quot;设置全局代理&quot;">​</a></h4><p>使用以下命令为 <code>pnpm</code> 配置 HTTP 和 HTTPS 代理：</p><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">pnpm</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> config</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> set</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> proxy</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> http://127.0.0.1:7897</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">pnpm</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> config</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> set</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> https-proxy</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> http://127.0.0.1:7897</span></span></code></pre></div><p><strong>解释</strong>：</p><ul><li><code>proxy</code>：设置 HTTP 请求的代理。</li><li><code>https-proxy</code>：设置 HTTPS 请求的代理。</li></ul><hr><p><strong>验证代理设置</strong>：</p><p>运行以下命令检查代理设置是否生效：</p><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">pnpm</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> config</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> get</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> proxy</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">pnpm</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> config</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> get</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> https-proxy</span></span></code></pre></div><p>确认输出为：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>http://127.0.0.1:7897</span></span></code></pre></div><hr><h4 id="取消代理-1" tabindex="-1">取消代理 <a class="header-anchor" href="#取消代理-1" aria-label="Permalink to &quot;取消代理&quot;">​</a></h4><p>如果不需要代理了，可以清除代理设置：</p><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">pnpm</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> config</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> delete</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> proxy</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">pnpm</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> config</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> delete</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> https-proxy</span></span></code></pre></div>`,50)]))}const g=i(n,[["render",e]]);export{c as __pageData,g as default};
