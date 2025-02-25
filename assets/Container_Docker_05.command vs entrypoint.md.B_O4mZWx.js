import{_ as o,c as t,o as a,ae as s}from"./chunks/framework.DI3LEz4j.js";const u=JSON.parse('{"title":"05.command vs entrypoint","description":"","frontmatter":{},"headers":[],"relativePath":"Container/Docker/05.command vs entrypoint.md","filePath":"Container/Docker/05.command vs entrypoint.md"}'),n={name:"Container/Docker/05.command vs entrypoint.md"};function i(d,e,c,r,p,l){return a(),t("div",null,e[0]||(e[0]=[s('<h1 id="_05-command-vs-entrypoint" tabindex="-1">05.command vs entrypoint <a class="header-anchor" href="#_05-command-vs-entrypoint" aria-label="Permalink to &quot;05.command vs entrypoint&quot;">​</a></h1><h2 id="容器和虚拟机" tabindex="-1">容器和虚拟机 <a class="header-anchor" href="#容器和虚拟机" aria-label="Permalink to &quot;容器和虚拟机&quot;">​</a></h2><p>容器不仅旨在承载操作系统，而是专注于执行一个过程，或者只是分析计算。</p><p>当容器内的进程运行完毕时，容器也随之退出，<strong>容器和容器内的进程的生命周期是相同的</strong>，如果容器内部的应用程序停止或者崩溃，则容器也会退出。</p><p>所以，谁来定义容器内运行的应用程序？</p><p>在一般的 Dockerfile 中，最后一般会有一个 <code>CMD</code> 指令，如 <code>CMD [&quot;nginx&quot;]</code> ，该命令将定义在容器内运行的进程。</p><h2 id="cmd-指令" tabindex="-1"><code>CMD</code> 指令 <a class="header-anchor" href="#cmd-指令" aria-label="Permalink to &quot;`CMD` 指令&quot;">​</a></h2><p><code>CMD</code> 指令中的每个参数需要单独分隔：</p><p>✔ <code>CMD [&quot;sleep&quot;, &quot;5&quot;]</code></p><p>❌ <code>CMD [&quot;sleep 5&quot;]</code></p><p>在启动容器时，显式指定参数将覆盖掉 <code>CMD</code> 中指定的参数。</p><h2 id="entrypoinyt-指令" tabindex="-1"><code>ENTRYPOINYT</code> 指令 <a class="header-anchor" href="#entrypoinyt-指令" aria-label="Permalink to &quot;`ENTRYPOINYT` 指令&quot;">​</a></h2><p>和 <code>CMD</code> 指令不同， <code>ENTRYPOINT</code> 指令将指定一个运行的程序，而可以不指定参数，如 <code>ENTRYPOINT [&quot;sleep&quot;]</code> ，当运行容器时，在运行指令后加上参数，即可表示在 <code>ENTRYPOINT</code> 指令后<strong>附加</strong>的参数。</p><p>如 Dockerfile 中指定了 <code>ENTRYPOINT [&quot;sleep&quot;]</code> ，在启动容器时，使用指令 <code>docker run ubuntu-sleeper 10</code> ，则相当于在 Dockerfile 中指定了 <code>CMD [&quot;sleep&quot;, &quot;10&quot;]</code> 。</p><p>如果在运行容器时不指定附加参数，将会出现<strong>参数缺失</strong>的错误，因此，要指定<strong>默认值</strong>，需要 <code>ENTRYPOINT</code> 和 <code>CMD</code> 结合使用：</p><div class="language-dockerfile vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">dockerfile</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">FROM</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ubuntu</span></span>\n<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">ENTRYPOINT</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;sleep&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>\n<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">CMD</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;5&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span></code></pre></div><p>这样，当不显式指定附加参数时，将 sleep 5s ，在显式指定参数时，如 <code>docker run ubuntu-sleeper 10</code> ，则最后的 <code>CMD</code> 参数<strong>将被替换</strong>，sleep 10s 。</p><h3 id="修改-entrypoint-入口程序" tabindex="-1">修改 <code>ENTRYPOINT</code> 入口程序 <a class="header-anchor" href="#修改-entrypoint-入口程序" aria-label="Permalink to &quot;修改 `ENTRYPOINT` 入口程序&quot;">​</a></h3><p>如果在运行容器时还需要修改 <code>ENTRYPOINT</code> 的入口程序，需要参数 <code>--entrypoint</code> ，如 <code>docker run --entrypoint sleep2.0 ubuntu-sleeper 10</code> ，这将会使用 <code>sleep2.0</code> 作为入口程序。</p>',19)]))}const k=o(n,[["render",i]]);export{u as __pageData,k as default};
