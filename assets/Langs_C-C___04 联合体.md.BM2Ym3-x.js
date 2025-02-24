import{_ as a,c as i,o as n,ae as t}from"./chunks/framework.DI3LEz4j.js";const r=JSON.parse('{"title":"联合体","description":"","frontmatter":{},"headers":[],"relativePath":"Langs/C-C++/04 联合体.md","filePath":"Langs/C-C++/04 联合体.md"}'),e={name:"Langs/C-C++/04 联合体.md"};function p(l,s,h,k,d,o){return n(),i("div",null,s[0]||(s[0]=[t(`<h1 id="联合体" tabindex="-1">联合体 <a class="header-anchor" href="#联合体" aria-label="Permalink to &quot;联合体&quot;">​</a></h1><p>联合体和结构体类似，但联合体中的成员变量共享内存空间。</p><div class="language-c++ vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">c++</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">union</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> USER</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">{</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">	short</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> a;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">	int</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> b;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">	double</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> c;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">};</span></span></code></pre></div><p>变量<code>a、b、c</code>是共享内存的，其中<code>USER</code>的大小为最大成员变量的大小，8B。</p><h2 id="匿名联合体" tabindex="-1">匿名联合体 <a class="header-anchor" href="#匿名联合体" aria-label="Permalink to &quot;匿名联合体&quot;">​</a></h2><p>匿名联合体即没有名字的联合体，也可以声明匿名结构体。</p><div class="language-c++ vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">c++</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">union</span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">  // 可以声明匿名联合体（结构体也可以），匿名联合体只能声明 1 次</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">{</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">	int</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> a;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}anonymous_union;</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">// anonymous_union理解为使用该联合体的代号</span></span></code></pre></div>`,7)]))}const E=a(e,[["render",p]]);export{r as __pageData,E as default};
