import{_ as i,c as a,o as n,ae as t}from"./chunks/framework.DI3LEz4j.js";const E=JSON.parse('{"title":"03.函数","description":"","frontmatter":{},"headers":[],"relativePath":"Langs/Rust/03.函数.md","filePath":"Langs/Rust/03.函数.md"}'),p={name:"Langs/Rust/03.函数.md"};function l(e,s,h,k,r,d){return n(),a("div",null,s[0]||(s[0]=[t(`<h1 id="_03-函数" tabindex="-1">03.函数 <a class="header-anchor" href="#_03-函数" aria-label="Permalink to &quot;03.函数&quot;">​</a></h1><h2 id="函数的声明" tabindex="-1">函数的声明 <a class="header-anchor" href="#函数的声明" aria-label="Permalink to &quot;函数的声明&quot;">​</a></h2><p>函数使用 <code>fn</code> 关键字声明。</p><h2 id="函数的命名规范" tabindex="-1">函数的命名规范 <a class="header-anchor" href="#函数的命名规范" aria-label="Permalink to &quot;函数的命名规范&quot;">​</a></h2><p>rust 使用 snake case 命名规范：</p><ul><li>所有字母均为小写，单词之间使用下划线分隔</li></ul><p>示例：</p><div class="language-rust vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">rust</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">fn</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() {</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    println!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;Hello, world!&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">);</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    another_function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">();</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">fn</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> another_function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() {</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    println!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;this is another function&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span></code></pre></div><div class="note custom-block github-alert"><p class="custom-block-title">NOTE</p><p></p><p>rust 中函数的声明位置不固定，不一定要求调用的位置在声明的位置之后，只要改函数可被调用，即可在任意位置声明，如上。</p></div><h2 id="函数参数" tabindex="-1">函数参数 <a class="header-anchor" href="#函数参数" aria-label="Permalink to &quot;函数参数&quot;">​</a></h2><p>函数的参数分为 2 种：parameters（形参）和 arguments（实参），其中，parameters 是声明函数时声明的参数，arguments 是在调用函数时传入的参数。</p><p>在声明形参时，必须给出参数的类型：</p><div class="language-rust vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">rust</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">fn</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() {</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    another_function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">fn</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> another_function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> i32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) {</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    println!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;x is {}&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, x);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span></code></pre></div><h2 id="语句和表达式" tabindex="-1">语句和表达式 <a class="header-anchor" href="#语句和表达式" aria-label="Permalink to &quot;语句和表达式&quot;">​</a></h2><p>rust 是基于表达式的语言。</p><p>语句不会返回值，表达式会返回值。</p><p>基于上述原理，不能将语句赋值，因为语句没有返回值。</p><p>每一行代码大部分都可以理解为一个语句，而语句中的某些数据可以理解为表达式，如 <code>let a = 3;</code> 中，<code>3</code> 就是表达式（字面值），整行代码 <code>let a = 3;</code> 就是语句；在 <code>let a = 5 + 6;</code> 中，<code>5 + 6</code> 就是表达式。</p><p>可以进行如下示例：</p><div class="language-rust vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">rust</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">let</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    let</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 5</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">};</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">println!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;y is {}&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, y)</span></span></code></pre></div><p>由于 <code>x + 5</code> 没有以分号结尾，因此是一个表达式，产生返回值，为 6，因此，<code>y</code> 将获得返回值 6；如果在 <code>x + 5</code> 后面加上分号，则变为语句，语句没有返回值，因此 <code>y</code> 会报错：</p><div class="language-rust vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">rust</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">let</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    let</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">};</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">println!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;y is {}&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, y)</span></span></code></pre></div><p>编译时报错：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>error[E0277]: \`()\` doesn&#39;t implement \`std::fmt::Display\`</span></span>
<span class="line"><span> --&gt; src\\main.rs:8:25</span></span>
<span class="line"><span>  |</span></span>
<span class="line"><span>6 |         x + 5;</span></span>
<span class="line"><span>  |              - help: remove this semicolon</span></span>
<span class="line"><span>7 |     };</span></span>
<span class="line"><span>8 |     println!(&quot;y is {}&quot;, y)</span></span>
<span class="line"><span>  |                         ^ \`()\` cannot be formatted with the default formatter</span></span>
<span class="line"><span>  |</span></span>
<span class="line"><span>  = help: the trait \`std::fmt::Display\` is not implemented for \`()\`</span></span>
<span class="line"><span>  = note: in format strings you may be able to use \`{:?}\` (or {:#?} for pretty-print) instead</span></span>
<span class="line"><span>  = note: this error originates in the macro \`$crate::format_args_nl\` which comes from the expansion of the macro \`println\` (in Nightly builds, run with -Z macro-backtrace for more info)</span></span></code></pre></div><p>如果函数没有最终的返回值，默认的返回值为一个空元组：<code>()</code></p><h2 id="函数返回值注解" tabindex="-1">函数返回值注解 <a class="header-anchor" href="#函数返回值注解" aria-label="Permalink to &quot;函数返回值注解&quot;">​</a></h2><p>和 Python 一样，使用箭头符号 <code>-&gt;</code> 为函数的返回值指定类型：</p><div class="language-rust vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">rust</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">fn</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> five</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> i32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    5</span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">  // 这里没有使用分号，是一个表达式，也是这个函数最终的返回值</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span></code></pre></div>`,28)]))}const c=i(p,[["render",l]]);export{E as __pageData,c as default};
