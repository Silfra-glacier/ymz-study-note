import{_ as i,c as a,o as t,ae as h}from"./chunks/framework.DI3LEz4j.js";const E=JSON.parse('{"title":"02.数据类型","description":"","frontmatter":{},"headers":[],"relativePath":"CS-Langs/Rust/02.数据类型.md","filePath":"CS-Langs/Rust/02.数据类型.md","lastUpdated":1740753730000}'),n={name:"CS-Langs/Rust/02.数据类型.md"};function l(p,s,e,k,d,r){return t(),a("div",null,s[0]||(s[0]=[h(`<h1 id="_02-数据类型" tabindex="-1">02.数据类型 <a class="header-anchor" href="#_02-数据类型" aria-label="Permalink to &quot;02.数据类型&quot;">​</a></h1><h2 id="数据类型概述" tabindex="-1">数据类型概述 <a class="header-anchor" href="#数据类型概述" aria-label="Permalink to &quot;数据类型概述&quot;">​</a></h2><ul><li>在 rust 中，数据类型主要有<strong>标量</strong>和<strong>复合</strong>类型</li><li>rust 是静态的编译型语言，在编译时需要知道所有的变量的类型，因此： <ul><li>一般情况下， rust 编译器可以推断出绝大多数的变量类型</li><li>在同一个变量可能具有的变量类型种类比较多时，必须要为变量<strong>指定类型标注</strong>，否则会导致编译错误</li></ul></li></ul><h2 id="指定类型标注" tabindex="-1">指定类型标注 <a class="header-anchor" href="#指定类型标注" aria-label="Permalink to &quot;指定类型标注&quot;">​</a></h2><p>在这个示例中，需要为变量 <code>x</code> 指定一个整数类型，这里指定了 u32 ，因为整数类型很多，如 u32 ， i32 等， rust 不知道该整数具体的类型：</p><div class="language-rust vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">rust</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">fn</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() {</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    let</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> u32</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;12&quot;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">parse</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">expect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;这不是数字&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">);</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    println!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;x = {}&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, x)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span></code></pre></div><p>如果不指定类型，将会报错：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>...</span></span>
<span class="line"><span>2 |     let x = &quot;12&quot;.parse().expect(&quot;这不是数字&quot;);</span></span>
<span class="line"><span>  |         ^        ----- type must be known at this point</span></span>
<span class="line"><span>...</span></span></code></pre></div><h2 id="标量类型" tabindex="-1">标量类型 <a class="header-anchor" href="#标量类型" aria-label="Permalink to &quot;标量类型&quot;">​</a></h2><h3 id="整数类型" tabindex="-1">整数类型 <a class="header-anchor" href="#整数类型" aria-label="Permalink to &quot;整数类型&quot;">​</a></h3><p>整数类型分为<strong>有符号</strong>和<strong>无符号</strong>类型，分别用 <code>i</code> 和 <code>u</code> 表示，如 <code>u32</code> 是无符号类型，占据 32 bit 的空间。</p><p>rust 的整数类型的长度分别有 8、16、32、64 和 128 bit 。</p><p>除了 <code>i32</code> 等，还有 <code>isize</code> 和 <code>usize</code> 类型，分别表示当前计算机的位数的长度，如果计算机是 64 位的，则长度为 64 bit 。</p><h4 id="字面值" tabindex="-1">字面值 <a class="header-anchor" href="#字面值" aria-label="Permalink to &quot;字面值&quot;">​</a></h4><table tabindex="0"><thead><tr><th>数字形式</th><th>示例</th></tr></thead><tbody><tr><td>十进制</td><td>98_222</td></tr><tr><td>十六进制</td><td><strong>0x</strong>ff</td></tr><tr><td>八进制</td><td><strong>0o</strong>77</td></tr><tr><td>二进制</td><td><strong>0b</strong>1111_0000</td></tr><tr><td>字节（ Byte ，仅限于 <code>u8</code> 类型）</td><td><strong>b</strong>‘A’</td></tr></tbody></table><div class="note custom-block github-alert"><p class="custom-block-title">NOTE</p><p></p><p>除了字节类型外，所有整数数字形式都允许以类型作为后缀，如 <code>57u8</code> 。</p><p>整数的默认类型是 <code>i32</code> 。</p></div><h4 id="整数溢出" tabindex="-1">整数溢出 <a class="header-anchor" href="#整数溢出" aria-label="Permalink to &quot;整数溢出&quot;">​</a></h4><p>如果整数溢出，则：</p><p>在调试模式下编译：程序 panic</p><p>在发布模式下编译：执行“环绕”操作，如将 256 变为 0，将 257 变为 1 等，程序不会 panic</p><h3 id="浮点类型" tabindex="-1">浮点类型 <a class="header-anchor" href="#浮点类型" aria-label="Permalink to &quot;浮点类型&quot;">​</a></h3><p>浮点类型分为单精度 <code>f32</code> 和双精度 <code>f64</code> 类型，其中 <code>f64</code> 为默认类型。</p><h3 id="布尔类型" tabindex="-1">布尔类型 <a class="header-anchor" href="#布尔类型" aria-label="Permalink to &quot;布尔类型&quot;">​</a></h3><p>分为 <code>true</code> 和 <code>false</code> 类型，占用大小为 1 字节，符号为 <code>bool</code> 。</p><h3 id="字符类型" tabindex="-1">字符类型 <a class="header-anchor" href="#字符类型" aria-label="Permalink to &quot;字符类型&quot;">​</a></h3><p>使用 <code>char</code> 声明一个字符，使用单引号，占用 4 个字节。</p><p>其中， rust 中的字符类型为 <strong>unicode</strong> 标量值，可以表示比 ascii 更多的内容。</p><h2 id="复合类型" tabindex="-1">复合类型 <a class="header-anchor" href="#复合类型" aria-label="Permalink to &quot;复合类型&quot;">​</a></h2><p>复合类型是可以将多个类型的值放到一个类型里，如 Python 中的列表或元组等。Rust 提供 2 种基础的复合类型：元组和数组</p><h3 id="元组" tabindex="-1">元组 <a class="header-anchor" href="#元组" aria-label="Permalink to &quot;元组&quot;">​</a></h3><p>元组在小括号中声明，元组的<strong>长度不可变</strong>。</p><p>声明一个元组的示例如下，其中，访问元组内部元素使用点标记法访问，使用点 + 元素的索引：</p><div class="language-rust vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">rust</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">let</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> tup</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">i32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">f64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">u8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">500</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">6.4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">);</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">println!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;tup = {}, {}, {}&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, tup</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, tup</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, tup</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">);</span></span></code></pre></div><h4 id="元组解构" tabindex="-1">元组解构 <a class="header-anchor" href="#元组解构" aria-label="Permalink to &quot;元组解构&quot;">​</a></h4><p>使用模式匹配解构一个元组的值，接上例：</p><div class="language-rust vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">rust</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">let</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y, z) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> tup;</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">println!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;x = {}, y = {}, z = {}&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, x, y, z)</span></span></code></pre></div><h3 id="数组" tabindex="-1">数组 <a class="header-anchor" href="#数组" aria-label="Permalink to &quot;数组&quot;">​</a></h3><p>数组在中括号中声明。</p><div class="tip custom-block github-alert"><p class="custom-block-title">TIP</p><p></p><p>如果想要数据存放在**栈（stack）而不是堆（heap）**上，或者想要保证固定数量的元素，使用数组更好。</p></div><p>声明一个数组的例子：</p><div class="language-rust vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">rust</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">let</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> month </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">	&quot;Jan&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;Feb&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;Mar&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;Apr&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;May&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;Jun&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;Jul&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;Aug&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;Sep&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;Oct&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;Nov&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;Dec&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">];</span></span></code></pre></div><h4 id="数组的类型表示" tabindex="-1">数组的类型表示 <a class="header-anchor" href="#数组的类型表示" aria-label="Permalink to &quot;数组的类型表示&quot;">​</a></h4><p>格式如下：</p><p><code>[类型; 长度]</code></p><p>如：</p><div class="language-rust vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">rust</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">let</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> a</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">i32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">];</span></span></code></pre></div><h4 id="数组的特殊声明" tabindex="-1">数组的特殊声明 <a class="header-anchor" href="#数组的特殊声明" aria-label="Permalink to &quot;数组的特殊声明&quot;">​</a></h4><p>就像数组的类型表示那样，第一个元素不再是类型，而是数组元素的初始值，如 <code>[3; 5]</code> ，则表示该数组有 5 个元素，每个元素的值为 3 ：</p><div class="language-rust vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">rust</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">let</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> b </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">];</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">println!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;b = {:?}&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, b)</span></span></code></pre></div><p>输出：b = [3, 3, 3, 3, 3]</p><h4 id="数组元素访问" tabindex="-1">数组元素访问 <a class="header-anchor" href="#数组元素访问" aria-label="Permalink to &quot;数组元素访问&quot;">​</a></h4><p>和 Python 一样，<strong>使用索引访问数组中的元素</strong>。</p><p>如果访问的元素下标超出了数组长度，<strong>在编译时不会报错（非绝对，在某些简单的情况下会报错），但在运行时会报错</strong>。</p><blockquote><p>rust 不允许访问越界的内存。</p></blockquote>`,54)]))}const g=i(n,[["render",l]]);export{E as __pageData,g as default};
