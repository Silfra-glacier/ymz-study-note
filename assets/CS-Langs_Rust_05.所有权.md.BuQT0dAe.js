import{_ as t,c as e,o as a,ag as o}from"./chunks/framework.BUKXnrp5.js";const i="/ymz-study-note/assets/05.%E6%89%80%E6%9C%89%E6%9D%83.CBDWuf91.png",p="/ymz-study-note/assets/05.%E6%89%80%E6%9C%89%E6%9D%832.TuIwwJoa.png",n="/ymz-study-note/assets/image-20250209180904505.DSrWQwU9.png",c="/ymz-study-note/assets/image-20250209181415713.VoSkBSWp.png",d="/ymz-study-note/assets/image-20250209182901007.D1U1pD1v.png",l="/ymz-study-note/assets/image-20250209200822649.Dt2kxp7M.png",b=JSON.parse('{"title":"05.所有权","description":"","frontmatter":{},"headers":[],"relativePath":"CS-Langs/Rust/05.所有权.md","filePath":"CS-Langs/Rust/05.所有权.md","lastUpdated":1740753730000}'),r={name:"CS-Langs/Rust/05.所有权.md"};function h(k,s,g,u,E,m){return a(),e("div",null,s[0]||(s[0]=[o('<h1 id="_05-所有权" tabindex="-1">05.所有权 <a class="header-anchor" href="#_05-所有权" aria-label="Permalink to &quot;05.所有权&quot;">​</a></h1><p>在 Rust 中，<strong>基本数据类型（如 <code>i32</code>, <code>f64</code>, <code>bool</code>, <code>char</code> 等）和固定大小的 <code>struct</code></strong> 变量通常存储在<strong>栈（stack）</strong> 中，因为它们的大小在编译时是已知的，并且可以直接被分配。</p><p><strong>动态大小的数据（如 <code>String</code>, <code>Vec&lt;T&gt;</code>, <code>Box&lt;T&gt;</code> 等）</strong>，它们的指针和元数据存储在栈上，而实际的数据存储在 <strong>堆（heap）</strong> 中。</p><h2 id="栈帧" tabindex="-1">栈帧 <a class="header-anchor" href="#栈帧" aria-label="Permalink to &quot;栈帧&quot;">​</a></h2><p>在 Rust 中，每次调用一个函数，都会在<strong>调用栈（stack）<strong>上创建一个新的</strong>栈帧（stack frame）</strong>。该函数中的变量等存放在该栈帧中。</p><div class="note custom-block github-alert"><p class="custom-block-title">NOTE</p><p></p><p>栈和栈帧不是同一个概念，栈以先进后出的方式存储，栈帧存放在栈中，而栈帧不一定按照先进后出的方式存储。</p><p>栈帧由 Rust 管理，Rust 不允许手动管理内存。</p></div><p>以下是 2 个代码段对应栈和栈帧内容的示例：</p><p><strong>示例 1</strong>，共 3 行代码，分别对应 3 个图示：</p><p><img src="'+i+'" alt="05.所有权"></p><p><strong>示例 2</strong>，3 张图示分别代表序号 ① ② ③：</p><p><img src="'+p+'" alt="05.所有权2"></p><h2 id="堆和栈" tabindex="-1">堆和栈 <a class="header-anchor" href="#堆和栈" aria-label="Permalink to &quot;堆和栈&quot;">​</a></h2><blockquote><p>固定大小的变量在分配时被分配到栈上。</p></blockquote><p>我们使用以下方法声明变量时，数据将被分配到栈上，使用 <code>let b = a;</code> 时，发生浅拷贝，即将数据复制一份给到 <code>b</code>，这样会占用较多的内存：</p><p><img src="'+n+'" alt="image-20250209180904505"></p><p>我们可以使用指针，将数据分配到堆上，堆中的数据不依赖与栈帧，可以长期存活。使用 <code>let b = a;</code> 时，<code>b</code> 也是一个指针，此时 <code>a</code> 被移动：</p><p><img src="'+c+'" alt="image-20250209181415713"></p><p><strong>栈和堆的区别可以总结如下：</strong></p><p>栈用于保存与特定函数相关联的数据，堆用于保存可能比函数存活更长时间的数据。</p><h2 id="box-智能指针" tabindex="-1">Box 智能指针 <a class="header-anchor" href="#box-智能指针" aria-label="Permalink to &quot;Box 智能指针&quot;">​</a></h2><p>在 Rust 中的 box，将其理解为一个智能指针，<code>Box&lt;T&gt;</code> 允许将类型 <code>T</code> 的<strong>数据存储在堆上</strong>，并在<strong>栈上只存储对该数据的指针</strong>。</p><p>使用 box 的集合有 <code>Vec, String, &amp; HashMap ...</code> 。</p><div class="important custom-block github-alert"><p class="custom-block-title">IMPORTANT</p><p></p><p><strong>Box 内存释放原则：</strong></p><p>如果一个变量绑定到一个 Box，当 rust 释放变量的栈帧时，rust 也会释放 box 的堆内存。</p></div><p><strong>案例 1：</strong></p><p><img src="'+d+'" alt="image-20250209182901007"></p><p>解释：</p><p>执行 L4 所在行代码时，将 <code>first</code> 变量当作参数传递给了 <code>add_suffix</code> 函数，此时，由于数据本身是一个 <code>String</code> ，因此，在栈上新分配一个指针（这里为 <code>name</code> 变量），指向堆上的这个数据，如图 L2，<strong>数据 <code>Ferris</code> 的所有权发生转移</strong>，从原本的 <code>first</code> 变量转移到了 <code>name</code> 变量，执行完 L3 所在行代码时，由于使用新增字符串的方法，因此原本堆上的数据后面增加了 <code> Jr.</code> ，使用原本内存所在地址，最终，数据赋给 <code>full</code> 变量，数据的所有权从 <code>name</code> 变量转移给了 <code>full</code>。此时，<code>first</code> 已不再可用（移动堆数据原则，见下）。</p><blockquote><p>虽然所有权发生了转移，但指针变量占用的栈上的内存空间不会随着所有权的转移而立即释放，而是等到整个函数执行完成，由 rust 将整个函数的栈帧释放时释放。如上述的 <code>first</code> 指针变量的所有权虽然转移到了 <code>add_suffix</code> 内部，但占用的栈内存不会立即释放，而是等到 <code>main()</code> 函数执行完成后随着整个 <code>main</code> 函数栈帧被释放时释放，但在所有权发生转移后，变量 <code>first</code> 将不再可用。</p></blockquote><div class="important custom-block github-alert"><p class="custom-block-title">IMPORTANT</p><p></p><p><strong>移动堆数据原则：</strong></p><p>如果变量 <code>x</code> 将堆中的数据的所有权移动到了另一个变量 <code>y</code>，那么移动后，<code>x</code> 将不再可用。</p><p><strong>避免移动：</strong></p><p>方法之一是使用 <code>.clone()</code> 方法进行克隆。如下示例，所有权将不会发生移动：</p><p><img src="'+l+`" alt="image-20250209200822649"></p></div><h2 id="所有权的经典问题" tabindex="-1">所有权的经典问题 <a class="header-anchor" href="#所有权的经典问题" aria-label="Permalink to &quot;所有权的经典问题&quot;">​</a></h2><p>问题 1，以下代码能否通过编译并执行？如果能，输出是什么？</p><div class="language-rust vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">rust</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">let</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> String</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">from</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;hello&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">);</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">let</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s2;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">let</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> b </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> b {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    s2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">println!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;{}&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, s);</span></span></code></pre></div><p>答案：代码不能通过编译。</p><p>原因：</p><p>在运行时 <code>b</code> 是一个未定变量，rust 无法确定 <code>if</code> 中的语句能否执行，而 rust 需要在编译时就确定所有权的归属。其中，<code>s</code> 是一个在栈上的变量，指向分配到堆上的数据 <code>hello</code> 。如果 <code>if</code> 可以执行，则 <code>s</code> 则将自己指向的数据归属到了 <code>s2</code>，不再拥有数据 <code>hello</code> 的所有权，此时再访问 <code>s</code> 将出错。而无法保证 <code>if</code> 能否执行，导致 rust 无法确认 <code>s</code> 是否可访问，编译不会通过。</p>`,35)]))}const y=t(r,[["render",h]]);export{b as __pageData,y as default};
