import { notFound } from 'next/navigation';
import Link from 'next/link';
import { blogs }    from '@/data/blogs/index';
import { enBlogs, enBlogMap } from '@/data/en/index';
import MobileSidebar from '@/components/MobileSidebar';

// Generate static paths for all English blogs
export function generateStaticParams() {
  return enBlogs.map((b) => ({ slug: b.slug }));
}

export async function generateMetadata({ params }) {
  const { slug } = await params;
  const en = enBlogMap[slug];
  if (!en) return {};
  return { title: en.title, description: en.description };
}

export default async function EnglishBlogPost({ params }) {
  const { slug } = await params;
  const en = enBlogMap[slug];
  if (!en) notFound();

  // Merge with Bangla source for readTime, date
  const source = blogs.find((b) => b.slug === slug) ?? {};
  const blog   = { ...source, ...en };

  const currentIdx = enBlogs.findIndex((b) => b.slug === slug);
  const prevBlog   = currentIdx > 0 ? enBlogs[currentIdx - 1] : null;
  const nextBlog   = currentIdx < enBlogs.length - 1 ? enBlogs[currentIdx + 1] : null;

  const categories = [...new Set(enBlogs.map((b) => b.category))];
  const grouped    = categories.reduce((acc, cat) => {
    acc[cat] = enBlogs.filter((b) => b.category === cat).map(({ slug, title }) => ({ slug, title }));
    return acc;
  }, {});

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 py-6 sm:py-10">

      {/* Back */}
      <Link
        href="/en"
        className="inline-flex items-center gap-1.5 text-sm text-gray-400 hover:text-gray-700 transition-colors mb-6"
      >
        ← All Posts
      </Link>

      {/* Mobile progress strip */}
      <div className="lg:hidden mb-6 flex items-center gap-3 p-3 bg-gray-50 rounded-xl border border-gray-100">
        <div className="flex-1 h-1.5 bg-gray-200 rounded-full overflow-hidden">
          <div
            className="h-full bg-gray-900 rounded-full"
            style={{ width: `${((currentIdx + 1) / enBlogs.length) * 100}%` }}
          />
        </div>
        <span className="text-xs font-medium text-gray-500 shrink-0">{currentIdx + 1} / {enBlogs.length}</span>
      </div>

      {/* Mobile sidebar toggle */}
      <MobileSidebar
        grouped={grouped}
        currentSlug={slug}
        basePath="/en/blog"
        totalCount={enBlogs.length}
        currentIdx={currentIdx}
        labels={{ allPosts: 'All Blogs', progress: 'Reading progress' }}
      />

      <div className="flex flex-col lg:flex-row gap-10 items-start">

        {/* ── Article ── */}
        <main className="flex-1 min-w-0 w-full">

          <header className="mb-8">
            <div className="flex flex-wrap items-center gap-2 mb-4">
              <span className="text-xs px-2.5 py-1 bg-blue-50 text-blue-700 border border-blue-200 rounded-full font-medium">
                {blog.category}
              </span>
              <span className="text-xs px-2.5 py-1 bg-emerald-50 text-emerald-700 border border-emerald-200 rounded-full font-medium">
                English
              </span>
              {blog.readTime && (
                <span className="text-xs text-gray-400">{blog.readTime} min read</span>
              )}
            </div>

            <h1 className="text-xl sm:text-3xl font-bold text-gray-900 leading-snug mb-4">
              {blog.title}
            </h1>
            <p className="text-gray-500 text-sm sm:text-base leading-relaxed border-l-4 border-gray-200 pl-4">
              {blog.description}
            </p>
          </header>

          <hr className="border-gray-100 mb-8" />

          {/* Body */}
          <div
            className="prose prose-sm sm:prose-base max-w-none
              prose-headings:text-gray-900 prose-headings:font-bold
              prose-h3:text-lg prose-h3:sm:text-xl prose-h3:mt-8 prose-h3:sm:mt-10 prose-h3:mb-3 prose-h3:border-b prose-h3:border-gray-100 prose-h3:pb-2
              prose-h4:text-base prose-h4:mt-6 prose-h4:mb-2
              prose-p:text-gray-600 prose-p:leading-relaxed
              prose-a:text-blue-600 prose-a:no-underline hover:prose-a:underline
              prose-strong:text-gray-800 prose-strong:font-semibold
              prose-code:text-blue-700 prose-code:bg-blue-50 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-sm prose-code:before:content-none prose-code:after:content-none
              prose-pre:bg-gray-950 prose-pre:text-gray-200 prose-pre:rounded-xl prose-pre:text-xs prose-pre:sm:text-sm prose-pre:leading-relaxed prose-pre:my-4
              prose-table:text-xs prose-table:sm:text-sm prose-th:border prose-th:border-gray-200 prose-th:bg-gray-50 prose-th:px-2 prose-th:sm:px-3 prose-th:py-2 prose-th:text-left prose-th:font-semibold prose-td:border prose-td:border-gray-200 prose-td:px-2 prose-td:sm:px-3 prose-td:py-2 prose-td:text-gray-600 prose-td:align-top
              prose-ul:text-gray-600 prose-li:text-gray-600 prose-li:marker:text-gray-400
              prose-blockquote:border-l-blue-300 prose-blockquote:text-gray-500 prose-blockquote:bg-blue-50 prose-blockquote:rounded-r-lg"
            dangerouslySetInnerHTML={{ __html: blog.content }}
          />

          <hr className="border-gray-100 mt-12 mb-8" />

          {/* Prev / Next */}
          <div className="flex flex-col sm:flex-row items-stretch sm:items-start gap-3 sm:gap-6">
            {prevBlog ? (
              <Link href={`/en/blog/${prevBlog.slug}`} className="group flex-1 p-3 sm:p-0 rounded-xl border border-gray-100 sm:border-0 bg-gray-50 sm:bg-transparent">
                <div className="flex items-center gap-1 text-xs text-gray-400 mb-1">← Previous</div>
                <p className="text-sm font-semibold text-gray-700 group-hover:text-gray-900 transition-colors leading-snug line-clamp-2">
                  {prevBlog.title}
                </p>
              </Link>
            ) : <div className="flex-1 hidden sm:block" />}

            {nextBlog ? (
              <Link href={`/en/blog/${nextBlog.slug}`} className="group flex-1 p-3 sm:p-0 rounded-xl border border-gray-100 sm:border-0 bg-gray-50 sm:bg-transparent sm:text-right">
                <div className="flex items-center sm:justify-end gap-1 text-xs text-gray-400 mb-1">Next →</div>
                <p className="text-sm font-semibold text-gray-700 group-hover:text-gray-900 transition-colors leading-snug line-clamp-2">
                  {nextBlog.title}
                </p>
              </Link>
            ) : <div className="flex-1 hidden sm:block" />}
          </div>

        </main>

        {/* ── Sidebar (desktop only) ── */}
        <aside className="hidden lg:block w-64 xl:w-72 shrink-0">
          <div className="sticky top-20">

            <div className="mb-5 p-4 bg-gray-50 rounded-xl border border-gray-100">
              <p className="text-xs text-gray-500 mb-1">Reading progress</p>
              <div className="flex items-center gap-2">
                <div className="flex-1 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gray-900 rounded-full"
                    style={{ width: `${((currentIdx + 1) / enBlogs.length) * 100}%` }}
                  />
                </div>
                <span className="text-xs font-medium text-gray-600">{currentIdx + 1}/{enBlogs.length}</span>
              </div>
            </div>

            {Object.entries(grouped).map(([category, items]) => (
              <div key={category} className="mb-6">
                <p className="text-xs font-bold uppercase tracking-widest text-gray-400 mb-2 px-1">
                  {category}
                </p>
                <ol className="space-y-0.5">
                  {items.map((item, i) => (
                    <li key={item.slug}>
                      <Link
                        href={`/en/blog/${item.slug}`}
                        className={`flex items-start gap-2.5 px-3 py-2 rounded-lg text-xs transition-all duration-150 ${
                          item.slug === slug
                            ? 'bg-gray-900 text-white'
                            : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                        }`}
                      >
                        <span className={`shrink-0 mt-0.5 font-mono text-right w-4 ${item.slug === slug ? 'text-gray-400' : 'text-gray-300'}`}>
                          {i + 1}.
                        </span>
                        <span className="leading-snug">{item.title}</span>
                      </Link>
                    </li>
                  ))}
                </ol>
              </div>
            ))}

          </div>
        </aside>

      </div>
    </div>
  );
}
