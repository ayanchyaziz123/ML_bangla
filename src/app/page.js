import Link from 'next/link';
import { blogs } from '@/data/blogs/index';

const categoryColors = {
  'লিনিয়ার রিগ্রেশন': 'bg-blue-50 text-blue-700 border-blue-200',
};

export default function Home() {
  const categories = [...new Set(blogs.map((b) => b.category))];

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-10 sm:py-14">

      {/* Hero */}
      <section className="mb-12 pb-10 border-b border-gray-100">
        <span className="inline-block text-xs font-semibold tracking-widest uppercase text-amber-600 mb-4">
          বাংলা ব্লগ সিরিজ
        </span>
        <h1 className="text-3xl sm:text-5xl font-bold text-gray-900 leading-tight mb-4">
          বাংলায় মেশিন লার্নিং
        </h1>
        <p className="text-gray-500 text-base sm:text-lg max-w-2xl leading-relaxed">
          মেশিন লার্নিং-এর ধারণাগুলো সহজ বাংলায় — গণিত, কোড এবং বাস্তব উদাহরণ সহ।
          যারা বাংলায় শিখতে পছন্দ করেন তাদের জন্য।
        </p>
        <div className="flex flex-wrap gap-3 mt-6">
          {categories.map((cat) => (
            <span
              key={cat}
              className={`text-xs px-3 py-1 rounded-full border font-medium ${categoryColors[cat] ?? 'bg-gray-100 text-gray-600 border-gray-200'}`}
            >
              {cat}
            </span>
          ))}
          <span className="text-xs px-3 py-1 rounded-full border bg-gray-50 text-gray-500 border-gray-200">
            {blogs.length}টি আর্টিকেল
          </span>
        </div>
      </section>

      {/* Blog list */}
      <section>
        <div className="grid gap-4 sm:gap-5">
          {blogs.map((blog, i) => (
            <Link
              key={blog.slug}
              href={`/blog/${blog.slug}`}
              className="group flex flex-col sm:flex-row sm:items-start gap-4 p-5 sm:p-6 border border-gray-200 rounded-xl hover:border-gray-400 hover:shadow-md transition-all duration-200 bg-white"
            >
              {/* Number badge */}
              <div className="shrink-0 w-10 h-10 rounded-lg bg-gray-900 text-white flex items-center justify-center text-sm font-bold">
                {i + 1}
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex flex-wrap items-center gap-2 mb-2">
                  <span className={`text-xs px-2 py-0.5 rounded border font-medium ${categoryColors[blog.category] ?? 'bg-gray-100 text-gray-600 border-gray-200'}`}>
                    {blog.category}
                  </span>
                  {blog.readTime && (
                    <span className="text-xs text-gray-400">{blog.readTime} মিনিট পড়া</span>
                  )}
                </div>

                <h2 className="text-base sm:text-lg font-semibold text-gray-900 group-hover:text-gray-700 leading-snug mb-1.5 transition-colors">
                  {blog.title}
                </h2>
                <p className="text-sm text-gray-500 leading-relaxed line-clamp-2">
                  {blog.description}
                </p>
              </div>

              {/* Arrow */}
              <div className="shrink-0 self-center text-gray-300 group-hover:text-gray-600 group-hover:translate-x-1 transition-all duration-200 text-xl">
                →
              </div>
            </Link>
          ))}
        </div>
      </section>

    </div>
  );
}
