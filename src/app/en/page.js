'use client';

import Link from 'next/link';
import { useState } from 'react';
import { enBlogs }  from '@/data/en/index';
import { blogs }    from '@/data/blogs/index';

// Merge Bangla metadata (readTime, date) with English content
const mergedBlogs = blogs.map((b) => {
  const en = enBlogs.find((e) => e.slug === b.slug);
  return en ? { ...b, ...en } : null;
}).filter(Boolean);

const categoryColors = {
  'Linear Regression':   { badge: 'bg-blue-50 text-blue-700 border-blue-200',    btn: 'bg-blue-600 text-white border-blue-600',    btnOff: 'bg-blue-50 text-blue-700 border-blue-200 hover:bg-blue-100' },
  'Logistic Regression': { badge: 'bg-violet-50 text-violet-700 border-violet-200', btn: 'bg-violet-600 text-white border-violet-600', btnOff: 'bg-violet-50 text-violet-700 border-violet-200 hover:bg-violet-100' },
  'Decision Tree':       { badge: 'bg-emerald-50 text-emerald-700 border-emerald-200', btn: 'bg-emerald-600 text-white border-emerald-600', btnOff: 'bg-emerald-50 text-emerald-700 border-emerald-200 hover:bg-emerald-100' },
  'Naive Bayes':         { badge: 'bg-amber-50 text-amber-700 border-amber-200',  btn: 'bg-amber-500 text-white border-amber-500',  btnOff: 'bg-amber-50 text-amber-700 border-amber-200 hover:bg-amber-100' },
  'Support Vector Machine': { badge: 'bg-rose-50 text-rose-700 border-rose-200', btn: 'bg-rose-600 text-white border-rose-600', btnOff: 'bg-rose-50 text-rose-700 border-rose-200 hover:bg-rose-100' },
};

const fallbackColors = {
  badge:  'bg-gray-100 text-gray-600 border-gray-200',
  btn:    'bg-gray-700 text-white border-gray-700',
  btnOff: 'bg-gray-100 text-gray-600 border-gray-200 hover:bg-gray-200',
};

export default function EnglishHome() {
  const [active, setActive] = useState(null);

  const categories = [...new Set(mergedBlogs.map((b) => b.category))];
  const filtered   = active ? mergedBlogs.filter((b) => b.category === active) : mergedBlogs;

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-10 sm:py-14">

      {/* Hero */}
      <section className="mb-10 pb-10 border-b border-gray-100">
        <span className="inline-block text-xs font-semibold tracking-widest uppercase text-amber-600 mb-4">
          English Blog Series
        </span>
        <h1 className="text-3xl sm:text-5xl font-bold text-gray-900 leading-tight mb-4">
          Machine Learning in English
        </h1>
        <p className="text-gray-500 text-base sm:text-lg max-w-2xl leading-relaxed">
          Core machine learning concepts explained with mathematics, Python code, and real examples.
          Each post covers theory and hands-on implementation.
        </p>
      </section>

      {/* Category filter */}
      <section className="mb-8">
        <div className="flex flex-wrap gap-2 items-center">
          <button
            onClick={() => setActive(null)}
            className={`text-xs px-4 py-1.5 rounded-full border font-semibold transition-all duration-150 cursor-pointer ${
              active === null
                ? 'bg-gray-900 text-white border-gray-900'
                : 'bg-gray-100 text-gray-600 border-gray-200 hover:bg-gray-200'
            }`}
          >
            All ({mergedBlogs.length})
          </button>

          {categories.map((cat) => {
            const colors  = categoryColors[cat] ?? fallbackColors;
            const isActive = active === cat;
            const count    = mergedBlogs.filter((b) => b.category === cat).length;
            return (
              <button
                key={cat}
                onClick={() => setActive(isActive ? null : cat)}
                className={`text-xs px-4 py-1.5 rounded-full border font-semibold transition-all duration-150 cursor-pointer ${
                  isActive ? colors.btn : colors.btnOff
                }`}
              >
                {cat} ({count})
              </button>
            );
          })}
        </div>

        {active && (
          <p className="mt-3 text-sm text-gray-500">
            Showing {filtered.length} posts in{' '}
            <span className="font-medium text-gray-700">{active}</span>
            <button
              onClick={() => setActive(null)}
              className="ml-2 text-gray-400 hover:text-gray-700 text-xs underline cursor-pointer"
            >
              Show all
            </button>
          </p>
        )}
      </section>

      {/* Blog list */}
      <section>
        <div className="grid gap-4 sm:gap-5">
          {filtered.map((blog) => {
            const colors      = categoryColors[blog.category] ?? fallbackColors;
            const globalIndex = mergedBlogs.indexOf(blog);
            return (
              <Link
                key={blog.slug}
                href={`/en/blog/${blog.slug}`}
                className="group flex flex-col sm:flex-row sm:items-start gap-4 p-5 sm:p-6 border border-gray-200 rounded-xl hover:border-gray-400 hover:shadow-md transition-all duration-200 bg-white"
              >
                <div className="shrink-0 w-10 h-10 rounded-lg bg-gray-900 text-white flex items-center justify-center text-sm font-bold">
                  {globalIndex + 1}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex flex-wrap items-center gap-2 mb-2">
                    <span className={`text-xs px-2 py-0.5 rounded border font-medium ${colors.badge}`}>
                      {blog.category}
                    </span>
                    {blog.readTime && (
                      <span className="text-xs text-gray-400">{blog.readTime} min read</span>
                    )}
                  </div>
                  <h2 className="text-base sm:text-lg font-semibold text-gray-900 group-hover:text-gray-700 leading-snug mb-1.5 transition-colors">
                    {blog.title}
                  </h2>
                  <p className="text-sm text-gray-500 leading-relaxed line-clamp-2">
                    {blog.description}
                  </p>
                </div>
                <div className="shrink-0 self-center text-gray-300 group-hover:text-gray-600 group-hover:translate-x-1 transition-all duration-200 text-xl">
                  →
                </div>
              </Link>
            );
          })}
        </div>
      </section>

    </div>
  );
}
