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
  'K-Nearest Neighbors':    { badge: 'bg-cyan-50 text-cyan-700 border-cyan-200',   btn: 'bg-cyan-600 text-white border-cyan-600',   btnOff: 'bg-cyan-50 text-cyan-700 border-cyan-200 hover:bg-cyan-100' },
  'Clustering':             { badge: 'bg-teal-50 text-teal-700 border-teal-200',   btn: 'bg-teal-600 text-white border-teal-600',   btnOff: 'bg-teal-50 text-teal-700 border-teal-200 hover:bg-teal-100' },
  'Neural Network':              { badge: 'bg-purple-50 text-purple-700 border-purple-200', btn: 'bg-purple-600 text-white border-purple-600', btnOff: 'bg-purple-50 text-purple-700 border-purple-200 hover:bg-purple-100' },
  'Convolutional Neural Network': { badge: 'bg-indigo-50 text-indigo-700 border-indigo-200', btn: 'bg-indigo-600 text-white border-indigo-600', btnOff: 'bg-indigo-50 text-indigo-700 border-indigo-200 hover:bg-indigo-100' },
  'Graph Neural Network':         { badge: 'bg-orange-50 text-orange-700 border-orange-200', btn: 'bg-orange-600 text-white border-orange-600', btnOff: 'bg-orange-50 text-orange-700 border-orange-200 hover:bg-orange-100' },
  'Transformers':                    { badge: 'bg-pink-50 text-pink-700 border-pink-200',     btn: 'bg-pink-600 text-white border-pink-600',     btnOff: 'bg-pink-50 text-pink-700 border-pink-200 hover:bg-pink-100' },
  'Dimensionality Reduction':        { badge: 'bg-lime-50 text-lime-700 border-lime-200',     btn: 'bg-lime-600 text-white border-lime-600',     btnOff: 'bg-lime-50 text-lime-700 border-lime-200 hover:bg-lime-100' },
  'Autoencoders':                    { badge: 'bg-sky-50 text-sky-700 border-sky-200',       btn: 'bg-sky-600 text-white border-sky-600',       btnOff: 'bg-sky-50 text-sky-700 border-sky-200 hover:bg-sky-100' },
  'Reinforcement Learning':          { badge: 'bg-red-50 text-red-700 border-red-200',       btn: 'bg-red-600 text-white border-red-600',       btnOff: 'bg-red-50 text-red-700 border-red-200 hover:bg-red-100' },
  'NLP Fundamentals':                { badge: 'bg-fuchsia-50 text-fuchsia-700 border-fuchsia-200', btn: 'bg-fuchsia-600 text-white border-fuchsia-600', btnOff: 'bg-fuchsia-50 text-fuchsia-700 border-fuchsia-200 hover:bg-fuchsia-100' },
  'Anomaly Detection':               { badge: 'bg-yellow-50 text-yellow-700 border-yellow-200', btn: 'bg-yellow-500 text-white border-yellow-500', btnOff: 'bg-yellow-50 text-yellow-700 border-yellow-200 hover:bg-yellow-100' },
  'Model Explainability':            { badge: 'bg-slate-50 text-slate-700 border-slate-200',  btn: 'bg-slate-600 text-white border-slate-600',   btnOff: 'bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100' },
  'Time Series':                     { badge: 'bg-green-50 text-green-700 border-green-200',  btn: 'bg-green-600 text-white border-green-600',   btnOff: 'bg-green-50 text-green-700 border-green-200 hover:bg-green-100' },
  'GANs':                            { badge: 'bg-zinc-50 text-zinc-700 border-zinc-200',    btn: 'bg-zinc-700 text-white border-zinc-700',     btnOff: 'bg-zinc-50 text-zinc-700 border-zinc-200 hover:bg-zinc-100' },
  'MLOps':                           { badge: 'bg-neutral-100 text-neutral-700 border-neutral-200', btn: 'bg-neutral-700 text-white border-neutral-700', btnOff: 'bg-neutral-100 text-neutral-700 border-neutral-200 hover:bg-neutral-200' },
  'Hyperparameter Optimization':     { badge: 'bg-stone-50 text-stone-700 border-stone-200',  btn: 'bg-stone-600 text-white border-stone-600',   btnOff: 'bg-stone-50 text-stone-700 border-stone-200 hover:bg-stone-100' },
  'LLMs & Fine-tuning':              { badge: 'bg-violet-100 text-violet-800 border-violet-300', btn: 'bg-violet-700 text-white border-violet-700', btnOff: 'bg-violet-100 text-violet-800 border-violet-300 hover:bg-violet-200' },
  'Recommender Systems':             { badge: 'bg-cyan-100 text-cyan-800 border-cyan-300',     btn: 'bg-cyan-700 text-white border-cyan-700',     btnOff: 'bg-cyan-100 text-cyan-800 border-cyan-300 hover:bg-cyan-200' },
  'Statistics for ML':               { badge: 'bg-emerald-100 text-emerald-800 border-emerald-300', btn: 'bg-emerald-700 text-white border-emerald-700', btnOff: 'bg-emerald-100 text-emerald-800 border-emerald-300 hover:bg-emerald-200' },
  'ML Interview Questions':          { badge: 'bg-gray-900 text-white border-gray-700',        btn: 'bg-gray-900 text-white border-gray-900',     btnOff: 'bg-gray-100 text-gray-800 border-gray-300 hover:bg-gray-200' },
};

const fallbackColors = {
  badge:  'bg-gray-100 text-gray-600 border-gray-200',
  btn:    'bg-gray-700 text-white border-gray-700',
  btnOff: 'bg-gray-100 text-gray-600 border-gray-200 hover:bg-gray-200',
};

function NeuralNetworkSVG() {
  const iX = 44, h1X = 134, h2X = 224, oX = 314;
  const iY = [65, 120, 175];
  const hY = [40, 90, 145, 195];
  const oY = [90, 155];
  const r  = 13;
  return (
    <svg viewBox="0 0 358 228" xmlns="http://www.w3.org/2000/svg" className="w-full h-auto select-none">
      <defs>
        <radialGradient id="gIen" cx="35%" cy="35%"><stop offset="0%" stopColor="#93c5fd"/><stop offset="100%" stopColor="#2563eb"/></radialGradient>
        <radialGradient id="gH1en" cx="35%" cy="35%"><stop offset="0%" stopColor="#c4b5fd"/><stop offset="100%" stopColor="#7c3aed"/></radialGradient>
        <radialGradient id="gH2en" cx="35%" cy="35%"><stop offset="0%" stopColor="#d8b4fe"/><stop offset="100%" stopColor="#9333ea"/></radialGradient>
        <radialGradient id="gOen" cx="35%" cy="35%"><stop offset="0%" stopColor="#f9a8d4"/><stop offset="100%" stopColor="#db2777"/></radialGradient>
        <style>{`
          @keyframes nnpen{0%,100%{opacity:.9}50%{opacity:.4}}
          .nien{animation:nnpen 2.6s ease-in-out infinite}
          .nh1en{animation:nnpen 2.6s ease-in-out infinite .35s}
          .nh2en{animation:nnpen 2.6s ease-in-out infinite .7s}
          .noen{animation:nnpen 2.6s ease-in-out infinite 1.05s}
        `}</style>
      </defs>
      {iY.flatMap(y1 => hY.map(y2 => <line key={`ih${y1}${y2}`} x1={iX+r} y1={y1} x2={h1X-r} y2={y2} stroke="#e2e8f0" strokeWidth="0.9"/>))}
      {hY.flatMap(y1 => hY.map(y2 => <line key={`hh${y1}${y2}`} x1={h1X+r} y1={y1} x2={h2X-r} y2={y2} stroke="#e2e8f0" strokeWidth="0.9"/>))}
      {hY.flatMap(y1 => oY.map(y2 => <line key={`ho${y1}${y2}`} x1={h2X+r} y1={y1} x2={oX-r} y2={y2} stroke="#e2e8f0" strokeWidth="0.9"/>))}
      <line x1={iX+r} y1={120} x2={h1X-r} y2={90}  stroke="#818cf8" strokeWidth="1.6" opacity="0.55"/>
      <line x1={h1X+r} y1={90} x2={h2X-r} y2={90}  stroke="#a78bfa" strokeWidth="1.6" opacity="0.55"/>
      <line x1={h2X+r} y1={90} x2={oX-r}  y2={90}  stroke="#c084fc" strokeWidth="1.6" opacity="0.55"/>
      {iY.map((y,i)  => <circle key={`ni${i}`}  className="nien"  cx={iX}  cy={y} r={r} fill="url(#gIen)"  style={{animationDelay:`${i*.15}s`}}/>)}
      {hY.map((y,i)  => <circle key={`nh1${i}`} className="nh1en" cx={h1X} cy={y} r={r} fill="url(#gH1en)" style={{animationDelay:`${.35+i*.15}s`}}/>)}
      {hY.map((y,i)  => <circle key={`nh2${i}`} className="nh2en" cx={h2X} cy={y} r={r} fill="url(#gH2en)" style={{animationDelay:`${.7+i*.15}s`}}/>)}
      {oY.map((y,i)  => <circle key={`no${i}`}  className="noen"  cx={oX}  cy={y} r={r} fill="url(#gOen)"  style={{animationDelay:`${1.05+i*.15}s`}}/>)}
      <text x={iX}  y={215} textAnchor="middle" fontSize="9" fill="#94a3b8" fontFamily="sans-serif">Input</text>
      <text x={h1X} y={215} textAnchor="middle" fontSize="9" fill="#94a3b8" fontFamily="sans-serif">Hidden 1</text>
      <text x={h2X} y={215} textAnchor="middle" fontSize="9" fill="#94a3b8" fontFamily="sans-serif">Hidden 2</text>
      <text x={oX}  y={215} textAnchor="middle" fontSize="9" fill="#94a3b8" fontFamily="sans-serif">Output</text>
    </svg>
  );
}

export default function EnglishHome() {
  const [active, setActive] = useState(null);

  const categories = [...new Set(mergedBlogs.map((b) => b.category))];
  const filtered   = active ? mergedBlogs.filter((b) => b.category === active) : mergedBlogs;

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-10 sm:py-14">

      {/* Hero */}
      <section className="mb-10 pb-10 border-b border-gray-100">
        <div className="flex flex-col lg:flex-row items-center gap-8 lg:gap-10">
          <div className="flex-1 min-w-0">
            <span className="inline-block text-xs font-semibold tracking-widest uppercase text-amber-600 mb-4">
              English Blog Series
            </span>
            <h1 className="text-3xl sm:text-5xl font-bold text-gray-900 leading-tight mb-4">
              Machine Learning in English
            </h1>
            <p className="text-gray-500 text-base sm:text-lg leading-relaxed">
              Core machine learning concepts explained with mathematics, Python code, and real examples.
              Each post covers theory and hands-on implementation.
            </p>
          </div>
          <div className="flex-shrink-0 w-full max-w-[280px] lg:max-w-[320px] opacity-90">
            <NeuralNetworkSVG />
          </div>
        </div>
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
