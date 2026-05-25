'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';

export default function MobileSidebar({ grouped, currentSlug, basePath, totalCount, currentIdx, labels }) {
  const [open, setOpen] = useState(false);

  // Close whenever user navigates to a new slug
  useEffect(() => { setOpen(false); }, [currentSlug]);

  // Prevent body scroll when panel is open
  useEffect(() => {
    document.body.style.overflow = open ? 'hidden' : '';
    return () => { document.body.style.overflow = ''; };
  }, [open]);

  return (
    <>
      {/* ── Floating trigger button (mobile only) ── */}
      <button
        onClick={() => setOpen(true)}
        aria-label={labels.allPosts}
        className="lg:hidden fixed bottom-6 right-4 z-40 flex items-center gap-2 bg-gray-900 text-white text-xs font-semibold px-4 py-2.5 rounded-full shadow-xl hover:bg-gray-700 active:scale-95 transition-all duration-150"
      >
        {/* hamburger icon */}
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none" xmlns="http://www.w3.org/2000/svg">
          <rect y="1"  width="14" height="1.5" rx="0.75" fill="currentColor"/>
          <rect y="6"  width="14" height="1.5" rx="0.75" fill="currentColor"/>
          <rect y="11" width="14" height="1.5" rx="0.75" fill="currentColor"/>
        </svg>
        {labels.allPosts}
      </button>

      {/* ── Backdrop ── */}
      {open && (
        <div
          className="lg:hidden fixed inset-0 bg-black/50 z-40 backdrop-blur-sm"
          onClick={() => setOpen(false)}
        />
      )}

      {/* ── Slide-in panel ── */}
      <div
        className={`lg:hidden fixed inset-y-0 right-0 z-50 w-80 max-w-[88vw] bg-white shadow-2xl flex flex-col transition-transform duration-300 ease-in-out ${
          open ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        {/* Panel header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100 shrink-0">
          <span className="font-bold text-sm text-gray-900">{labels.allPosts}</span>
          <button
            onClick={() => setOpen(false)}
            className="w-7 h-7 flex items-center justify-center rounded-full hover:bg-gray-100 text-gray-500 hover:text-gray-900 transition-colors text-lg leading-none"
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        {/* Progress bar */}
        <div className="px-4 py-3 border-b border-gray-100 shrink-0">
          <p className="text-xs text-gray-400 mb-1.5">{labels.progress}</p>
          <div className="flex items-center gap-2">
            <div className="flex-1 h-1.5 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-gray-900 rounded-full transition-all"
                style={{ width: `${((currentIdx + 1) / totalCount) * 100}%` }}
              />
            </div>
            <span className="text-xs font-semibold text-gray-600 shrink-0">{currentIdx + 1}/{totalCount}</span>
          </div>
        </div>

        {/* Category + blog list — scrollable */}
        <div className="flex-1 overflow-y-auto px-3 py-3">
          {Object.entries(grouped).map(([category, items]) => (
            <div key={category} className="mb-5">
              {/* Category heading */}
              <p className="text-[10px] font-bold uppercase tracking-widest text-gray-400 px-2 mb-1.5">
                {category}
              </p>
              <ol className="space-y-0.5">
                {items.map((item, i) => {
                  const isActive = item.slug === currentSlug;
                  return (
                    <li key={item.slug}>
                      <Link
                        href={`${basePath}/${item.slug}`}
                        onClick={() => setOpen(false)}
                        className={`flex items-start gap-2.5 px-3 py-2 rounded-lg text-xs transition-all duration-150 ${
                          isActive
                            ? 'bg-gray-900 text-white'
                            : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                        }`}
                      >
                        <span className={`shrink-0 mt-0.5 font-mono w-5 text-right ${isActive ? 'text-gray-400' : 'text-gray-300'}`}>
                          {i + 1}.
                        </span>
                        <span className="leading-snug">{item.title}</span>
                      </Link>
                    </li>
                  );
                })}
              </ol>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}
