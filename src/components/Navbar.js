'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

export default function Navbar() {
  const pathname = usePathname();
  const isEnglish = pathname.startsWith('/en');

  // For blog pages, swap between /blog/[slug] and /en/blog/[slug]
  const getAltLangHref = () => {
    if (isEnglish) {
      return pathname.replace(/^\/en/, '') || '/';
    } else {
      return `/en${pathname === '/' ? '' : pathname}`;
    }
  };

  const homeHref   = isEnglish ? '/en' : '/';
  const allPostsHref = isEnglish ? '/en' : '/';

  return (
    <header className="border-b border-gray-100 sticky top-0 bg-white/95 backdrop-blur-sm z-50">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
        <Link href={homeHref} className="text-base font-semibold text-gray-900 tracking-tight">
          {isEnglish ? 'ML Blog' : 'বাংলায় ML'}
        </Link>

        <nav className="flex items-center gap-4 text-sm text-gray-600">
          <Link href={allPostsHref} className="hover:text-gray-900 transition-colors">
            {isEnglish ? 'All Blogs' : 'সব ব্লগ'}
          </Link>

          {/* Language toggle */}
          <div className="flex items-center border border-gray-200 rounded-lg overflow-hidden text-xs font-semibold">
            <Link
              href={isEnglish ? getAltLangHref() : pathname}
              className={`px-3 py-1.5 transition-colors ${
                !isEnglish
                  ? 'bg-gray-900 text-white'
                  : 'text-gray-500 hover:bg-gray-50'
              }`}
            >
              বাংলা
            </Link>
            <Link
              href={isEnglish ? pathname : getAltLangHref()}
              className={`px-3 py-1.5 transition-colors ${
                isEnglish
                  ? 'bg-gray-900 text-white'
                  : 'text-gray-500 hover:bg-gray-50'
              }`}
            >
              English
            </Link>
          </div>
        </nav>
      </div>
    </header>
  );
}
