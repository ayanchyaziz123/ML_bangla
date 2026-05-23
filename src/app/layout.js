import "./globals.css";

export const metadata = {
  title: "বাংলায় মেশিন লার্নিং",
  description: "লিনিয়ার রিগ্রেশন সহ মেশিন লার্নিং-এর ধারণাগুলো সহজ বাংলায়।",
};

export default function RootLayout({ children }) {
  return (
    <html lang="bn" className="h-full">
      <body className="min-h-full flex flex-col bg-white text-gray-900 antialiased">
        <header className="border-b border-gray-100 sticky top-0 bg-white/95 backdrop-blur-sm z-50">
          <div className="max-w-5xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
            <a href="/" className="text-base font-semibold text-gray-900 tracking-tight">
              বাংলায় ML
            </a>
            <nav className="flex items-center gap-6 text-sm text-gray-600">
              <a href="/" className="hover:text-gray-900 transition-colors">সব ব্লগ</a>
              <span className="px-2.5 py-0.5 bg-amber-50 text-amber-700 border border-amber-200 rounded text-xs font-medium">বাংলা</span>
            </nav>
          </div>
        </header>
        <main className="flex-1">{children}</main>
        <footer className="border-t border-gray-100 py-8 mt-16">
          <div className="max-w-5xl mx-auto px-4 sm:px-6 text-center text-sm text-gray-400">
            <p>বাংলায় মেশিন লার্নিং · আজিজুর রহমান</p>
          </div>
        </footer>
      </body>
    </html>
  );
}
