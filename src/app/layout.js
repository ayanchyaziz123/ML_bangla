import "./globals.css";
import Navbar from "@/components/Navbar";
import { Suspense } from "react";

export const metadata = {
  title: "ML Blog — Machine Learning in Bangla & English",
  description: "Machine learning concepts explained in both Bangla and English — math, code, and real examples.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="bn" className="h-full">
      <body className="min-h-full flex flex-col bg-white text-gray-900 antialiased">
        <Suspense fallback={
          <header className="border-b border-gray-100 h-14 bg-white" />
        }>
          <Navbar />
        </Suspense>
        <main className="flex-1">{children}</main>
        <footer className="border-t border-gray-100 py-8 mt-16">
          <div className="max-w-5xl mx-auto px-4 sm:px-6 text-center text-sm text-gray-400">
            <p>Machine Learning Blog · Azizur Rahman</p>
          </div>
        </footer>
      </body>
    </html>
  );
}
