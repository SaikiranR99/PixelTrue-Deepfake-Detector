"use client"

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from "@/components/ui/button";
import { Shield } from "lucide-react";

export default function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const router = useRouter();

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    // Check credentials
    if (email === 'rangarajans@cardiff.ac.uk' && password === 'pixeltrue') {
      sessionStorage.setItem('isLoggedIn', 'true');
      router.push('/dashboard');
    } else {
      setError('Invalid email or password.');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-black to-slate-900 text-white font-mono flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="panel-container">
          <div className="panel !p-8">
            <div className="corner-brackets"></div>
            <div className="flex flex-col items-center mb-6">
              <Shield className="h-12 w-12 text-cyan-400 mb-3" />
              <h1 className="text-2xl font-bold uppercase tracking-wider text-white glow-text">
                PixelTrue Login
              </h1>
              <p className="text-sm text-slate-400">Access the Analysis Dashboard</p>
            </div>

            <form onSubmit={handleLogin} className="space-y-6">
              <div>
                <label className="text-sm text-cyan-300/70 block mb-2">User Email</label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="Enter your email"
                  className="w-full bg-slate-900/50 border border-cyan-400/30 rounded-md px-4 py-2 text-white placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500 transition-all"
                  required
                />
              </div>

              <div>
                <label className="text-sm text-cyan-300/70 block mb-2">Password</label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter your password"
                  className="w-full bg-slate-900/50 border border-cyan-400/30 rounded-md px-4 py-2 text-white placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500 transition-all"
                  required
                />
              </div>

              {error && (
                <p className="text-red-500 text-sm text-center">{error}</p>
              )}

              <Button type="submit" className="futuristic-button w-full !py-3">
                Authenticate & Proceed
              </Button>
            </form>
          </div>
        </div>
      </div>
      <style jsx>{`
        .glow-text { text-shadow: 0 0 8px rgba(72, 187, 255, 0.7); }
        .panel-container { position: relative; }
        .panel {
          background: rgba(15, 23, 42, 0.6);
          border: 1px solid rgba(72, 187, 255, 0.3);
          border-radius: 8px;
          padding: 1.5rem;
          box-shadow: 0 0 25px rgba(72, 187, 255, 0.1);
          backdrop-filter: blur(12px);
          -webkit-backdrop-filter: blur(12px);
        }
        .corner-brackets::before, .corner-brackets::after {
          content: '';
          position: absolute;
          width: 20px; height: 20px;
          border: 2px solid rgba(72, 187, 255, 0.5);
        }
        .corner-brackets::before { top: -6px; left: -6px; border-right: none; border-bottom: none; }
        .corner-brackets::after { bottom: -6px; right: -6px; border-left: none; border-top: none; }
        .futuristic-button {
          background: rgba(72, 187, 255, 0.1);
          border: 1px solid rgba(72, 187, 255, 0.6);
          color: rgb(103, 232, 249);
          text-transform: uppercase;
          font-weight: bold; letter-spacing: 0.05em;
          transition: all 0.3s ease;
          padding: 0.5rem 1rem;
          display: flex; align-items: center; justify-content: center;
        }
        .futuristic-button:hover:not(:disabled) {
          background: rgba(72, 187, 255, 0.25);
          box-shadow: 0 0 20px rgba(72, 187, 255, 0.4);
          border-color: rgba(72, 187, 255, 1);
        }
      `}</style>
    </div>
  );
}