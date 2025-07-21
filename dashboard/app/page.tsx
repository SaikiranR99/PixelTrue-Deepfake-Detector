"use client"

import React, { useRef, useEffect } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Shield } from "lucide-react"

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    let particles: any[] = [];
    const particleCount = 80;

    class Particle {
      x: number; y: number; size: number; speedX: number; speedY: number; color: string;
      constructor() {
        this.x = Math.random() * canvas!.width;
        this.y = Math.random() * canvas!.height;
        this.size = Math.random() * 2 + 1;
        this.speedX = (Math.random() - 0.5) * 0.2;
        this.speedY = (Math.random() - 0.5) * 0.2;
        this.color = `rgba(72, 187, 255, ${Math.random() * 0.4 + 0.1})`;
      }
      update() {
        this.x += this.speedX;
        this.y += this.speedY;
        if (this.x > (canvas?.width ?? 0) || this.x < 0) this.speedX *= -1;
        if (this.y > (canvas?.height ?? 0) || this.y < 0) this.speedY *= -1;
      }
      draw() {
        if (!ctx) return;
        ctx.fillStyle = this.color;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    function init() {
      for (let i = 0; i < particleCount; i++) {
        particles.push(new Particle());
      }
    }
    init();

    let animationFrameId: number;
    function animate() {
      if (!canvas || !ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      for (const particle of particles) {
        particle.update();
        particle.draw();
      }
      animationFrameId = requestAnimationFrame(animate);
    }
    animate();
    
    const handleResize = () => {
      if (!canvas) return;
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      particles = [];
      init();
    };
    window.addEventListener("resize", handleResize);

    return () => {
        window.removeEventListener("resize", handleResize);
        cancelAnimationFrame(animationFrameId);
    }
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-black to-slate-900 text-white font-mono relative overflow-hidden">
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full opacity-50 z-0" />
        <div className="relative z-10">
            {/* Navigation */}
            <nav className="fixed top-0 left-0 right-0 z-50 bg-black/30 backdrop-blur-sm">
                <div className="mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center justify-between h-16 my-2">
                    <div className="flex items-center gap-16">
                    <Link href="/" className="flex items-center gap-3">
                        <Shield className="h-8 w-8 text-cyan-400" />
                        <h1 className="text-xl font-bold uppercase tracking-wider text-white glow-text">
                        PixelTrue
                        </h1>
                    </Link>
                    </div>
                    <Link href="/login">
                        <Button className="futuristic-button">Launch Dashboard</Button>
                    </Link>
                </div>
                </div>
            </nav>

            {/* Hero Section */}
            <div className="relative pt-32 pb-40 sm:pt-40">
                <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
                    <div
                        className={`inline-flex items-center rounded-full px-4 py-1 mb-8
                        bg-cyan-900/50 border border-cyan-700/50
                        shadow-[0_0_30px_-5px_rgba(72,187,255,0.5)]`}
                    >
                        <div className="w-2 h-2 rounded-full bg-cyan-400 mr-2 animate-pulse" />
                        <span className="text-sm text-cyan-300">Advanced Deepfake Detection</span>
                    </div>

                    <h1 className="text-4xl sm:text-6xl lg:text-7xl font-bold tracking-tight text-white mb-6">
                        Unmask Reality
                        <br />
                        with <span className="text-cyan-400 glow-text">AI Analysis.</span>
                    </h1>

                    <p className="max-w-2xl mx-auto text-lg sm:text-xl text-gray-300 mb-16">
                        PixelTrue offers state-of-the-art deepfake detection. Secure your content with our powerful, user-friendly analysis tools.
                    </p>

                    {/* The broken image section has been removed from here */}

                </div>
            </div>
        </div>
        <style jsx>{`
            .glow-text { text-shadow: 0 0 8px rgba(72, 187, 255, 0.7); }
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
  )
}