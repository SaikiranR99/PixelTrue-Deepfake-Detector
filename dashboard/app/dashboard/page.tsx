"use client"

import React, { useState, useRef, useEffect } from "react"
import { useRouter } from 'next/navigation';
import { BarChart3, TrendingUp, TrendingDown, Shield, FileImage, FileVideo, Scan, ServerCrash, BarChart as BarIcon, AreaChart as AreaIcon, Download, LucideIcon, ChevronLeft, ChevronRight, LogOut, AlertTriangle } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
    XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    BarChart, Bar, ReferenceLine, AreaChart, Area, Cell, Brush
} from 'recharts';
import { Badge } from "@/components/ui/badge"

// Interface for the analysis results from the backend
interface AnalysisResult {
  prediction: string;
  confidence: number;
  frameMetrics: number[];
  gradcamImages: string[];
  summaryStats: {
    average: number;
    highest: number;
    lowest: number;
  };
  featureMetrics?: number[];
  transformedVideoPath?: string;
  featureThreshold?: number;
}

// Toast component for displaying temporary messages
const Toast = ({ message, onDone }: { message: string, onDone: () => void }) => {
    useEffect(() => {
        const timer = setTimeout(() => {
            onDone();
        }, 4000); // Hide after 4 seconds

        return () => clearTimeout(timer);
    }, [onDone]);

    return (
        <div className="fixed top-5 right-5 bg-red-800/80 border border-red-500/50 text-white p-4 rounded-lg shadow-lg flex items-center gap-3 animate-fade-in-down z-50">
            <AlertTriangle className="h-6 w-6 text-yellow-400" />
            <span>{message}</span>
        </div>
    );
};


const Sidebar = ({
    isOpen,
    selectedType,
    setSelectedType,
    selectedFile,
    handleFileChange,
    handleInitiateScan,
    isLoading,
    status,
    error,
    setAnalysisResult,
    setSelectedFile,
    setStatus,
    setError
} : {
    isOpen: boolean,
    selectedType: "Video" | "Image",
    setSelectedType: (type: "Video" | "Image") => void,
    selectedFile: File | null,
    handleFileChange: (event: React.ChangeEvent<HTMLInputElement>) => void,
    handleInitiateScan: () => void,
    isLoading: boolean,
    status: string,
    error: string | null,
    setAnalysisResult: (result: AnalysisResult | null) => void,
    setSelectedFile: (file: File | null) => void,
    setStatus: (status: string) => void,
    setError: (error: string | null) => void,
}) => {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const router = useRouter();

    const handleSelectClick = () => {
        if(fileInputRef.current) {
            fileInputRef.current.click();
        }
    }

    const handleLogout = () => {
        sessionStorage.removeItem('isLoggedIn');
        router.push('/');
    }

    return (
        <aside className={`w-80 bg-slate-900/50 border-r border-cyan-400/20 p-6 flex flex-col gap-6 fixed top-0 left-0 h-full z-20 transition-transform duration-300 ease-in-out ${isOpen ? 'transform-none' : '-translate-x-full'}`}>
            <div className="flex items-center gap-3">
                <Shield className="h-8 w-8 text-cyan-400" />
                <h1 className="text-xl font-bold uppercase tracking-wider text-white glow-text">
                   PixelTrue
                </h1>
            </div>

            <div className="panel-container flex-grow flex flex-col">
                <div className="panel !p-0 flex-grow flex flex-col">
                     <h2 className="panel-title !p-4 border-b border-cyan-400/20">SYSTEM CONTROL - INPUT</h2>
                     <div className="space-y-4 p-4 flex-grow">
                        <Select
                            value={selectedType}
                            onValueChange={(value: "Video" | "Image") => {
                                setSelectedType(value);
                                setAnalysisResult(null);
                                setSelectedFile(null);
                                setStatus("STATUS: Awaiting target selection...");
                                setError(null);
                            }}
                        >
                            <SelectTrigger className="w-full futuristic-button text-left justify-start bg-slate-900 border-cyan-400/40 text-cyan-300">
                              {selectedType === 'Video' ? <FileVideo className="h-4 w-4 mr-2"/> : <FileImage className="h-4 w-4 mr-2"/>}
                              <SelectValue placeholder="Select target type..." />
                            </SelectTrigger>
                            <SelectContent className="bg-slate-900 border-cyan-400/40 text-cyan-300">
                                <SelectItem value="Video" className="hover:bg-cyan-400/20 focus:bg-cyan-400/20">Video</SelectItem>
                                <SelectItem value="Image" className="hover:bg-cyan-400/20 focus:bg-cyan-400/20">Image</SelectItem>
                            </SelectContent>
                        </Select>

                        <input type="file" ref={fileInputRef} onChange={handleFileChange} accept={selectedType === 'Video' ? "video/*" : "image/*"} className="hidden" />

                        <Button onClick={handleSelectClick} className="futuristic-button w-full">
                            {selectedFile ? 'CHANGE TARGET' : 'SELECT TARGET'}
                        </Button>

                        <Button onClick={handleInitiateScan} className="futuristic-button w-full" disabled={isLoading || !selectedFile}>
                            <Scan className="h-4 w-4 mr-2"/> {isLoading ? "ANALYZING..." : "INITIATE SCAN"}
                        </Button>

                        <div className="status-display">
                            <span className="text-cyan-300/70 text-sm block overflow-hidden text-ellipsis whitespace-nowrap" title={status}>{status}</span>
                            {error && <div className="text-red-500 text-sm mt-1 flex items-center gap-2"><ServerCrash className="h-4 w-4"/> ERROR: {error}</div>}
                        </div>
                    </div>
                     <div className="p-4 border-t border-cyan-400/20">
                         <Button onClick={handleLogout} className="futuristic-button w-full !bg-red-900/30 !border-red-500/50 hover:!bg-red-900/60">
                             <LogOut className="h-4 w-4 mr-2"/> LOGOUT
                         </Button>
                     </div>
                </div>
            </div>
        </aside>
    );
}

function MetricCard({ icon: Icon, title, value, color }: { icon: LucideIcon; title: string; value: string | number; color: 'cyan' | 'red' | 'green' }) {
    const colorClasses = {
        cyan: "from-cyan-500 to-blue-500 border-cyan-500/30",
        red: "from-red-500 to-orange-500 border-red-500/30",
        green: "from-green-500 to-emerald-500 border-green-500/30",
    };
    const iconColor = {
        cyan: "text-cyan-400",
        red: "text-red-400",
        green: "text-green-400",
    }

    return (
        <div className={`panel-container metric-card`}>
            <div className={`panel !p-4 border ${colorClasses[color]}`}>
                 <div className="flex items-center justify-between mb-2">
                    <div className="text-sm text-slate-400">{title}</div>
                    <Icon className={`h-5 w-5 ${iconColor[color]}`} />
                </div>
                <div className="text-3xl font-bold text-slate-100">{value}</div>
                <div className={`absolute -bottom-8 -right-8 h-20 w-20 rounded-full bg-gradient-to-r opacity-20 blur-xl ${colorClasses[color]}`}></div>
            </div>
        </div>
    );
}

const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
        const dataKey = payload[0].dataKey;
        const value = payload[0].value;

        if (dataKey === 'value') { // Feature Metrics
            return (
                <div className="bg-slate-800/80 border border-cyan-400/50 p-2 rounded-md text-sm font-mono">
                    <p className="label text-cyan-300">{`Feature ${label} Value: ${value.toFixed(4)}`}</p>
                </div>
            );
        } else if (dataKey === "score") { // Frame-by-frame analysis
            return (
                <div className="bg-slate-800/80 border border-cyan-400/50 p-2 rounded-md text-sm font-mono">
                    <p className="label text-cyan-300">{`Frame ${label} Score: ${value.toFixed(3)}`}</p>
                </div>
            );
        } else if (dataKey === "count") { // Score distribution
            return (
                <div className="bg-slate-800/80 border border-cyan-400/50 p-2 rounded-md text-sm font-mono">
                    <p className="label text-cyan-300">{`Range ${label}: ${value} occurrences`}</p>
                </div>
            );
        }
        return null;
    }
    return null;
};


export default function FuturisticDashboard() {
  const router = useRouter();
  const [authLoading, setAuthLoading] = useState(true);

  const [selectedType, setSelectedType] = useState<"Video" | "Image">("Image");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [status, setStatus] = useState("STATUS: Awaiting target selection...");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [highlightedBin, setHighlightedBin] = useState<string | null>(null);
  const [focusedFrame, setFocusedFrame] = useState<{ frame: number; score: number; camImage: string; } | null>(null);
  const [toastMessage, setToastMessage] = useState<string | null>(null); // State for toast messages

  useEffect(() => {
    const isLoggedIn = sessionStorage.getItem('isLoggedIn');
    if (!isLoggedIn) {
      router.push('/login');
    } else {
      setAuthLoading(false);
    }
  }, [router]);


  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      // --- VALIDATION LOGIC ---
      const MAX_FILE_SIZE = 200 * 1024 * 1024; // 200MB

      // 1. Check file size
      if (file.size > MAX_FILE_SIZE) {
        setToastMessage(`File exceeds the 200MB size limit.`);
        event.target.value = ''; // Reset file input
        return;
      }

      // 2. Check file type
      const isImageType = file.type.startsWith('image/');
      const isVideoType = file.type.startsWith('video/');

      if (selectedType === 'Image' && !isImageType) {
        setToastMessage(`Incorrect file type. Please select an image.`);
        event.target.value = ''; // Reset file input
        return;
      }

      if (selectedType === 'Video' && !isVideoType) {
        setToastMessage(`Incorrect file type. Please select a video.`);
        event.target.value = ''; // Reset file input
        return;
      }
      // --- END VALIDATION ---

      setSelectedFile(file);
      setAnalysisResult(null);
      setError(null);
      setHighlightedBin(null);
      setFocusedFrame(null);
      setStatus(`STATUS: File selected - ${file.name}`);
    }
  };

  const handleInitiateScan = async () => {
    if (!selectedFile) {
      setError("Please select a video or image file first.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setAnalysisResult(null);
    setHighlightedBin(null);
    setFocusedFrame(null);
    setStatus("STATUS: Uploading and initiating deep analysis protocol...");

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('type', selectedType);

    try {
      const response = await fetch('http://localhost:5000/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || `HTTP error! status: ${response.status}`);
      }

      let data: AnalysisResult = await response.json();
      setAnalysisResult(data);
      setStatus("STATUS: Scan complete. Results available.");

    } catch (err: any) {
      console.error("Analysis failed:", err);
      setError(err.message || "An unknown error occurred during analysis.");
      setStatus("STATUS: Critical error during analysis.");
    } finally {
      setIsLoading(false);
    }
  };

  const chartData = analysisResult?.frameMetrics.map((value, index) => ({
    frame: index + 1,
    score: value,
  })) || [];

  const scoreDistributionData = analysisResult ?
      analysisResult.frameMetrics.reduce((acc, score) => {
          const bin = Math.floor(score * 10) / 10;
          const binLabel = `${bin.toFixed(1)}-${(bin + 0.1).toFixed(1)}`;
          const existing = acc.find(d => d.name === binLabel);
          if (existing) existing.count += 1;
          else acc.push({ name: binLabel, count: 1 });
          return acc;
      }, [] as {name: string, count: number}[])
      .sort((a,b) => parseFloat(a.name.split('-')[0]) - parseFloat(b.name.split('-')[0]))
  : [];

  const featureLineData = analysisResult?.featureMetrics?.map((value, index) => ({
      index: index + 1,
      value: value,
  })) || [];

  const gradCamData = analysisResult?.gradcamImages.map((base64Image, i) => ({
      image: base64Image,
      frame: i + 1,
      score: analysisResult.frameMetrics[i]
  })) || [];

  const filteredGradCamData = highlightedBin
      ? gradCamData.filter(item => {
          const [min, max] = highlightedBin.split('-').map(parseFloat);
          return item.score >= min && item.score < max;
        })
      : gradCamData;

  const hasMeaningfulFeatureMetrics = analysisResult?.featureMetrics &&
                                       analysisResult.featureMetrics.length > 0 &&
                                       analysisResult.featureMetrics.some(val => val !== 0);

  useEffect(() => {
    if (authLoading) return;

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
  }, [authLoading]);
  
  const CustomizedDot = (props: any) => {
      const { cx, cy, payload } = props;
      if (!highlightedBin) {
          return null;
      }

      const score = payload.score;
      const [min, max] = highlightedBin.split('-').map(parseFloat);
      if (score >= min && score < max) {
          return <circle cx={cx} cy={cy} r={5} fill="#facc15" stroke="#0f172a" strokeWidth={1} />;
      }
      return <circle cx={cx} cy={cy} r={2} fill="#00ffff" opacity={0.4} />;
  };

  if (authLoading) {
    return (
        <div className="min-h-screen bg-gradient-to-br from-black to-slate-900 flex items-center justify-center">
            <p className="text-cyan-400 animate-pulse">Verifying session...</p>
        </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-black to-slate-900 text-white font-mono relative overflow-hidden">

      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full opacity-50 z-0" />

      {toastMessage && <Toast message={toastMessage} onDone={() => setToastMessage(null)} />}

      {focusedFrame && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 backdrop-blur-md" onClick={() => setFocusedFrame(null)}>
            <div className="panel-container" onClick={(e) => e.stopPropagation()}>
                <div className="panel !w-[40vw] max-w-[800px]">
                     <h2 className="panel-title">FOCUSED FRAME: {focusedFrame.frame}</h2>
                     <div className="bg-black/20 p-2 rounded-md">
                        <img src={`data:image/jpeg;base64,${focusedFrame.camImage}`} alt={`Focused Grad-CAM for frame ${focusedFrame.frame}`} className="w-full h-auto object-contain rounded-md" />
                     </div>
                     <p className="text-center mt-4 text-cyan-300">Frame Score: <span className="font-bold text-white">{focusedFrame.score.toFixed(4)}</span></p>
                     <Button onClick={() => setFocusedFrame(null)} className="futuristic-button w-full mt-4">CLOSE</Button>
                </div>
            </div>
        </div>
      )}

      {isLoading && (
        <div className="absolute inset-0 bg-black/80 flex items-center justify-center z-50 backdrop-blur-sm">
          <div className="flex flex-col items-center">
            <div className="relative w-24 h-24">
              <div className="absolute inset-0 border-4 border-cyan-500/30 rounded-full animate-ping"></div>
              <div className="absolute inset-2 border-4 border-t-cyan-500 border-r-transparent border-b-transparent border-l-transparent rounded-full animate-spin"></div>
              <div className="absolute inset-4 border-4 border-r-purple-500 border-t-transparent border-b-transparent border-l-transparent rounded-full animate-spin-slow"></div>
              <div className="absolute inset-6 border-4 border-b-blue-500 border-t-transparent border-r-transparent border-l-transparent rounded-full animate-spin-slower"></div>
            </div>
            <div className="mt-6 text-cyan-400 font-mono text-sm tracking-wider glow-text">
              INITIATING DEEP ANALYSIS PROTOCOL...
            </div>
          </div>
        </div>
      )}

      <Button
          onClick={() => setIsSidebarOpen(!isSidebarOpen)}
          className={`futuristic-button !p-2 fixed top-4 z-30 transition-all duration-300 ease-in-out ${isSidebarOpen ? 'left-[19rem]' : 'left-4'}`}
          title={isSidebarOpen ? "Close Sidebar" : "Open Sidebar"}
      >
          {isSidebarOpen ? <ChevronLeft className="h-5 w-5" /> : <ChevronRight className="h-5 w-5" />}
      </Button>

      <Sidebar
        isOpen={isSidebarOpen}
        selectedType={selectedType}
        setSelectedType={setSelectedType}
        selectedFile={selectedFile}
        handleFileChange={handleFileChange}
        handleInitiateScan={handleInitiateScan}
        isLoading={isLoading}
        status={status}
        error={error}
        setAnalysisResult={setAnalysisResult}
        setSelectedFile={setSelectedFile}
        setStatus={setStatus}
        setError={setError}
      />

      <main className={`relative z-10 p-6 transition-all duration-300 ${isSidebarOpen ? 'ml-80' : 'ml-20'}`}>
        <header className="w-full mb-6">
            <h1 className="text-2d md:text-3xl font-bold uppercase tracking-wider text-white glow-text">
                DEEPFAKE ANALYSIS DASHBOARD
            </h1>
        </header>

        <div className="grid grid-cols-3 gap-6">
            <div className="col-span-1">
                <div className="panel-container h-full">
                    <div className="panel h-full">
                        <div className="corner-brackets"></div>
                        <h2 className="panel-title">ANALYSIS RESULT</h2>
                        {isLoading && <div className="flex items-center justify-center h-full text-center text-cyan-400/50">Awaiting analysis results...</div>}
                        {analysisResult && (
                            <div className="text-center space-y-4 flex flex-col justify-center h-full">
                                <Badge variant="outline" className={`text-lg py-1 px-4 uppercase self-center ${analysisResult.prediction === 'Deepfake' ? 'border-red-400/50 bg-red-900/20 text-red-300' : 'border-green-400/50 bg-green-900/20 text-green-300'}`}>
                                    {analysisResult.prediction === 'Deepfake' ? 'Deepfake Signature' : 'Authentic Signature'}
                                </Badge>
                                <div className={`text-xl font-bold uppercase glow-text ${analysisResult.prediction === 'Deepfake' ? 'text-red-400' : 'text-green-400'}`}>
                                    DETECTED
                                </div>
                                <div className="relative flex items-center justify-center pt-4">
                                    <div className="circular-scanner">
                                        <div className="scanner-ring ring-1"></div>
                                        <div className="scanner-ring ring-2"></div>
                                        <div className="scanner-ring ring-3"></div>
                                        <div className="scanner-center">
                                            <span className="text-4xl font-bold text-cyan-300">{analysisResult.confidence.toFixed(0)}%</span>
                                            <div className="text-sm text-cyan-400/70">CONFIDENCE</div>
                                        </div>
                                    </div>
                                </div>
                                {analysisResult.prediction === 'Deepfake' && analysisResult.transformedVideoPath && (
                                    <a
                                      href={`http://localhost:5000/${analysisResult.transformedVideoPath}`}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="futuristic-button mt-4 !bg-red-900/30 !border-red-500/50 hover:!bg-red-900/60"
                                    >
                                        <Download className="h-4 w-4 mr-2"/>
                                        VIEW ANNOTATED VIDEO
                                    </a>
                                )}
                            </div>
                        )}
                        {!analysisResult && !isLoading && <div className="flex items-center justify-center h-full text-center text-cyan-400/50">Awaiting analysis initiation...</div>}
                    </div>
                </div>
            </div>

            <div className="col-span-2">
                <div className="panel-container h-full">
                    <div className="panel h-full">
                        <div className="corner-brackets"></div>
                        <h2 className="panel-title">GRAD-CAM FEED<span className="blinking-cursor">_</span></h2>
                        <p className="text-cyan-400/70 text-sm mb-4">
                            {highlightedBin ? `Showing frames with scores in range: ${highlightedBin}` : "Red areas indicate regions the model focused on for its prediction."}
                        </p>
                        <div className="horizontal-scroll">
                        {analysisResult && analysisResult.prediction === "Deepfake" && filteredGradCamData.length > 0 ? (
                            filteredGradCamData.map((item, i) => (
                                <div key={i} className="evidence-frame relative">
                                    <img src={`data:image/jpeg;base64,${item.image}`} alt={`Grad-CAM Evidence ${item.frame}`} className="w-full h-full object-cover" />
                                    <div className="absolute bottom-1 right-1 bg-black/50 text-white text-xs px-1 rounded">
                                        F:{item.frame} S:{item.score.toFixed(2)}
                                    </div>
                                </div>
                            ))
                        ) : (
                            <div className="flex items-center justify-center h-full text-cyan-400/50">
                                {analysisResult && analysisResult.prediction !== "Deepfake" ? "Grad-CAM is only generated for Deepfake detections." : "No visual evidence available for this selection."}
                            </div>
                        )}
                        </div>
                    </div>
                </div>
            </div>

            <div className="col-span-3 grid grid-cols-1 md:grid-cols-3 gap-6">
                <MetricCard icon={BarChart3} title="Avg. Score" value={analysisResult ? analysisResult.summaryStats.average.toFixed(3) : "---"} color="cyan"/>
                <MetricCard icon={TrendingUp} title="Max. Score (Highest Threat)" value={analysisResult ? analysisResult.summaryStats.highest.toFixed(3) : "---"} color="red" />
                <MetricCard icon={TrendingDown} title="Min. Score (Lowest Threat)" value={analysisResult ? analysisResult.summaryStats.lowest.toFixed(3) : "---"} color="green" />
            </div>

            {analysisResult && (
                <div className="col-span-3 grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className={`panel-container ${selectedType === 'Video' ? 'md:col-span-2' : ''}`}>
                        <div className="panel">
                            <div className="corner-brackets"></div>
                            <h2 className="panel-title flex items-center gap-2"><BarIcon className="h-5 w-5 text-cyan-400"/>SCORE DISTRIBUTION</h2>
                            <div className="chart-placeholder h-60">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={scoreDistributionData} margin={{ top: 5, right: 20, left: -10, bottom: 5 }} onClick={() => setHighlightedBin(null)}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(72, 187, 255, 0.2)" />
                                        <XAxis dataKey="name" stroke="rgba(72, 187, 255, 0.7)" fontSize={12} />
                                        <YAxis stroke="rgba(72, 187, 255, 0.7)" fontSize={12} />
                                        <Tooltip content={<CustomTooltip />} cursor={{fill: 'rgba(72, 187, 255, 0.1)'}}/>
                                        <Bar dataKey="count" onClick={(data, index, e) => {
                                            e.stopPropagation();
                                            setHighlightedBin(prev => (prev === data.name ? null : data.name));
                                        }}>
                                          {scoreDistributionData.map((entry, index) => {
                                            const score = parseFloat(entry.name.split('-')[0]);
                                            const isHighlighted = highlightedBin === entry.name;
                                            const isDimmed = highlightedBin && !isHighlighted;
                                            return <Cell key={`cell-${index}`}
                                                         fill={score >= 0.5 ? "rgba(248, 113, 113, 0.6)" : "rgba(74, 222, 128, 0.6)"}
                                                         opacity={isDimmed ? 0.3 : 1}
                                                         className={!isDimmed ? "cursor-pointer" : ""}
                                                         />;
                                          })}
                                        </Bar>
                                        <Brush dataKey="name" height={20} stroke="#00ffff" fill="rgba(72, 187, 255, 0.1)" tickFormatter={() => ''} />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    </div>
                    
                    {selectedType === 'Image' && (
                        <>
                            {hasMeaningfulFeatureMetrics ? (
                                <div className="col-span-1 panel-container">
                                    <div className="panel">
                                        <div className="corner-brackets"></div>
                                        <h2 className="panel-title flex items-center gap-2"><AreaIcon className="h-5 w-5 text-cyan-400"/>FEATURE METRICS PROFILE</h2>
                                        <div className="chart-placeholder h-60">
                                            <ResponsiveContainer width="100%" height="100%">
                                                <AreaChart data={featureLineData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                                                    <defs>
                                                        <linearGradient id="colorFeature" x1="0" y1="0" x2="0" y2="1">
                                                            <stop offset="5%" stopColor="#00ffff" stopOpacity={0.8}/>
                                                            <stop offset="95%" stopColor="#00ffff" stopOpacity={0}/>
                                                        </linearGradient>
                                                    </defs>
                                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(72, 187, 255, 0.2)" />
                                                    <XAxis
                                                        type="number"
                                                        dataKey="index"
                                                        name="Feature Index"
                                                        stroke="rgba(72, 187, 255, 0.7)"
                                                        fontSize={12}
                                                        domain={['dataMin', 'dataMax']}
                                                        tickFormatter={(tick) => `F${tick}`}
                                                        interval={Math.ceil(featureLineData.length / 10)}
                                                    />
                                                    <YAxis
                                                        type="number"
                                                        dataKey="value"
                                                        name="Feature Value"
                                                        stroke="rgba(72, 187, 255, 0.7)"
                                                        fontSize={12}
                                                        domain={['dataMin', 'dataMax']}
                                                    />
                                                    <Tooltip cursor={{ strokeDasharray: '3 3', stroke: 'rgba(72, 187, 255, 0.5)' }} content={<CustomTooltip />} />
                                                    {analysisResult && analysisResult.featureThreshold && (
                                                        <ReferenceLine
                                                            y={analysisResult.featureThreshold}
                                                            label={{ value: 'Deepfake Threshold', position: 'insideTopRight', fill: '#f87171', fontSize: 10 }}
                                                            stroke="#f87171"
                                                            strokeDasharray="4 4"
                                                        />
                                                    )}
                                                    <Area
                                                        type="monotone"
                                                        dataKey="value"
                                                        stroke="#00ffff"
                                                        strokeWidth={2}
                                                        activeDot={{ r: 8, fill: '#00ffff', stroke: '#fff', strokeWidth: 2 }}
                                                        dot={{ r: 4, fill: '#00ffff', stroke: '#00ffff', strokeWidth: 1 }}
                                                        fillOpacity={1}
                                                        fill="url(#colorFeature)"
                                                    />
                                                </AreaChart>
                                            </ResponsiveContainer>
                                        </div>
                                    </div>
                                </div>
                            ) : (
                                <div className="col-span-1 panel-container">
                                    <div className="panel">
                                        <div className="corner-brackets"></div>
                                        <h2 className="panel-title flex items-center gap-2"><AreaIcon className="h-5 w-5 text-cyan-400"/>FEATURE METRICS PROFILE</h2>
                                        <div className="flex items-center justify-center h-full text-center text-cyan-400/50">
                                            No distinct feature metrics identified for this input.
                                        </div>
                                    </div>
                                </div>
                            )}
                        </>
                    )}
                </div>
            )}

            {analysisResult && selectedType === "Video" && (
                <div className="col-span-3 panel-container">
                    <div className="panel">
                        <div className="corner-brackets"></div>
                        <h2 className="panel-title flex items-center gap-2"><AreaIcon className="h-5 w-5 text-cyan-400"/>SCORE DENSITY - FRAME-BY-FRAME ANALYSIS</h2>
                        <div className="chart-placeholder h-72">
                            <ResponsiveContainer width="100%" height="100%">
                                 <AreaChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }} onClick={(e) => {
                                     if (e && e.activePayload && e.activePayload.length > 0) {
                                         const frameData = e.activePayload[0].payload;
                                         const frameIndex = frameData.frame - 1;
                                         if (analysisResult?.gradcamImages?.[frameIndex]) {
                                             setFocusedFrame({
                                                 frame: frameData.frame,
                                                 score: frameData.score,
                                                 camImage: analysisResult.gradcamImages[frameIndex]
                                             });
                                         }
                                     }
                                 }}>
                                     <defs>
                                        <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#00ffff" stopOpacity={0.8}/>
                                            <stop offset="95%" stopColor="#00ffff" stopOpacity={0}/>
                                        </linearGradient>
                                     </defs>
                                     <CartesianGrid strokeDasharray="3 3" stroke="rgba(72, 187, 255, 0.2)" />
                                     <XAxis dataKey="frame" stroke="rgba(72, 187, 255, 0.7)" fontSize={12}/>
                                     <YAxis stroke="rgba(72, 187, 255, 0.7)" fontSize={12} domain={[0, 1]}/>
                                     <Tooltip content={<CustomTooltip />} />
                                     <ReferenceLine y={0.5} label={{ value: 'Detection Threshold', position: 'top', fill: '#f87171' }} stroke="#f87171" strokeDasharray="4 4" />
                                     <Area type="monotone" dataKey="score" stroke="#00ffff" fillOpacity={1} fill="url(#colorScore)" dot={<CustomizedDot />} activeDot={{ r: 8 }}/>
                                     <Brush dataKey="frame" height={20} stroke="#00ffff" fill="rgba(72, 187, 255, 0.1)" />
                                 </AreaChart>
                             </ResponsiveContainer>
                         </div>
                    </div>
                </div>
            )}
        </div>
      </main>

      <style jsx>{`
        @keyframes fade-in-down {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        .animate-fade-in-down {
            animation: fade-in-down 0.5s ease-out forwards;
        }
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
          height: 100%;
          transition: all 0.3s ease;
        }
        .panel-container:hover .panel {
            border-color: rgba(72, 187, 255, 0.5);
        }
        .corner-brackets::before, .corner-brackets::after {
          content: '';
          position: absolute;
          width: 20px; height: 20px;
          border: 2px solid rgba(72, 187, 255, 0.5);
          transition: all 0.3s ease;
        }
        .panel-container:hover .corner-brackets::before,
        .panel-container:hover .corner-brackets::after {
            border-color: rgba(72, 187, 255, 1);
        }
        .corner-brackets::before { top: -6px; left: -6px; border-right: none; border-bottom: none; }
        .corner-brackets::after { bottom: -6px; right: -6px; border-left: none; border-top: none; }
        .panel-title {
          color: rgb(103, 232, 249);
          font-size: 1rem;
          font-weight: bold;
          text-transform: uppercase;
          letter-spacing: 0.1em;
          margin-bottom: 1rem;
          text-shadow: 0 0 5px rgba(72, 187, 255, 0.5);
        }
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
        .futuristic-button:disabled { cursor: not-allowed; opacity: 0.5; }
        .status-display {
          background: rgba(0, 0, 0, 0.5);
          border: 1px solid rgba(72, 187, 255, 0.2);
          border-radius: 4px; padding: 0.75rem;
        }
        .circular-scanner { position: relative; width: 150px; height: 150px; display: flex; align-items: center; justify-content: center; }
        .scanner-ring {
          position: absolute;
          border: 2px dashed rgba(72, 187, 255, 0.3);
          border-radius: 50%;
          animation: rotate 4s linear infinite;
        }
        .ring-1 { width: 120px; height: 120px; animation-duration: 3s; }
        .ring-2 { width: 140px; height: 140px; animation-duration: 4s; animation-direction: reverse; }
        .ring-3 { width: 160px; height: 160px; animation-duration: 5s; }
        .scanner-center { text-align: center; z-index: 10; }
        @keyframes rotate { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        .blinking-cursor { animation: blink 1s infinite; }
        @keyframes blink { 0%, 50% { opacity: 1; } 51%, 100% { opacity: 0; } }
        .horizontal-scroll { display: flex; gap: 1rem; overflow-x: auto; padding-bottom: 1rem; height: calc(100% - 70px); }
        .evidence-frame {
          flex-shrink: 0; width: 150px; height: 150px;
          border: 1px solid rgba(72, 187, 255, 0.4);
          border-radius: 4px; overflow: hidden;
          transition: all 0.3s ease;
          background-color: rgba(0,0,0,0.3);
        }
        .evidence-frame:hover { border-color: rgba(72, 187, 255, 0.9); transform: scale(1.05); }
        .chart-placeholder { height: 300px; position: relative; }
        .metric-card .panel {
            overflow: hidden;
        }
        .recharts-text {
            fill: rgba(148, 163, 184, 0.9) !important;
        }
        .recharts-legend-item-text {
            color: rgba(255, 255, 255, 0.9) !important;
        }
        .recharts-surface {
            cursor: pointer;
        }
      `}</style>
    </div>
  )
}
